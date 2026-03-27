import json
import logging
from html import unescape
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from agent_scraper.core.llm import LLMService
from agent_scraper.core.models import ExtractionGoal
from autoscraper.auto_scraper import AutoScraper
from autoscraper.utils import normalize
from agent_scraper.pipeline.prompts import SAMPLE_PROMPT, CSS_SELECTOR_PROMPT

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()
        self._css_rule_cache: list[dict] = []
        self._trained_scraper: AutoScraper | None = None

    async def _llm_call(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def _clean_html(self, html: str) -> str:
        """结构化清洗：剔除 JS、CSS、SVG 等干扰项，保留核心数据结构"""
        if not html:
            return ""

        # 预处理转义字符
        html = unescape(html)
        soup = BeautifulSoup(html, "lxml")

        # 1. 剔除完全无关的标签
        # script: JS代码 | style: CSS样式 | svg: 图标代码 | canvas: 绘图
        # meta/link: 元数据 | noscript: 无脚本备份
        for tag in soup(["script", "style", "svg", "canvas", "meta", "link", "noscript"]):
            tag.decompose()

        # 2. 移除注释
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 3. 仅保留 body 内容（如果存在）
        content = soup.body if soup.body else soup

        # 4. 使用 autoscraper 官方规范化函数进一步处理
        return normalize(str(content))

    async def extract(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """优先使用 AutoScraper (ML)，只有失败时才用简单的 CSS LLM 兜底"""
        expected_fields = set(goal.fields.keys())

        # 执行深度清洗：剔除 JS/CSS/SVG 等
        normalized_html = self._clean_html(html)

        has_samples = bool(goal.samples)

        # 1. AutoScraper 路径 (机器学习)
        if not self._trained_scraper:
            wanted_dict = goal.samples if goal.samples else {}
            if wanted_dict:
                # 展开 URL 样本：完整 URL → 同时包含相对路径版本，方便匹配 HTML 中的 href
                wanted_dict = self._expand_url_samples(wanted_dict)
                logger.info("[Extractor] 策略: AutoScraper (ML) — 有样本 %s",
                            {k: len(v) for k, v in wanted_dict.items()})
                scraper = AutoScraper()
                # 使用规范化后的 HTML 进行训练
                scraper.build(html=normalized_html, wanted_dict=wanted_dict)
                # 打印学到的 XPath 规则
                rules = scraper.get_result_xpath_rule()
                if rules:
                    logger.info("[Extractor] AutoScraper 训练完成，学到 %d 条 XPath 规则:", len(rules))
                    for alias, xpath in rules.items():
                        logger.info("[Extractor]   %s → %s", alias, xpath)
                else:
                    logger.warning("[Extractor] AutoScraper 训练完成，但未学到任何 XPath 规则")
                # URL 字段：从文本字段 XPath 推导 /@href
                self._derive_url_rules(scraper, goal)
                self._trained_scraper = scraper
            else:
                logger.info("[Extractor] 无样本，跳过 AutoScraper")

        if self._trained_scraper:
            # 同样使用规范化后的 HTML 进行提取
            as_result = self._trained_scraper.get_result_similar(html=normalized_html, group_by_alias=True)
            result_counts = {k: len(v) for k, v in as_result.items()}
            total = sum(result_counts.values())
            if total > 0:
                logger.info("[Extractor] AutoScraper 提取成功: %s", result_counts)
                return as_result
            else:
                logger.warning("[Extractor] AutoScraper 提取结果为空，降级到 LLM CSS 选择器")

        # 2. 简单 CSS 兜底（带缓存）
        logger.info("[Extractor] 策略: LLM CSS 选择器 (兜底)")
        return await self._css_selector_extract(normalized_html, goal, expected_fields)

    # ── URL 字段 XPath 推导 ────────────────────────────────

    @staticmethod
    def _derive_url_rules(scraper: AutoScraper, goal: ExtractionGoal):
        """对 URL 字段，从关联文本字段的 XPath 推导 /@href 规则。

        思路：文本字段（如 file_name）和 URL 字段共享 DOM 结构，
        文本节点通常在 <a> 内部，截断 XPath 到 <a> 再加 /@href 即可。
        """
        rules = scraper.get_result_xpath_rule()
        if not rules:
            return

        # 找出无样本的 URL 字段
        url_fields = []
        _URL_INDICATORS = ("url", "链接", "link", "href")
        for field_name, field_desc in goal.fields.items():
            text = (field_name + " " + field_desc).lower()
            if any(kw in text for kw in _URL_INDICATORS):
                if not goal.samples or field_name not in goal.samples:
                    url_fields.append(field_name)

        if not url_fields:
            return

        # 选源 XPath：优先 title/name 类字段
        source_xpath = None
        _TITLE_KEYWORDS = ("title", "标题", "name", "名称")
        for alias, xpath in rules.items():
            combined = (alias + " " + goal.fields.get(alias, "")).lower()
            if any(kw in combined for kw in _TITLE_KEYWORDS):
                source_xpath = xpath
                break
        if not source_xpath:
            source_xpath = next(iter(rules.values()), None)

        if not source_xpath:
            return

        # 推导: 找 XPath 中的 <a> 段截断，或追加 /a/@href
        href_xpath = Extractor._xpath_to_href(source_xpath)

        for url_field in url_fields:
            scraper._xpath_overrides[url_field] = href_xpath
            logger.info("[Extractor] URL 字段 '%s' XPath 推导: %s", url_field, href_xpath)

    @staticmethod
    def _expand_url_samples(wanted_dict: dict[str, list[str]]) -> dict[str, list[str]]:
        """展开 URL 样本：用户提供完整 URL 时，追加相对路径版本。

        用户从浏览器复制的样本通常是完整 URL:
            https://example.com/data/file.csv?download=true
        但 HTML 中 href 往往是相对路径:
            /data/file.csv?download=true
        AutoScraper 需要匹配 HTML 中的实际文本，所以要把两个版本都给它。
        """
        expanded = {}
        for field, values in wanted_dict.items():
            seen = set(values)
            new_values = list(values)
            for val in values:
                if not isinstance(val, str) or not val.startswith("http"):
                    continue
                parsed = urlparse(val)
                # 相对路径 = path + query + fragment
                relative = parsed.path
                if parsed.query:
                    relative += "?" + parsed.query
                if parsed.fragment:
                    relative += "#" + parsed.fragment
                if relative and relative not in seen:
                    new_values.append(relative)
                    seen.add(relative)
            if len(new_values) > len(values):
                logger.info("[Extractor] URL 样本展开: %s 增加 %d 个相对路径",
                            field, len(new_values) - len(values))
            expanded[field] = new_values
        return expanded

    @staticmethod
    def _xpath_to_href(xpath: str) -> str | None:
        """从文本字段 XPath 推导 URL XPath。

        三种情况：
        1. 路径中有 <a>（祖先）: //div/a/span → //div/a/@href
        2. 末尾就是 <a>:         //div/a      → //div/a/@href
        3. 路径中无 <a>（子节点）: //div/h3     → //div/h3/a/@href
        """
        parts = xpath.split("/")
        last_a_idx = None
        for i, part in enumerate(parts):
            tag = part.split("[")[0]
            if tag == "a":
                last_a_idx = i

        if last_a_idx is not None:
            # <a> 在路径中：截断到 <a> + /@href
            return "/".join(parts[:last_a_idx + 1]) + "/@href"

        # <a> 不在路径中：作为叶子节点的子元素追加
        return xpath + "/a/@href"

    async def _css_selector_extract(self, html: str, goal: ExtractionGoal, expected_fields: set[str]) -> dict:
        soup = BeautifulSoup(normalize(unescape(html)), "lxml")
        body = soup.body or soup
        snippet = str(body)[:20000]
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in goal.fields.items())
        prompt = CSS_SELECTOR_PROMPT.format(fields_desc=fields_desc, html_snippet=snippet)
        try:
            content = await self.llm_service.call(prompt, caller="Extractor")
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            selectors = json.loads(content)
            logger.info("[Extractor] LLM 生成 CSS 选择器:")
            for field, info in selectors.items():
                logger.info("[Extractor]   %s → selector='%s', attr='%s'",
                            field, info.get("selector", "?"), info.get("attr", "text"))
            result = self._apply_css_selectors(html, selectors)
            result_counts = {k: len(v) for k, v in result.items()}
            logger.info("[Extractor] CSS 选择器提取结果: %s", result_counts)
            return result
        except Exception as e:
            logger.warning("CSS 选择器提取失败: %s", e)
            return {}

    @staticmethod
    def _apply_css_selectors(html: str, selectors: dict) -> dict[str, list]:
        soup = BeautifulSoup(html, "lxml")
        result = {}
        for field, info in selectors.items():
            sel = info.get("selector")
            attr = info.get("attr", "text")
            nodes = soup.select(sel)
            vals = []
            for n in nodes:
                val = n.get_text(strip=True) if attr == "text" else n.get(attr, "")
                if val: vals.append(val)
            result[field] = vals
        return result
