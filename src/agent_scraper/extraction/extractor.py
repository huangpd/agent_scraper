import json
import logging
from html import unescape
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

        # 1. AutoScraper 路径 (机器学习)
        if not self._trained_scraper:
            wanted_dict = goal.samples if goal.samples else {}
            if wanted_dict:
                scraper = AutoScraper()
                # 使用规范化后的 HTML 进行训练
                scraper.build(html=normalized_html, wanted_dict=wanted_dict)
                # URL 字段：从文本字段 XPath 推导 /@href
                self._derive_url_rules(scraper, goal)
                self._trained_scraper = scraper
        if self._trained_scraper:
            # 同样使用规范化后的 HTML 进行提取
            as_result = self._trained_scraper.get_result_similar(html=normalized_html, group_by_alias=True)
            if any(len(v) > 0 for v in as_result.values()):
                return as_result

        # 2. 简单 CSS 兜底（带缓存）
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
            return self._apply_css_selectors(html, selectors)
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
