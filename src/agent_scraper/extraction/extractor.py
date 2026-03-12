"""核心提取模块：CSS Selector 缓存优先 + AutoScraper + 三级降级
第一页 LLM 生成 CSS 选择器并缓存，后续页面直接复用，零 LLM 调用。
"""

import json
import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from agent_scraper.core.llm import create_openai_client, get_model_name
from agent_scraper.core.models import ExtractionGoal
from autoscraper import AutoScraper

logger = logging.getLogger(__name__)

# ── LLM 采样 Prompt ──────────────────────────────────────

SAMPLE_PROMPT = """\
你是一个数据提取专家。从下面的 HTML 片段中，为每个字段提取 2-3 个真实样本值。

要提取的字段：
{fields_desc}

HTML 片段（截取自页面主内容区域）：
```html
{html_snippet}
```

要求：
1. 每个字段提取 2-3 个 **真实存在于 HTML 中的** 样本值
2. 样本必须是 HTML 中的原始文本或属性值，不能自己编造
3. 对于 URL 类字段，提取 href 属性的完整值（包含相对路径）
4. 选择页面中不同位置的样本，以确保规则泛化

输出格式（严格JSON，不要多余文字）：
{{
  "字段名1": ["样本1", "样本2"],
  "字段名2": ["样本1", "样本2"]
}}
"""

# ── CSS Selector Prompt ──────────────────────────────────

CSS_SELECTOR_PROMPT = """\
你是一个前端专家。分析下面的 HTML，为每个字段生成 CSS 选择器来提取数据。

要提取的字段：
{fields_desc}

HTML 片段：
```html
{html_snippet}
```

要求：
1. 每个选择器应能匹配页面中 **所有** 同类元素（不是只匹配一个）
2. 对于文本内容，选择器指向包含文本的元素
3. 对于 URL，选择器指向包含 href 的 <a> 标签
4. 说明是提取 text 还是某个属性（如 href）
5. 选择器要**通用**，能在结构相似的不同页面上复用

输出格式（严格JSON）：
{{
  "字段名1": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}},
  "字段名2": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}}
}}
"""

MAX_HTML_SIZE = 300 * 1024  # 300KB


class Extractor:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or create_openai_client()
        self.model = get_model_name()
        # 缓存：一次 LLM 生成，所有页面复用
        self._cached_css_selectors: dict | None = None
        self._trained_scraper: AutoScraper | None = None

    async def _llm_call(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    async def extract(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """
        混合提取策略：AutoScraper ML 提取到的保留，缺失字段用 CSS Selector 补齐。
          1. 缓存的 CSS 选择器（第 2+ 页，零 LLM）
          2. AutoScraper ML（擅长文本字段）
          3. CSS Selector 补齐缺失字段（擅长 URL/属性字段）
        """
        expected_fields = set(goal.fields.keys())

        # ── 快速路径：用缓存的 CSS 选择器（零 LLM 调用）──
        if self._cached_css_selectors:
            result = self._apply_css_selectors(html, self._cached_css_selectors)
            if result and self._validate_result(result, expected_fields):
                logger.info("缓存 CSS 选择器命中: %s", {k: len(v) for k, v in result.items()})
                return result
            logger.info("缓存选择器失败，重新提取...")

        # ── AutoScraper 路径 ──
        as_result = {}
        if self._trained_scraper:
            as_result = self._apply_trained_scraper(html)
            if as_result and self._validate_result(as_result, expected_fields):
                logger.info("复用 AutoScraper 规则成功: %s", {k: len(v) for k, v in as_result.items()})
                return as_result

        if not as_result:
            if goal.samples:
                logger.info("使用用户样本训练 AutoScraper...")
                wanted_dict = self._normalize_url_samples(goal.samples)
            else:
                logger.info("LLM 采样生成样本...")
                wanted_dict = await self._llm_sample(html, goal)
                if not wanted_dict or not any(wanted_dict.values()):
                    return await self._css_selector_extract(html, goal, expected_fields)

            as_result = self._autoscraper_extract(html, wanted_dict)

        # 完整提取成功 → 直接返回
        if as_result and self._validate_result(as_result, expected_fields):
            logger.info("AutoScraper 完整提取成功: %s", {k: len(v) for k, v in as_result.items()})
            return as_result

        # ── 混合路径：AutoScraper 部分成功 + CSS Selector 补缺 ──
        got_fields = {k for k, v in as_result.items() if v} if as_result else set()
        missing_fields = expected_fields - got_fields

        if got_fields and missing_fields:
            logger.info("AutoScraper 部分成功: %s，CSS 补齐: %s", got_fields, missing_fields)
            css_result = await self._css_selector_for_missing(html, goal, missing_fields)
            if css_result:
                merged = {**as_result, **css_result}
                if self._validate_result(merged, expected_fields):
                    logger.info("混合提取成功: %s", {k: len(v) for k, v in merged.items()})
                    return merged

        # ── 全量 CSS Selector 兜底 ──
        return await self._css_selector_extract(html, goal, expected_fields)

    # ── CSS Selector：生成 / 缓存 / 应用 ────────────────

    async def _css_selector_extract(
        self, html: str, goal: ExtractionGoal, expected_fields: set[str]
    ) -> dict[str, list]:
        """LLM 生成 CSS 选择器，提取数据，成功后缓存选择器"""
        logger.info("LLM 生成 CSS 选择器（将缓存供后续页面复用）...")
        selectors = await self._generate_css_selectors(html, goal)
        if not selectors:
            return {}

        result = self._apply_css_selectors(html, selectors)

        if result and self._validate_result(result, expected_fields):
            # 缓存成功的选择器，后续页面不再调用 LLM
            self._cached_css_selectors = selectors
            logger.info("CSS 选择器提取成功并已缓存: %s", {k: len(v) for k, v in result.items()})
        elif result:
            logger.warning("CSS 选择器提取结果（未通过验证）: %s", {k: len(v) for k, v in result.items()})

        return result or {}

    async def _css_selector_for_missing(
        self, html: str, goal: ExtractionGoal, missing_fields: set[str]
    ) -> dict[str, list]:
        """只为缺失的字段生成 CSS 选择器（复用缓存中已有的）"""
        # 先检查缓存里是否已有这些字段的选择器
        if self._cached_css_selectors:
            cached_missing = {k: v for k, v in self._cached_css_selectors.items() if k in missing_fields}
            if cached_missing:
                result = self._apply_css_selectors(html, cached_missing)
                has_data = {k for k, v in result.items() if v}
                if missing_fields.issubset(has_data):
                    logger.info("从缓存 CSS 补齐: %s", list(cached_missing.keys()))
                    return result

        # 缓存没有，LLM 生成（只生成缺失字段的选择器）
        missing_goal_fields = {k: v for k, v in goal.fields.items() if k in missing_fields}
        snippet = self._get_main_content_snippet(html)
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in missing_goal_fields.items())

        prompt = CSS_SELECTOR_PROMPT.format(fields_desc=fields_desc, html_snippet=snippet)

        try:
            content = await self._llm_call(prompt)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            selectors = json.loads(content)
        except Exception as e:
            logger.error("CSS 补齐生成失败: %s", e)
            return {}

        result = self._apply_css_selectors(html, selectors)

        # 将新选择器合并到缓存
        if self._cached_css_selectors is None:
            self._cached_css_selectors = {}
        self._cached_css_selectors.update(selectors)
        logger.info("CSS 补齐结果: %s（已缓存）", {k: len(v) for k, v in result.items()})

        return result

    async def _generate_css_selectors(self, html: str, goal: ExtractionGoal) -> dict | None:
        """LLM 生成 CSS 选择器规则"""
        snippet = self._get_main_content_snippet(html)
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in goal.fields.items())

        prompt = CSS_SELECTOR_PROMPT.format(
            fields_desc=fields_desc,
            html_snippet=snippet,
        )

        try:
            content = await self._llm_call(prompt)

            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            selectors = json.loads(content)
            logger.info("LLM 生成选择器: %s", json.dumps(selectors, ensure_ascii=False))
            return selectors
        except Exception as e:
            logger.error("CSS 选择器生成失败: %s", e)
            return None

    @staticmethod
    def _apply_css_selectors(html: str, selectors: dict) -> dict[str, list]:
        """用 CSS 选择器从 HTML 提取数据（纯代码，无 LLM）"""
        soup = BeautifulSoup(html, "lxml")
        result = {}

        for field_name, sel_info in selectors.items():
            selector = sel_info.get("selector", "")
            attr = sel_info.get("attr", "text")

            try:
                elements = soup.select(selector)
            except Exception:
                elements = []

            values = []
            for el in elements:
                if attr == "text":
                    val = el.get_text(strip=True)
                else:
                    val = el.get(attr, "")
                if val:
                    values.append(str(val))

            result[field_name] = values

        return result

    # ── AutoScraper ──────────────────────────────────────

    def _autoscraper_extract(self, html: str, wanted_dict: dict[str, list]) -> dict[str, list]:
        scraper = AutoScraper()
        try:
            build_result = scraper.build(html=html, wanted_dict=wanted_dict)
            if isinstance(build_result, dict) and build_result:
                self._trained_scraper = scraper
                return build_result

            result = scraper.get_result_similar(html=html, group_by_alias=True)
            if isinstance(result, dict) and result:
                self._trained_scraper = scraper
                return result
        except Exception as e:
            logger.error("AutoScraper 出错: %s", e)
        return {}

    def _apply_trained_scraper(self, html: str) -> dict[str, list]:
        try:
            result = self._trained_scraper.get_result_similar(html=html, group_by_alias=True)
            if isinstance(result, dict):
                return result
        except Exception as e:
            logger.error("复用规则出错: %s", e)
        return {}

    # ── LLM 采样 ────────────────────────────────────────

    async def _llm_sample(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        snippet = self._get_main_content_snippet(html)
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in goal.fields.items())

        prompt = SAMPLE_PROMPT.format(
            fields_desc=fields_desc,
            html_snippet=snippet,
        )

        try:
            content = await self._llm_call(prompt)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            wanted_dict = json.loads(content)
            cleaned = {}
            for key, values in wanted_dict.items():
                if isinstance(values, list) and values:
                    cleaned[key] = [str(v) for v in values if v]
            return cleaned
        except Exception as e:
            logger.error("LLM 采样出错: %s", e)
            return {}

    # ── 工具方法 ─────────────────────────────────────────

    def _get_main_content_snippet(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()

        main = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|container", re.I))
            or soup.body
            or soup
        )

        content = str(main)
        if len(content) > MAX_HTML_SIZE:
            content = content[:MAX_HTML_SIZE]
        return content

    @staticmethod
    def _normalize_url_samples(samples: dict[str, list[str]]) -> dict[str, list[str]]:
        result = {}
        for key, values in samples.items():
            expanded = list(values)
            for v in values:
                if v.startswith("http"):
                    path = urlparse(v).path
                    if path and path not in expanded:
                        expanded.append(path)
            result[key] = expanded
        return result

    @staticmethod
    def _validate_result(result: dict[str, list], expected_fields: set[str] | None = None) -> bool:
        if not result:
            return False

        if expected_fields:
            missing = expected_fields - set(result.keys())
            if missing:
                logger.warning("验证失败: 缺少字段 %s", missing)
                return False

        lengths = [len(v) for v in result.values()]
        if any(l == 0 for l in lengths):
            return False

        if len(lengths) > 1:
            max_len = max(lengths)
            min_len = min(lengths)
            if min_len == 0 or max_len / min_len > 2:
                return False

        return True
