import json
import logging
from html import unescape

from openai import AsyncOpenAI
from bs4 import BeautifulSoup

from agent_scraper.core.llm import create_openai_client, get_model_name
from agent_scraper.core.models import ExtractionGoal
from autoscraper.auto_scraper import AutoScraper
from autoscraper.utils import normalize

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

输出格式（严格JSON）：
{{
  "字段名1": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}},
  "字段名2": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}}
}}
"""

class Extractor:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or create_openai_client()
        self.model = get_model_name()
        self._css_rule_cache: list[dict] = []
        self._trained_scraper: AutoScraper | None = None

    async def _llm_call(self, prompt: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    async def extract(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """优先使用 AutoScraper (ML)，只有失败时才用简单的 CSS LLM 兜底"""
        expected_fields = set(goal.fields.keys())

        # 1. AutoScraper 路径 (机器学习)
        if not self._trained_scraper:
            wanted_dict = goal.samples if goal.samples else {}
            if wanted_dict:
                scraper = AutoScraper()
                scraper.build(html=html, wanted_dict=wanted_dict)
                self._trained_scraper = scraper

        if self._trained_scraper:
            as_result = self._trained_scraper.get_result_similar(html=html, group_by_alias=True)
            if any(len(v) > 0 for v in as_result.values()):
                return as_result

        # 2. 简单 CSS 兜底
        return await self._css_selector_extract(html, goal, expected_fields)

    async def _css_selector_extract(self, html: str, goal: ExtractionGoal, expected_fields: set[str]) -> dict:
        # 与 AutoScraper._get_soup 保持一致的预处理链
        soup = BeautifulSoup(normalize(unescape(html)), "lxml")
        body = soup.body or soup
        snippet = str(body)[:20000]
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in goal.fields.items())
        prompt = CSS_SELECTOR_PROMPT.format(fields_desc=fields_desc, html_snippet=snippet)
        try:
            content = await self._llm_call(prompt)
            if "```" in content: content = content.split("```")[1].replace("json", "").strip()
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
