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

    async def extract(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """优先使用 AutoScraper (ML)，只有失败时才用简单的 CSS LLM 兜底"""
        expected_fields = set(goal.fields.keys())

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

        return await self._css_selector_extract(html, goal, expected_fields)

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
