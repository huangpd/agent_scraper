"""RuleDiscoverer: LLM 分析页面结构，寻找遍历规则。具备强大的 HTML 压缩能力。"""

import json
import logging
import re
from bs4 import BeautifulSoup

from agent_scraper.core.llm import LLMService
from agent_scraper.core.models import PageRules
from agent_scraper.pipeline.prompts import DISCOVER_PROMPT, DISCOVER_RETRY_PROMPT

logger = logging.getLogger(__name__)

MAX_HTML_SIZE = 50 * 1024

class RuleDiscoverer:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    async def discover(
        self, html: str, current_url: str = "", traversal_hints: list[str] | None = None
    ) -> PageRules:
        if not traversal_hints:
            logger.info("用户未要求遍历，单页模式")
            return PageRules()

        snippet = self._get_clean_snippet(html)
        
        prompt = DISCOVER_PROMPT.format(
            current_url=current_url,
            requested_modes=", ".join(traversal_hints),
            html_snippet=snippet,
        )

        try:
            content = await self.llm_service.call(prompt, caller="RuleDiscoverer")
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            data = json.loads(content)

            filtered = {}
            if "load_more" in traversal_hints:
                filtered["load_more_selector"] = data.get("load_more_selector")
            if "next_button" in traversal_hints:
                filtered["next_button_selector"] = data.get("next_button_selector")
            if "pagination" in traversal_hints:
                filtered["pagination_url"] = data.get("pagination_url")
                filtered["pagination_max"] = data.get("pagination_max")
            if "sub_pages" in traversal_hints:
                filtered["sub_page_selector"] = data.get("sub_page_selector")
                filtered["sub_page_url_attr"] = data.get("sub_page_url_attr", "href")
                filtered["sub_page_url_filter"] = data.get("sub_page_url_filter")
                filtered["sub_page_recursive"] = data.get("sub_page_recursive", False)

            rules = PageRules(**{k: v for k, v in filtered.items() if v is not None})
            self._log_rules(rules)
            return rules
        except Exception as e:
            logger.error("RuleDiscoverer 失败: %s", e)
            return PageRules()

    async def discover_retry(
        self, html: str, current_url: str = "",
        traversal_hints: list[str] | None = None, missing_modes: list[str] | None = None,
    ) -> PageRules:
        """用更强硬的重试提示词重新发现缺失的遍历规则"""
        if not traversal_hints:
            return PageRules()

        snippet = self._get_clean_snippet(html)
        missing_desc = ", ".join(missing_modes or traversal_hints)

        prompt = DISCOVER_RETRY_PROMPT.format(
            missing_modes=missing_desc,
            current_url=current_url,
            requested_modes=", ".join(traversal_hints),
            html_snippet=snippet,
        )

        try:
            content = await self.llm_service.call(prompt, caller="RuleDiscoverer.retry")
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            data = json.loads(content)

            filtered = {}
            if "load_more" in traversal_hints:
                filtered["load_more_selector"] = data.get("load_more_selector")
            if "next_button" in traversal_hints:
                filtered["next_button_selector"] = data.get("next_button_selector")
            if "pagination" in traversal_hints:
                filtered["pagination_url"] = data.get("pagination_url")
                filtered["pagination_max"] = data.get("pagination_max")
            if "sub_pages" in traversal_hints:
                filtered["sub_page_selector"] = data.get("sub_page_selector")
                filtered["sub_page_url_attr"] = data.get("sub_page_url_attr", "href")
                filtered["sub_page_url_filter"] = data.get("sub_page_url_filter")
                filtered["sub_page_recursive"] = data.get("sub_page_recursive", False)

            rules = PageRules(**{k: v for k, v in filtered.items() if v is not None})
            self._log_rules(rules)
            return rules
        except Exception as e:
            logger.error("RuleDiscoverer 重试失败: %s", e)
            return PageRules()

    @staticmethod
    def validate_selectors(html: str, rules: PageRules) -> list[str]:
        """校验 LLM 返回的 CSS selector 是否在 HTML 中真的有匹配元素（防幻觉）。
        返回无效的字段名列表，如 ['load_more_selector', 'sub_page_selector']。
        """
        soup = BeautifulSoup(html, "lxml")
        invalid = []
        for field_name, selector in [
            ("load_more_selector", rules.load_more_selector),
            ("next_button_selector", rules.next_button_selector),
            ("sub_page_selector", rules.sub_page_selector),
        ]:
            if not selector:
                continue
            try:
                if not soup.select(selector):
                    invalid.append(field_name)
            except Exception:
                # selector 语法本身有误
                invalid.append(field_name)
        return invalid

    @staticmethod
    def _log_rules(rules: PageRules):
        found = []
        if rules.load_more_selector:
            found.append(f"load_more: {rules.load_more_selector}")
        if rules.next_button_selector:
            found.append(f"next_button: {rules.next_button_selector}")
        if rules.pagination_url:
            found.append(f"pagination: {rules.pagination_url} (max={rules.pagination_max})")
        if rules.sub_page_selector:
            found.append(f"sub_pages: {rules.sub_page_selector} (recursive={rules.sub_page_recursive})")
        if found:
            logger.info("发现 %d 条规则:", len(found))
            for r in found:
                logger.info("  - %s", r)
        else:
            logger.info("未找到匹配的遍历规则")

    @staticmethod
    def _get_clean_snippet(html: str) -> str:
        """精简 HTML：去噪 + 压缩重复列表项，只保留导航骨架。"""
        soup = BeautifulSoup(html, "lxml")

        # 1. 去掉无用标签
        for tag in soup.find_all(["script", "style", "noscript", "svg", "img", "picture", "video"]):
            tag.decompose()

        # 2. 去掉内联样式和 data-* 属性（减少噪声）
        for tag in soup.find_all(True):
            remove_attrs = [a for a in tag.attrs if a.startswith("data-") or a == "style"]
            for a in remove_attrs:
                del tag[a]

        main = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|container", re.I))
            or soup.body
            or soup
        )

        # 3. 压缩重复列表项：如果 <ul>/<ol>/<tbody> 有 >5 个同类子项，只保留前3+后1
        for container in main.find_all(["ul", "ol", "tbody", "div"]):
            children = [c for c in container.children if hasattr(c, "name") and c.name]
            if len(children) > 5:
                tag_names = [c.name for c in children]
                most_common = max(set(tag_names), key=tag_names.count)
                same_tag = [c for c in children if c.name == most_common]
                if len(same_tag) > 5:
                    keep_head = same_tag[:3]
                    keep_tail = [same_tag[-1]]
                    removed_count = len(same_tag) - 4
                    for item in same_tag:
                        if item not in keep_head and item not in keep_tail:
                            item.decompose()
                    placeholder = soup.new_string(f"\n<!-- ... 省略 {removed_count} 个同类元素 ... -->\n")
                    keep_head[-1].insert_after(placeholder)

        content = str(main)
        if len(content) > MAX_HTML_SIZE:
            content = content[:MAX_HTML_SIZE]
        logger.info("HTML 精简: %.0fKB → %.0fKB", len(html)/1024, len(content)/1024)
        return content
