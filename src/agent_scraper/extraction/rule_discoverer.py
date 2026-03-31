"""RuleDiscoverer: LLM 分析页面结构，寻找遍历规则。"""

import logging

from agent_scraper.core.llm import LLMService, get_strong_model_name
from agent_scraper.core.models import PageRules
from agent_scraper.extraction.css_engine import (
    preprocess_html, validate_css, parse_json_response, CSS_SYSTEM_PROMPT,
)
from agent_scraper.pipeline.prompts import DISCOVER_PROMPT, DISCOVER_RETRY_PROMPT

logger = logging.getLogger(__name__)


class RuleDiscoverer:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    async def discover(
        self, html: str, current_url: str = "", traversal_hints: list[str] | None = None
    ) -> PageRules:
        if not traversal_hints:
            logger.info("用户未要求遍历，单页模式")
            return PageRules()

        snippet = preprocess_html(html)

        prompt = DISCOVER_PROMPT.format(
            current_url=current_url,
            requested_modes=", ".join(traversal_hints),
            html_snippet=snippet,
        )

        try:
            content = await self.llm_service.call(
                prompt,
                system_msg=CSS_SYSTEM_PROMPT,
                caller="RuleDiscoverer",
                model=get_strong_model_name(),
            )
            data = parse_json_response(content)
            rules = self._build_rules(data, traversal_hints)

            # 纯生成器：校验由 DiscoverRulesTool + OutputGuard 统一处理
            self._log_rules(rules)
            return rules
        except Exception as e:
            logger.error("RuleDiscoverer 失败: %s", e)
            return PageRules()

    async def discover_retry(
        self, html: str, current_url: str = "",
        traversal_hints: list[str] | None = None, missing_modes: list[str] | None = None,
        failed_attempts: list[dict] | None = None,
    ) -> PageRules:
        """用更强硬的重试提示词重新发现缺失的遍历规则

        Args:
            failed_attempts: OutputGuard 返回的失败详情列表，每项含:
                field: 字段名, selector: 被拒选择器, reason: 拒绝原因
        """
        if not traversal_hints:
            return PageRules()

        snippet = preprocess_html(html)
        missing_desc = ", ".join(missing_modes or traversal_hints)

        # 格式化失败反馈
        if failed_attempts:
            feedback_lines = []
            for fa in failed_attempts:
                feedback_lines.append(
                    f"- {fa['field']}: `{fa['selector']}` → 失败原因: {fa['reason']}"
                )
            failed_feedback = "\n".join(feedback_lines)
        else:
            failed_feedback = "（无具体失败记录）"

        prompt = DISCOVER_RETRY_PROMPT.format(
            missing_modes=missing_desc,
            failed_feedback=failed_feedback,
            current_url=current_url,
            requested_modes=", ".join(traversal_hints),
            html_snippet=snippet,
        )

        try:
            content = await self.llm_service.call(
                prompt,
                system_msg=CSS_SYSTEM_PROMPT,
                caller="RuleDiscoverer.retry",
                model=get_strong_model_name(),
            )
            data = parse_json_response(content)
            rules = self._build_rules(data, traversal_hints)
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
        invalid = []
        for field_name, selector in [
            ("load_more_selector", rules.load_more_selector),
            ("next_button_selector", rules.next_button_selector),
            ("sub_page_selector", rules.sub_page_selector),
        ]:
            if not selector:
                continue
            result = validate_css(html, selector)
            if not result["valid"]:
                invalid.append(field_name)
        return invalid

    @staticmethod
    def _build_rules(data: dict, traversal_hints: list[str]) -> PageRules:
        """从 LLM 返回的 JSON 构建 PageRules，只保留用户请求的模式。"""
        filtered = {}
        if "load_more" in traversal_hints:
            filtered["load_more_selector"] = data.get("load_more_selector")
        if "next_button" in traversal_hints:
            filtered["next_button_selector"] = data.get("next_button_selector")
            filtered["pagination_max"] = data.get("pagination_max")
        if "sub_pages" in traversal_hints:
            filtered["sub_page_selector"] = data.get("sub_page_selector")
            filtered["sub_page_url_attr"] = data.get("sub_page_url_attr", "href")
            filtered["sub_page_url_filter"] = data.get("sub_page_url_filter")
            filtered["sub_page_recursive"] = data.get("sub_page_recursive", False)
        return PageRules(**{k: v for k, v in filtered.items() if v is not None})

    @staticmethod
    def _invalid_to_modes(invalid_fields: list[str]) -> list[str]:
        """将无效字段名映射回遍历模式名。"""
        mapping = {
            "load_more_selector": "load_more",
            "next_button_selector": "next_button",
            "sub_page_selector": "sub_pages",
        }
        return [mapping[f] for f in invalid_fields if f in mapping]

    @staticmethod
    def _merge_rules(original: PageRules, retry: PageRules, invalid_fields: list[str]) -> PageRules:
        """合并规则：用重试结果替换无效字段。"""
        data = original.model_dump()
        retry_data = retry.model_dump()
        for field in invalid_fields:
            if retry_data.get(field):
                data[field] = retry_data[field]
            else:
                data[field] = None
        return PageRules(**{k: v for k, v in data.items() if v is not None})

    @staticmethod
    def _log_rules(rules: PageRules):
        found = []
        if rules.load_more_selector:
            found.append(f"load_more: {rules.load_more_selector}")
        if rules.next_button_selector:
            found.append(f"next_button: {rules.next_button_selector} (max={rules.pagination_max})")
        if rules.sub_page_selector:
            found.append(f"sub_pages: {rules.sub_page_selector} (recursive={rules.sub_page_recursive})")
        if found:
            logger.info("发现 %d 条规则:", len(found))
            for r in found:
                logger.info("  - %s", r)
        else:
            logger.info("未找到匹配的遍历规则")
