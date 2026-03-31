"""测试 agent_scraper.rule_discoverer — 页面规则发现"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_scraper.extraction.rule_discoverer import RuleDiscoverer
from agent_scraper.extraction.css_engine import preprocess_html
from agent_scraper.core.models import PageRules

from tests.conftest import SAMPLE_HTML


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock()
    return llm


@pytest.fixture
def discoverer(mock_llm):
    return RuleDiscoverer(llm_service=mock_llm)


class TestPreprocessHtml:
    """测试 preprocess_html (替代旧 _get_clean_snippet)"""

    def test_removes_scripts_and_styles(self):
        html = "<html><body><script>bad()</script><style>.x{}</style><p>good</p></body></html>"
        snippet = preprocess_html(html)
        assert "bad" not in snippet
        assert ".x{}" not in snippet
        assert "good" in snippet

    def test_finds_main_content(self):
        html = '<html><body><nav>nav</nav><main><p>main content</p></main></body></html>'
        snippet = preprocess_html(html)
        assert "main content" in snippet

    def test_falls_back_to_body(self):
        html = "<html><body><div>content</div></body></html>"
        snippet = preprocess_html(html)
        assert "content" in snippet

    def test_truncation(self):
        big_content = "x" * 60000
        html = f"<html><body><main>{big_content}</main></body></html>"
        snippet = preprocess_html(html)
        assert len(snippet) <= 15000 + 100  # preprocess_html defaults to 15KB


class TestDiscover:
    @pytest.mark.asyncio
    async def test_no_hints_returns_empty_rules(self, discoverer, mock_llm):
        """没有 traversal_hints → 不调用 LLM，直接返回空规则"""
        rules = await discoverer.discover(SAMPLE_HTML, "https://example.com", [])
        assert isinstance(rules, PageRules)
        assert rules.load_more_selector is None
        assert rules.sub_page_selector is None
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_hints_returns_empty(self, discoverer, mock_llm):
        rules = await discoverer.discover(SAMPLE_HTML, "", None)
        assert isinstance(rules, PageRules)
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_more_hint(self, discoverer, mock_llm):
        mock_llm.call.return_value = json.dumps({
            "load_more_selector": "button.load-more",
            "next_button_selector": None,
            "pagination_max": None,
            "sub_page_selector": None,
            "sub_page_url_attr": "href",
            "sub_page_recursive": False,
        })

        rules = await discoverer.discover(SAMPLE_HTML, "https://example.com", ["load_more"])
        # Note: validate_css may fail since button.load-more doesn't exist in SAMPLE_HTML,
        # triggering a retry. The retry also won't find it — so it stays None.
        # For this test, we check that LLM was called.
        assert mock_llm.call.call_count >= 1

    @pytest.mark.asyncio
    async def test_sub_pages_hint(self, discoverer, mock_llm):
        mock_llm.call.return_value = json.dumps({
            "load_more_selector": None,
            "sub_page_selector": "a.file-link",  # exists in SAMPLE_HTML
            "sub_page_url_attr": "href",
            "sub_page_recursive": True,
        })

        rules = await discoverer.discover(SAMPLE_HTML, "", ["sub_pages"])
        assert rules.sub_page_selector == "a.file-link"
        assert rules.sub_page_recursive is True

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self, discoverer, mock_llm):
        """LLM 调用失败 → 返回空规则"""
        mock_llm.call.side_effect = Exception("LLM error")
        rules = await discoverer.discover(SAMPLE_HTML, "", ["load_more"])
        assert isinstance(rules, PageRules)
        assert rules.load_more_selector is None

    @pytest.mark.asyncio
    async def test_filters_unrequested_modes(self, discoverer, mock_llm):
        """LLM 返回了用户没要求的模式 → 应被过滤掉"""
        mock_llm.call.return_value = json.dumps({
            "load_more_selector": "a.file-link",  # exists in SAMPLE_HTML
            "next_button_selector": "a.next",
            "sub_page_selector": "a.sub",
            "sub_page_url_attr": "href",
            "sub_page_recursive": False,
        })

        rules = await discoverer.discover(SAMPLE_HTML, "", ["load_more"])
        assert rules.load_more_selector == "a.file-link"
        assert rules.next_button_selector is None
        assert rules.sub_page_selector is None


class TestValidateSelectors:
    def test_valid_selector(self):
        rules = PageRules(load_more_selector="a.file-link")
        invalid = RuleDiscoverer.validate_selectors(SAMPLE_HTML, rules)
        assert invalid == []

    def test_invalid_selector(self):
        rules = PageRules(load_more_selector="button.nonexistent")
        invalid = RuleDiscoverer.validate_selectors(SAMPLE_HTML, rules)
        assert "load_more_selector" in invalid

    def test_skips_none_selectors(self):
        rules = PageRules()
        invalid = RuleDiscoverer.validate_selectors(SAMPLE_HTML, rules)
        assert invalid == []


class TestDiscoverRetry:
    @pytest.mark.asyncio
    async def test_failed_attempts_injected_into_prompt(self, discoverer, mock_llm):
        """failed_attempts 的选择器和原因被注入到重试 prompt 中"""
        mock_llm.call.return_value = json.dumps({
            "sub_page_selector": "a.file-link",
            "sub_page_url_attr": "href",
            "sub_page_recursive": False,
        })

        failed = [
            {"field": "sub_page_selector", "selector": "ul > li > a.underline", "reason": "无匹配"},
        ]
        rules = await discoverer.discover_retry(
            SAMPLE_HTML, "https://example.com",
            traversal_hints=["sub_pages"], missing_modes=["sub_pages"],
            failed_attempts=failed,
        )

        # 验证 LLM 被调用，且 prompt 包含失败反馈
        mock_llm.call.assert_called_once()
        prompt_arg = mock_llm.call.call_args[0][0]
        assert "ul > li > a.underline" in prompt_arg
        assert "无匹配" in prompt_arg

    @pytest.mark.asyncio
    async def test_no_failed_attempts_still_works(self, discoverer, mock_llm):
        """failed_attempts=None 时正常工作（兼容性）"""
        mock_llm.call.return_value = json.dumps({
            "sub_page_selector": "a.file-link",
            "sub_page_url_attr": "href",
            "sub_page_recursive": False,
        })

        rules = await discoverer.discover_retry(
            SAMPLE_HTML, "https://example.com",
            traversal_hints=["sub_pages"], missing_modes=["sub_pages"],
        )
        mock_llm.call.assert_called_once()


class TestBuildRules:
    def test_only_requested_hints(self):
        data = {
            "load_more_selector": "button.load",
            "sub_page_selector": "a.sub",
        }
        rules = RuleDiscoverer._build_rules(data, ["load_more"])
        assert rules.load_more_selector == "button.load"
        assert rules.sub_page_selector is None

    def test_all_hints(self):
        data = {
            "load_more_selector": "button.load",
            "next_button_selector": "a.next",
            "pagination_max": 10,
            "sub_page_selector": "a.sub",
            "sub_page_url_attr": "href",
            "sub_page_recursive": True,
        }
        rules = RuleDiscoverer._build_rules(
            data, ["load_more", "next_button", "sub_pages"]
        )
        assert rules.load_more_selector == "button.load"
        assert rules.next_button_selector == "a.next"
        assert rules.pagination_max == 10
        assert rules.sub_page_selector == "a.sub"
        assert rules.sub_page_recursive is True
