"""测试 agent_scraper.rule_discoverer — 页面规则发现"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_scraper.extraction.rule_discoverer import RuleDiscoverer
from agent_scraper.core.models import PageRules

from tests.conftest import SAMPLE_HTML


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def discoverer(mock_client):
    return RuleDiscoverer(client=mock_client)


class TestGetCleanSnippet:
    def test_removes_scripts_and_styles(self):
        html = "<html><body><script>bad()</script><style>.x{}</style><p>good</p></body></html>"
        snippet = RuleDiscoverer._get_clean_snippet(html)
        assert "bad" not in snippet
        assert ".x{}" not in snippet
        assert "good" in snippet

    def test_finds_main_content(self):
        html = '<html><body><nav>nav</nav><main><p>main content</p></main></body></html>'
        snippet = RuleDiscoverer._get_clean_snippet(html)
        assert "main content" in snippet

    def test_falls_back_to_body(self):
        html = "<html><body><div>content</div></body></html>"
        snippet = RuleDiscoverer._get_clean_snippet(html)
        assert "content" in snippet

    def test_truncation(self):
        big_content = "x" * 60000
        html = f"<html><body><main>{big_content}</main></body></html>"
        snippet = RuleDiscoverer._get_clean_snippet(html)
        assert len(snippet) <= 50 * 1024 + 100


class TestDiscover:
    @pytest.mark.asyncio
    async def test_no_hints_returns_empty_rules(self, discoverer):
        """没有 traversal_hints → 不调用 LLM，直接返回空规则"""
        rules = await discoverer.discover(SAMPLE_HTML, "https://example.com", [])
        assert isinstance(rules, PageRules)
        assert rules.load_more_selector is None
        assert rules.sub_page_selector is None
        discoverer.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_hints_returns_empty(self, discoverer):
        rules = await discoverer.discover(SAMPLE_HTML, "", None)
        assert isinstance(rules, PageRules)
        discoverer.client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_more_hint(self, discoverer, mock_client):
        llm_response = json.dumps({
            "load_more_selector": "button.load-more",
            "next_button_selector": None,
            "pagination_url": None,
            "pagination_max": None,
            "sub_page_selector": None,
            "sub_page_url_attr": "href",
            "sub_page_recursive": False,
        })
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = llm_response
        mock_client.chat.completions.create.return_value = mock_resp

        rules = await discoverer.discover(SAMPLE_HTML, "https://example.com", ["load_more"])
        assert rules.load_more_selector == "button.load-more"
        assert rules.sub_page_selector is None  # 用户没要求

    @pytest.mark.asyncio
    async def test_sub_pages_hint(self, discoverer, mock_client):
        llm_response = json.dumps({
            "load_more_selector": None,
            "sub_page_selector": "a.folder",
            "sub_page_url_attr": "href",
            "sub_page_recursive": True,
        })
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = llm_response
        mock_client.chat.completions.create.return_value = mock_resp

        rules = await discoverer.discover(SAMPLE_HTML, "", ["sub_pages"])
        assert rules.sub_page_selector == "a.folder"
        assert rules.sub_page_recursive is True

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self, discoverer, mock_client):
        """LLM 调用失败 → 返回空规则"""
        mock_client.chat.completions.create.side_effect = Exception("LLM error")
        rules = await discoverer.discover(SAMPLE_HTML, "", ["load_more"])
        assert isinstance(rules, PageRules)
        assert rules.load_more_selector is None

    @pytest.mark.asyncio
    async def test_filters_unrequested_modes(self, discoverer, mock_client):
        """LLM 返回了用户没要求的模式 → 应被过滤掉"""
        llm_response = json.dumps({
            "load_more_selector": "button.load",
            "next_button_selector": "a.next",  # 用户没要求
            "pagination_url": None,
            "sub_page_selector": "a.sub",      # 用户没要求
            "sub_page_url_attr": "href",
            "sub_page_recursive": False,
        })
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = llm_response
        mock_client.chat.completions.create.return_value = mock_resp

        rules = await discoverer.discover(SAMPLE_HTML, "", ["load_more"])
        assert rules.load_more_selector == "button.load"
        assert rules.next_button_selector is None
        assert rules.sub_page_selector is None
