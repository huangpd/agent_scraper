"""测试 agent_scraper.extractor — CSS 选择器提取 + 缓存 + 验证"""

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from agent_scraper.extraction.extractor import Extractor
from agent_scraper.core.models import ExtractionGoal

from tests.conftest import SAMPLE_HTML


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def extractor(mock_client):
    return Extractor(client=mock_client)


class TestApplyCssSelectors:
    def test_text_extraction(self):
        selectors = {
            "name": {"selector": "a.file-link", "attr": "text"},
        }
        result = Extractor._apply_css_selectors(SAMPLE_HTML, selectors)
        assert "name" in result
        assert len(result["name"]) == 3
        assert "config.json" in result["name"]
        assert "model.bin" in result["name"]

    def test_href_extraction(self):
        selectors = {
            "url": {"selector": "a.file-link", "attr": "href"},
        }
        result = Extractor._apply_css_selectors(SAMPLE_HTML, selectors)
        assert len(result["url"]) == 3
        assert "/repo/blob/main/config.json" in result["url"]

    def test_invalid_selector(self):
        selectors = {
            "name": {"selector": "[[invalid", "attr": "text"},
        }
        result = Extractor._apply_css_selectors(SAMPLE_HTML, selectors)
        assert result["name"] == []

    def test_no_match(self):
        selectors = {
            "name": {"selector": "div.nonexistent", "attr": "text"},
        }
        result = Extractor._apply_css_selectors(SAMPLE_HTML, selectors)
        assert result["name"] == []

    def test_multiple_fields(self):
        selectors = {
            "name": {"selector": "a.file-link", "attr": "text"},
            "url": {"selector": "a.file-link", "attr": "href"},
            "size": {"selector": "span.size", "attr": "text"},
        }
        result = Extractor._apply_css_selectors(SAMPLE_HTML, selectors)
        assert len(result["name"]) == 3
        assert len(result["url"]) == 3
        assert len(result["size"]) == 3


class TestValidateResult:
    def test_valid(self):
        result = {"name": ["a", "b"], "url": ["/a", "/b"]}
        assert Extractor._validate_result(result, {"name", "url"})

    def test_empty_result(self):
        assert not Extractor._validate_result({})

    def test_missing_field(self):
        result = {"name": ["a", "b"]}
        assert not Extractor._validate_result(result, {"name", "url"})

    def test_empty_values(self):
        result = {"name": ["a", "b"], "url": []}
        assert not Extractor._validate_result(result, {"name", "url"})

    def test_length_mismatch_too_large(self):
        result = {"name": ["a"] * 10, "url": ["/a"] * 3}
        assert not Extractor._validate_result(result, {"name", "url"})

    def test_no_expected_fields(self):
        result = {"name": ["a", "b"]}
        assert Extractor._validate_result(result)


class TestNormalizeUrlSamples:
    def test_adds_path(self):
        samples = {"url": ["https://example.com/repo/file.txt"]}
        result = Extractor._normalize_url_samples(samples)
        assert "/repo/file.txt" in result["url"]
        assert "https://example.com/repo/file.txt" in result["url"]

    def test_non_url_unchanged(self):
        samples = {"name": ["config.json"]}
        result = Extractor._normalize_url_samples(samples)
        assert result["name"] == ["config.json"]

    def test_no_duplicate_paths(self):
        samples = {"url": ["https://a.com/path", "https://b.com/path"]}
        result = Extractor._normalize_url_samples(samples)
        path_count = result["url"].count("/path")
        assert path_count == 1


class TestGetMainContentSnippet:
    def test_removes_scripts(self):
        html = "<html><body><main><script>alert(1)</script><p>content</p></main></body></html>"
        ext = Extractor.__new__(Extractor)
        snippet = ext._get_main_content_snippet(html)
        assert "alert" not in snippet
        assert "content" in snippet

    def test_finds_main_tag(self):
        html = "<html><body><nav>nav</nav><main><p>real content</p></main></body></html>"
        ext = Extractor.__new__(Extractor)
        snippet = ext._get_main_content_snippet(html)
        assert "real content" in snippet

    def test_truncates_large_html(self):
        html = "<html><body><main>" + "x" * 60000 + "</main></body></html>"
        ext = Extractor.__new__(Extractor)
        snippet = ext._get_main_content_snippet(html)
        assert len(snippet) <= 50 * 1024 + 100  # some tag overhead


class TestCssCache:
    @pytest.mark.asyncio
    async def test_cache_reused_on_second_call(self, extractor):
        """第二次调用应该使用缓存，不调用 LLM"""
        goal = ExtractionGoal(fields={"name": "文件名", "url": "链接"})
        selectors = {
            "name": {"selector": "a.file-link", "attr": "text"},
            "url": {"selector": "a.file-link", "attr": "href"},
        }

        # 手动设置缓存
        extractor._cached_css_selectors = selectors

        result = await extractor.extract(SAMPLE_HTML, goal)

        # 应该使用缓存，不调用 LLM
        extractor.client.chat.completions.create.assert_not_called()
        assert len(result["name"]) == 3

    @pytest.mark.asyncio
    async def test_llm_called_when_no_cache(self, extractor, mock_client):
        """无缓存时应调用 LLM 生成 CSS 选择器"""
        goal = ExtractionGoal(fields={"name": "文件名"})

        # mock LLM 返回 CSS selectors
        css_response = json.dumps({
            "name": {"selector": "a.file-link", "attr": "text"},
        })
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = css_response
        mock_client.chat.completions.create.return_value = mock_resp

        result = await extractor.extract(SAMPLE_HTML, goal)
        assert mock_client.chat.completions.create.call_count >= 1


class TestExtractHybrid:
    @pytest.mark.asyncio
    async def test_partial_autoscraper_triggers_css_fill(self, extractor, mock_client):
        """AutoScraper 部分成功时，CSS 补齐缺失字段"""
        goal = ExtractionGoal(fields={"name": "文件名", "url": "链接"})

        # 模拟 AutoScraper 只提取到 name
        with patch.object(extractor, '_autoscraper_extract', return_value={"name": ["a", "b", "c"]}):
            # mock LLM 返回同时包含两个字段的 CSS selector（兜底全量提取）
            css_response = json.dumps({
                "name": {"selector": "a.file-link", "attr": "text"},
                "url": {"selector": "a.file-link", "attr": "href"},
            })
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = css_response
            mock_client.chat.completions.create.return_value = mock_resp

            result = await extractor.extract(SAMPLE_HTML, goal)
            # 最终结果应包含两个字段（通过 CSS 补齐或兜底）
            assert "url" in result
            assert len(result["url"]) >= 1
