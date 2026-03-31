"""测试 agent_scraper.extractor — CSS 选择器提取 + 验证"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_scraper.extraction.extractor import Extractor
from agent_scraper.extraction.css_engine import preprocess_html, validate_selectors_batch
from agent_scraper.core.models import ExtractionGoal

from tests.conftest import SAMPLE_HTML


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call = AsyncMock()
    return llm


@pytest.fixture
def extractor(mock_llm):
    return Extractor(llm_service=mock_llm)


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


class TestExpandUrlSamples:
    def test_adds_path(self):
        samples = {"url": ["https://example.com/repo/file.txt"]}
        result = Extractor._expand_url_samples(samples)
        assert "/repo/file.txt" in result["url"]
        assert "https://example.com/repo/file.txt" in result["url"]

    def test_non_url_unchanged(self):
        samples = {"name": ["config.json"]}
        result = Extractor._expand_url_samples(samples)
        assert result["name"] == ["config.json"]

    def test_no_duplicate_paths(self):
        samples = {"url": ["https://a.com/path", "https://b.com/path"]}
        result = Extractor._expand_url_samples(samples)
        path_count = result["url"].count("/path")
        assert path_count == 1


class TestPreprocessHtml:
    """测试 preprocess_html (替代旧 _get_main_content_snippet)"""

    def test_removes_scripts(self):
        html = "<html><body><main><script>alert(1)</script><p>content</p></main></body></html>"
        snippet = preprocess_html(html)
        assert "alert" not in snippet
        assert "content" in snippet

    def test_finds_main_tag(self):
        html = "<html><body><nav>nav</nav><main><p>real content</p></main></body></html>"
        snippet = preprocess_html(html)
        assert "real content" in snippet

    def test_truncates_large_html(self):
        html = "<html><body><main>" + "x" * 60000 + "</main></body></html>"
        snippet = preprocess_html(html)
        assert len(snippet) <= 15000 + 100


class TestCssExtract:
    @pytest.mark.asyncio
    async def test_llm_called_for_css(self, extractor, mock_llm):
        """CSS 兜底应调用 LLMService 生成 CSS 选择器"""
        goal = ExtractionGoal(fields={"name": "文件名"})
        mock_llm.call.return_value = json.dumps({
            "name": {"selector": "a.file-link", "attr": "text"},
        })

        result = await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count >= 1
        assert "name" in result
        assert len(result["name"]) == 3

    @pytest.mark.asyncio
    async def test_css_retry_on_invalid_selector(self, extractor, mock_llm):
        """无效选择器应触发重试"""
        goal = ExtractionGoal(fields={"name": "文件名"})
        # 第一次返回无效选择器，第二次返回有效的
        mock_llm.call.side_effect = [
            json.dumps({"name": {"selector": "div.nonexistent", "attr": "text"}}),
            json.dumps({"name": {"selector": "a.file-link", "attr": "text"}}),
        ]

        result = await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count == 2  # 首次 + 重试
        assert len(result["name"]) == 3


class TestCleanHtml:
    """验证 _clean_html 合并后行为：复用 preprocess_html + normalize"""

    def test_removes_script_and_style(self):
        ext = Extractor(llm_service=MagicMock())
        html = "<html><body><script>evil()</script><style>.x{}</style><p>data</p></body></html>"
        result = ext._clean_html(html)
        assert "evil" not in result
        assert ".x{}" not in result
        assert "data" in result

    def test_empty_html(self):
        ext = Extractor(llm_service=MagicMock())
        assert ext._clean_html("") == ""

    def test_preserves_all_list_items(self):
        """max_chars=0 不截断、不压缩：AutoScraper 需要完整列表"""
        ext = Extractor(llm_service=MagicMock())
        items = "".join(f'<li><a href="/f{i}">file{i}</a></li>' for i in range(20))
        html = f"<html><body><ul>{items}</ul></body></html>"
        result = ext._clean_html(html)
        # 20 个 li 全部保留，不被压缩
        assert "file0" in result
        assert "file19" in result
        assert "省略" not in result

    def test_unicode_normalized(self):
        """输出应经过 NFKC 标准化"""
        ext = Extractor(llm_service=MagicMock())
        # \uff21 = 全角 A，NFKC 标准化为普通 A
        html = "<html><body><p>\uff21\uff22\uff23</p></body></html>"
        result = ext._clean_html(html)
        assert "ABC" in result

    def test_html_entities_handled(self):
        """HTML 实体经过 unescape + BeautifulSoup 后保持可解析"""
        ext = Extractor(llm_service=MagicMock())
        html = "<html><body><p>price &amp; value</p></body></html>"
        result = ext._clean_html(html)
        # BeautifulSoup 输出合法 HTML，& 仍为 &amp;，但文本内容正确
        assert "price" in result
        assert "value" in result


class TestValidateSelectorsInExtractor:
    """验证 extractor 使用 validate_selectors_batch 的集成"""

    def test_valid_selectors_return_empty(self):
        selectors = {
            "name": {"selector": "a.file-link", "attr": "text"},
            "url": {"selector": "a.file-link", "attr": "href"},
        }
        invalid = validate_selectors_batch(SAMPLE_HTML, selectors)
        assert invalid == []

    def test_invalid_selectors_detected(self):
        selectors = {
            "name": {"selector": "div.nonexistent", "attr": "text"},
            "url": {"selector": "a.file-link", "attr": "href"},
        }
        invalid = validate_selectors_batch(SAMPLE_HTML, selectors)
        assert "name" in invalid
        assert "url" not in invalid

    def test_supports_css_key(self):
        """validate_selectors_batch 同时支持 'css' 和 'selector' key"""
        selectors_css_key = {
            "name": {"css": "a.file-link", "attr": "text"},
        }
        selectors_selector_key = {
            "name": {"selector": "a.file-link", "attr": "text"},
        }
        assert validate_selectors_batch(SAMPLE_HTML, selectors_css_key) == []
        assert validate_selectors_batch(SAMPLE_HTML, selectors_selector_key) == []

    def test_all_invalid(self):
        selectors = {
            "a": {"selector": "div.nope", "attr": "text"},
            "b": {"selector": "span.nope", "attr": "text"},
        }
        invalid = validate_selectors_batch(SAMPLE_HTML, selectors)
        assert set(invalid) == {"a", "b"}


class TestPreprocessHtmlMaxChars:
    """验证 preprocess_html 的 max_chars=0 行为"""

    def test_max_chars_zero_no_truncation(self):
        """max_chars=0 应跳过截断"""
        html = "<html><body><main>" + "x" * 60000 + "</main></body></html>"
        result = preprocess_html(html, max_chars=0)
        assert "HTML 已截断" not in result
        assert len(result) > 15000

    def test_max_chars_zero_no_compression(self):
        """max_chars=0 应跳过列表压缩"""
        items = "".join(f'<li>item{i}</li>' for i in range(20))
        html = f"<html><body><main><ul>{items}</ul></main></body></html>"
        result = preprocess_html(html, max_chars=0)
        assert "item0" in result
        assert "item19" in result
        assert "省略" not in result

    def test_default_max_chars_truncates(self):
        """默认 max_chars=15000 应截断"""
        html = "<html><body><main>" + "x" * 60000 + "</main></body></html>"
        result = preprocess_html(html)
        assert len(result) <= 15000 + 100

    def test_default_max_chars_compresses(self):
        """默认 max_chars=15000 应压缩重复列表"""
        items = "".join(f'<li>item{i}</li>' for i in range(20))
        html = f"<html><body><main><ul>{items}</ul></main></body></html>"
        result = preprocess_html(html)
        assert "省略" in result


class TestXpathToHref:
    def test_a_in_path(self):
        assert Extractor._xpath_to_href("//div/a/span") == "//div/a/@href"

    def test_ends_with_a(self):
        assert Extractor._xpath_to_href("//div/a") == "//div/a/@href"

    def test_no_a_in_path(self):
        assert Extractor._xpath_to_href("//div/h3") == "//div/h3/a/@href"


class TestCssRuleCache:
    """CSS selector 缓存：同站点不同页面不应重复调 LLM"""

    @pytest.mark.asyncio
    async def test_second_page_uses_cache(self, extractor, mock_llm):
        """第二次调用 _css_selector_extract 应使用缓存，不再调 LLM"""
        goal = ExtractionGoal(fields={"name": "文件名"})
        mock_llm.call.return_value = json.dumps({
            "name": {"selector": "a.file-link", "attr": "text"},
        })

        # 第一次：调 LLM
        result1 = await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count == 1
        assert len(result1["name"]) == 3

        # 第二次：用缓存，LLM 不被调用
        result2 = await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count == 1  # 仍然是 1，没新增
        assert len(result2["name"]) == 3

    @pytest.mark.asyncio
    async def test_cache_invalidated_when_empty(self, extractor, mock_llm):
        """缓存的 selector 在新页面提取为空时，应重新调 LLM"""
        goal = ExtractionGoal(fields={"name": "文件名"})
        mock_llm.call.return_value = json.dumps({
            "name": {"selector": "a.file-link", "attr": "text"},
        })

        # 第一次：成功，缓存建立
        await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count == 1

        # 第二次：用一个没有 a.file-link 的页面，缓存失效，应重新调 LLM
        new_html = "<html><body><div>empty page</div></body></html>"
        mock_llm.call.return_value = json.dumps({
            "name": {"selector": "div", "attr": "text"},
        })
        result = await extractor._css_selector_extract(new_html, goal, {"name"})
        assert mock_llm.call.call_count == 2  # 缓存失效后重新调了 LLM

    @pytest.mark.asyncio
    async def test_clear_cache(self, extractor, mock_llm):
        """clear_cache 后应重新调 LLM"""
        goal = ExtractionGoal(fields={"name": "文件名"})
        mock_llm.call.return_value = json.dumps({
            "name": {"selector": "a.file-link", "attr": "text"},
        })

        await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count == 1

        # 清缓存
        extractor._css_rule_cache.clear()

        await extractor._css_selector_extract(SAMPLE_HTML, goal, {"name"})
        assert mock_llm.call.call_count == 2
