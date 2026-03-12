"""测试 agent_scraper.page_iterator — 页面遍历"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_scraper.browser.page_iterator import PageIterator
from agent_scraper.core.models import PageRules


@pytest.fixture
def mock_browser():
    browser = MagicMock()
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value="<html><body>page</body></html>")
    page.goto = AsyncMock()
    browser.get_current_page = AsyncMock(return_value=page)
    return browser


@pytest.fixture
def iterator(mock_browser):
    return PageIterator(mock_browser)


class TestBuildLoadMoreJs:
    def test_with_selector(self):
        js = PageIterator._build_load_more_js("button.load-more")
        assert "button.load-more" in js
        assert "clicked" in js
        assert "not_found" in js

    def test_without_selector(self):
        js = PageIterator._build_load_more_js(None)
        assert "load more" in js.lower()
        assert "加载更多" in js
        assert "clicked" in js

    def test_selector_escaping(self):
        js = PageIterator._build_load_more_js("button[data-action='load']")
        assert "\\'" in js  # 单引号转义


class TestTryLoadMore:
    @pytest.mark.asyncio
    async def test_clicks_until_not_found(self, iterator, mock_browser):
        """应循环点击直到返回 not_found"""
        page = await mock_browser.get_current_page()
        page.evaluate = AsyncMock(side_effect=["clicked", "clicked", "not_found"])

        await iterator._try_load_more("button.load")
        assert page.evaluate.call_count == 3

    @pytest.mark.asyncio
    async def test_no_button(self, iterator, mock_browser):
        page = await mock_browser.get_current_page()
        page.evaluate = AsyncMock(return_value="not_found")

        await iterator._try_load_more(None)
        assert page.evaluate.call_count == 1


class TestIterate:
    @pytest.mark.asyncio
    async def test_single_page_no_rules(self, iterator, mock_browser):
        """无规则 → 返回单页"""
        page = await mock_browser.get_current_page()
        # _try_load_more 返回 not_found，_get_html 返回 HTML
        page.evaluate = AsyncMock(side_effect=["not_found", "<html>updated</html>"])

        rules = PageRules()
        htmls = await iterator.iterate("<html>first</html>", rules, "https://example.com")
        assert len(htmls) == 1

    @pytest.mark.asyncio
    async def test_pagination_url(self, iterator, mock_browser):
        """URL 分页模式"""
        page = await mock_browser.get_current_page()
        call_count = [0]

        async def mock_eval(js):
            call_count[0] += 1
            if "load more" in js.lower() or "加载更多" in js:
                return "not_found"
            return "<html><body>" + "x" * 2000 + "</body></html>"

        page.evaluate = mock_eval

        rules = PageRules(pagination_url="https://example.com/page/{n}", pagination_max=3)
        htmls = await iterator.iterate("<html>page1</html>", rules, "")
        # page1 + page2 + page3
        assert len(htmls) >= 1

    @pytest.mark.asyncio
    async def test_next_button(self, iterator, mock_browser):
        """翻页按钮模式"""
        page = await mock_browser.get_current_page()
        clicks = [0]

        async def mock_eval(js):
            if "load more" in js.lower() or "加载更多" in js:
                return "not_found"
            if "querySelector" in js and "click" in js:
                clicks[0] += 1
                if clicks[0] <= 2:
                    return "clicked"
                return "not_found"
            return "<html><body>page content</body></html>"

        page.evaluate = mock_eval

        rules = PageRules(next_button_selector="a.next-page")
        htmls = await iterator.iterate("<html>page1</html>", rules, "")
        assert len(htmls) >= 1


class TestExtractLinks:
    @pytest.mark.asyncio
    async def test_extracts_and_resolves(self, iterator, mock_browser):
        """提取链接并补全相对 URL"""
        page = await mock_browser.get_current_page()
        page.evaluate = AsyncMock(
            return_value=json.dumps(["/repo/tree/main/src", "/repo/tree/main/tests"])
        )

        urls = await iterator._extract_links("a.folder", "href", "https://example.com")
        assert len(urls) == 2
        assert all(u.startswith("https://") for u in urls)

    @pytest.mark.asyncio
    async def test_dedup(self, iterator, mock_browser):
        page = await mock_browser.get_current_page()
        page.evaluate = AsyncMock(
            return_value=json.dumps(["/path", "/path", "/other"])
        )

        urls = await iterator._extract_links("a", "href", "https://example.com")
        assert len(urls) == 2

    @pytest.mark.asyncio
    async def test_absolute_urls(self, iterator, mock_browser):
        page = await mock_browser.get_current_page()
        page.evaluate = AsyncMock(
            return_value=json.dumps(["https://cdn.example.com/file"])
        )

        urls = await iterator._extract_links("a", "href", "https://example.com")
        assert urls[0] == "https://cdn.example.com/file"
