"""测试 agent_scraper.page_iterator — 页面遍历（JS evaluate via CDP）"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_scraper.browser.page_iterator import PageIterator
from agent_scraper.core.models import PageRules


# ── helpers ──────────────────────────────────────────


def _make_page(*, html="<html><body>page</body></html>"):
    """构造 mock Page（browser_use actor Page）"""
    page = MagicMock()
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value=html)
    return page


def _make_browser(page=None):
    """构造 mock browser（BrowserSession）"""
    if page is None:
        page = _make_page()
    browser = MagicMock()
    browser.get_current_page = AsyncMock(return_value=page)
    browser.get_pages = AsyncMock(return_value=[page])
    browser.new_page = AsyncMock(return_value=page)
    return browser, page


# ── _get_page recovery ──────────────────────────────


class TestGetPage:
    @pytest.mark.asyncio
    async def test_returns_page_on_first_try(self):
        browser, page = _make_browser()
        it = PageIterator(browser)
        result = await it._get_page()
        assert result is page

    @pytest.mark.asyncio
    async def test_retries_on_none(self):
        page = _make_page()
        browser = MagicMock()
        browser.get_current_page = AsyncMock(side_effect=[None, page])
        it = PageIterator(browser)
        result = await it._get_page(retries=2)
        assert result is page

    @pytest.mark.asyncio
    async def test_falls_back_to_get_pages(self):
        page = _make_page()
        browser = MagicMock()
        browser.get_current_page = AsyncMock(return_value=None)
        browser.get_pages = AsyncMock(return_value=[page])
        it = PageIterator(browser)
        result = await it._get_page(retries=1)
        assert result is page

    @pytest.mark.asyncio
    async def test_falls_back_to_new_page(self):
        page = _make_page()
        browser = MagicMock()
        browser.get_current_page = AsyncMock(return_value=None)
        browser.get_pages = AsyncMock(return_value=[])
        browser.new_page = AsyncMock(return_value=page)
        it = PageIterator(browser)
        result = await it._get_page(retries=1)
        assert result is page

    @pytest.mark.asyncio
    async def test_raises_when_all_fail(self):
        browser = MagicMock()
        browser.get_current_page = AsyncMock(return_value=None)
        browser.get_pages = AsyncMock(return_value=[])
        browser.new_page = AsyncMock(side_effect=Exception("CDP error"))
        it = PageIterator(browser)
        with pytest.raises(RuntimeError, match="无法恢复"):
            await it._get_page(retries=1)


# ── _get_html ────────────────────────────────────────


class TestGetHtml:
    @pytest.mark.asyncio
    async def test_uses_page_evaluate(self):
        browser, page = _make_browser()
        page.evaluate = AsyncMock(return_value="<html>test</html>")
        it = PageIterator(browser)
        html = await it._get_html()
        assert html == "<html>test</html>"
        page.evaluate.assert_awaited_once()


# ── JS click helpers ────────────────────────────────


class TestClickByXpath:
    @pytest.mark.asyncio
    async def test_returns_true_on_click(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="clicked")
        assert await PageIterator._click_by_xpath(page, "//button", "Load more") is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="")
        assert await PageIterator._click_by_xpath(page, "//button", "") is False


class TestClickBySelector:
    @pytest.mark.asyncio
    async def test_returns_true_on_click(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="clicked")
        assert await PageIterator._click_by_selector(page, "button.load-more") is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="")
        assert await PageIterator._click_by_selector(page, "button.load-more") is False


class TestClickByText:
    @pytest.mark.asyncio
    async def test_returns_true_on_click(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="clicked")
        assert await PageIterator._click_by_text(page, "Load more") is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="")
        assert await PageIterator._click_by_text(page, "nonexistent") is False


# ── _try_load_more ───────────────────────────────────


class TestTryLoadMore:
    @pytest.mark.asyncio
    async def test_xpath_click(self):
        """XPath + button_text -> _click_by_xpath"""
        page = _make_page()
        call_count = [0]

        async def eval_dispatch(js, *args):
            call_count[0] += 1
            if "document.evaluate" in js:
                return "clicked" if call_count[0] <= 2 else ""
            if "scrollHeight" in js:
                return str(1000 + call_count[0] * 500)
            return ""

        page.evaluate = AsyncMock(side_effect=eval_dispatch)
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        await it._try_load_more(xpath="//li[@class='item']", button_text="Load more files")

    @pytest.mark.asyncio
    async def test_no_button_found(self):
        """所有方式都找不到 -> 不点击"""
        page = _make_page()
        page.evaluate = AsyncMock(return_value="")
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        await it._try_load_more(button_text="nonexistent")

    @pytest.mark.asyncio
    async def test_stops_on_no_height_change(self):
        """页面高度不变 -> 停止"""
        page = _make_page()
        heights = iter(["500", "500"])

        async def eval_dispatch(js, *args):
            if "querySelectorAll" in js and 'role=' not in js:
                return "clicked"
            if "scrollHeight" in js:
                return next(heights)
            return ""

        page.evaluate = AsyncMock(side_effect=eval_dispatch)
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        await it._try_load_more(selector="button.load")


# ── _extract_links ───────────────────────────────────


class TestExtractLinks:
    @pytest.mark.asyncio
    async def test_extracts_and_resolves(self):
        page = _make_page()
        page.evaluate = AsyncMock(
            return_value=json.dumps(["/repo/tree/main/src", "/repo/tree/main/tests"]))
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        urls = await it._extract_links("a.folder", "href", "https://example.com")
        assert len(urls) == 2
        assert all(u.startswith("https://") for u in urls)

    @pytest.mark.asyncio
    async def test_dedup(self):
        page = _make_page()
        page.evaluate = AsyncMock(
            return_value=json.dumps(["/path", "/path", "/other"]))
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        urls = await it._extract_links("a", "href", "https://example.com")
        assert len(urls) == 2

    @pytest.mark.asyncio
    async def test_absolute_urls(self):
        page = _make_page()
        page.evaluate = AsyncMock(
            return_value=json.dumps(["https://cdn.example.com/file"]))
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        urls = await it._extract_links("a", "href", "https://example.com")
        assert urls[0] == "https://cdn.example.com/file"

    @pytest.mark.asyncio
    async def test_empty_result(self):
        page = _make_page()
        page.evaluate = AsyncMock(return_value="[]")
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        urls = await it._extract_links("a", "href", "https://example.com")
        assert len(urls) == 0


# ── _do_next_button ──────────────────────────────────


class TestDoNextButton:
    @pytest.mark.asyncio
    async def test_clicks_until_not_found(self):
        page = _make_page()
        click_count = [0]

        async def eval_dispatch(js, *args):
            if "querySelectorAll" in js and 'role=' not in js:
                click_count[0] += 1
                return "clicked" if click_count[0] <= 2 else ""
            return "<html>page</html>"

        page.evaluate = AsyncMock(side_effect=eval_dispatch)
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        htmls = [h async for h in it._do_next_button("a.next-page", max_pages=5)]
        assert len(htmls) == 2

    @pytest.mark.asyncio
    async def test_respects_max_pages(self):
        page = _make_page()

        async def eval_dispatch(js, *args):
            if "querySelectorAll" in js and 'role=' not in js:
                return "clicked"
            return "<html>page</html>"

        page.evaluate = AsyncMock(side_effect=eval_dispatch)
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        htmls = [h async for h in it._do_next_button("a.next", max_pages=3)]
        assert len(htmls) == 2


# ── iterate (集成) ───────────────────────────────────


class TestIterate:
    @pytest.mark.asyncio
    async def test_single_page_no_rules(self):
        browser, page = _make_browser()
        it = PageIterator(browser)
        rules = PageRules()
        htmls = [h async for h in it.iterate("<html>first</html>", rules)]
        assert len(htmls) == 1
        assert htmls[0] == "<html>first</html>"

    @pytest.mark.asyncio
    async def test_pagination_url(self):
        page = _make_page(html="<html>" + "x" * 2000 + "</html>")
        browser, _ = _make_browser(page)
        it = PageIterator(browser)
        rules = PageRules(pagination_url="https://example.com/page/{n}", pagination_max=3)
        htmls = [h async for h in it.iterate("<html>page1</html>", rules)]
        assert len(htmls) == 3
