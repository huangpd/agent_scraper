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

    def test_contains_pseudo_selector(self):
        """LLM 返回 :contains() jQuery 语法 → 自动转为文本匹配"""
        js = PageIterator._build_load_more_js("button:contains('Load more files')")
        # 不应包含 querySelector(':contains(')
        assert ":contains(" not in js
        # 应转为正则文本匹配（re.escape 会转义空格）
        assert "Load" in js and "more" in js and "files" in js
        assert "querySelectorAll" in js
        assert "clicked" in js

    def test_contains_double_quotes(self):
        """双引号版本的 :contains()"""
        js = PageIterator._build_load_more_js('button:contains("更多内容")')
        assert ":contains(" not in js
        assert "更多内容" in js


class TestTryLoadMore:
    @pytest.mark.asyncio
    async def test_clicks_until_not_found(self, iterator, mock_browser):
        """应循环点击直到返回 not_found"""
        page = await mock_browser.get_current_page()
        # 每次循环: setAttribute → build_js → [clicked后:] scrollTo, getAttribute, scrollHeight
        # 循环1: setAttribute, "clicked", scrollTo, getAttribute("1"), scrollHeight(1000)
        # 循环2: setAttribute, "clicked", scrollTo, getAttribute("1"), scrollHeight(2000)
        # 循环3: setAttribute, "not_found" → 退出
        page.evaluate = AsyncMock(side_effect=[
            None, "clicked", None, "1", 1000,   # 循环1
            None, "clicked", None, "1", 2000,   # 循环2
            None, "not_found",                   # 循环3: 按钮消失
        ])

        await iterator._try_load_more("button.load")
        assert page.evaluate.call_count == 12

    @pytest.mark.asyncio
    async def test_no_button(self, iterator, mock_browser):
        page = await mock_browser.get_current_page()
        # setAttribute → build_load_more_js → "not_found"
        page.evaluate = AsyncMock(side_effect=[None, "not_found"])

        await iterator._try_load_more(None)
        assert page.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_stops_on_page_refresh(self, iterator, mock_browser):
        """点击后 DOM 标记消失 → 整页刷新 → 停止"""
        page = await mock_browser.get_current_page()
        # setAttribute, "clicked", scrollTo, getAttribute → None (标记消失=整页刷新)
        page.evaluate = AsyncMock(side_effect=[None, "clicked", None, None])

        await iterator._try_load_more("button.load")
        assert page.evaluate.call_count == 4


class TestIterate:
    @pytest.mark.asyncio
    async def test_single_page_no_rules(self, iterator, mock_browser):
        """无规则 → 返回单页"""
        page = await mock_browser.get_current_page()
        # _try_load_more 返回 not_found，_get_html 返回 HTML
        page.evaluate = AsyncMock(side_effect=["not_found", "<html>updated</html>"])

        rules = PageRules()
        htmls = [h async for h in iterator.iterate("<html>first</html>", rules, "https://example.com")]
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
        htmls = [h async for h in iterator.iterate("<html>page1</html>", rules, "")]
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
        htmls = [h async for h in iterator.iterate("<html>page1</html>", rules, "")]
        assert len(htmls) >= 1


class TestComboMode:
    """组合模式: 列表翻页 + 进详情页"""

    @pytest.mark.asyncio
    async def test_pagination_with_sub_pages(self, iterator, mock_browser):
        """pagination_url + sub_page_selector → 翻列表页 + 进详情页"""
        page = await mock_browser.get_current_page()
        goto_urls = []

        async def mock_goto(url):
            goto_urls.append(url)

        page.goto = mock_goto

        # 模拟: 列表页返回详情链接, 详情页返回 HTML
        eval_calls = []

        async def mock_eval(js):
            eval_calls.append(js)
            # _current_url
            if "window.location.href" in js:
                return "https://example.com/list"
            # _extract_links: 列表页上提取详情链接
            if "querySelectorAll" in js and "getAttribute" in js:
                # 根据 goto 历史判断在哪个列表页
                # 列表页 1 返回 2 个链接, 列表页 2 返回 1 个, 列表页 3 返回 0 个(停止)
                list_page_gotos = [u for u in goto_urls if "/list" in u]
                if len(list_page_gotos) <= 1:
                    return json.dumps(["/detail/1", "/detail/2"])
                elif len(list_page_gotos) == 2:
                    return json.dumps(["/detail/3"])
                else:
                    return json.dumps([])
            # _get_html: 返回足够大的 HTML
            if "outerHTML" in js:
                return "<html><body>" + "x" * 2000 + "</body></html>"
            return ""

        page.evaluate = mock_eval

        rules = PageRules(
            pagination_url="https://example.com/list?page={n}",
            pagination_max=5,
            sub_page_selector="a.title-link",
            sub_page_url_attr="href",
        )
        htmls = [h async for h in iterator.iterate("<html>list1</html>", rules, "https://example.com/list")]

        # 应该 yield 了 3 个详情页的 HTML (不含列表页)
        assert len(htmls) == 3
        # 应该访问过详情页 URL
        detail_gotos = [u for u in goto_urls if "/detail/" in u]
        assert len(detail_gotos) == 3

    @pytest.mark.asyncio
    async def test_next_button_with_sub_pages(self, iterator, mock_browser):
        """next_button + sub_page_selector → 翻页收集链接 + 逐一访问详情页"""
        page = await mock_browser.get_current_page()
        goto_urls = []
        click_count = [0]

        async def mock_goto(url):
            goto_urls.append(url)

        page.goto = mock_goto

        async def mock_eval(js):
            if "window.location.href" in js:
                return "https://example.com/list"
            # 翻页按钮: 点 2 次后 not_found
            if "querySelector" in js and "click" in js:
                click_count[0] += 1
                if click_count[0] <= 2:
                    return "clicked"
                return "not_found"
            # 提取链接: 每个列表页返回 2 个详情链接 (不同 URL)
            if "querySelectorAll" in js and "getAttribute" in js:
                page_idx = click_count[0]
                return json.dumps([
                    f"/detail/{page_idx * 2 + 1}",
                    f"/detail/{page_idx * 2 + 2}",
                ])
            if "outerHTML" in js:
                return "<html><body>detail</body></html>"
            return ""

        page.evaluate = mock_eval

        rules = PageRules(
            next_button_selector="a.next-page",
            sub_page_selector="a.entry",
            sub_page_url_attr="href",
        )
        htmls = [h async for h in iterator.iterate("<html>list</html>", rules, "https://example.com/list")]

        # 3 个列表页 × 2 个详情页 = 6 个
        assert len(htmls) == 6
        detail_gotos = [u for u in goto_urls if "/detail/" in u]
        assert len(detail_gotos) == 6

    @pytest.mark.asyncio
    async def test_sub_pages_only_no_pagination(self, iterator, mock_browser):
        """sub_page_selector 无翻页 → 走递归子页面模式 (yield 列表页 + 详情页)"""
        page = await mock_browser.get_current_page()
        depth = [0]

        async def mock_eval(js):
            if "querySelectorAll" in js and "getAttribute" in js:
                if depth[0] == 0:
                    depth[0] += 1
                    return json.dumps(["/sub/1", "/sub/2"])
                return json.dumps([])
            if "outerHTML" in js:
                return "<html><body>sub page</body></html>"
            return ""

        page.evaluate = mock_eval
        page.goto = AsyncMock()

        rules = PageRules(
            sub_page_selector="a.sub-link",
            sub_page_url_attr="href",
        )
        htmls = [h async for h in iterator.iterate("<html>root</html>", rules, "https://example.com")]

        # 1 (root/列表页) + 2 (子页面)
        assert len(htmls) == 3

    @pytest.mark.asyncio
    async def test_combo_dedup_across_pages(self, iterator, mock_browser):
        """组合模式: 不同列表页返回重复详情链接，应去重"""
        page = await mock_browser.get_current_page()

        async def mock_eval(js):
            if "window.location.href" in js:
                return "https://example.com/list"
            if "querySelectorAll" in js and "getAttribute" in js:
                # 每页都返回同一组链接
                return json.dumps(["/detail/1", "/detail/2"])
            if "outerHTML" in js:
                return "<html><body>" + "x" * 2000 + "</body></html>"
            return ""

        page.evaluate = mock_eval
        page.goto = AsyncMock()

        rules = PageRules(
            pagination_url="https://example.com/list?page={n}",
            pagination_max=3,
            sub_page_selector="a.link",
            sub_page_url_attr="href",
        )
        htmls = [h async for h in iterator.iterate("<html>list</html>", rules, "https://example.com/list")]

        # 虽然 3 个列表页都返回同样 2 个链接，去重后只访问 2 个
        assert len(htmls) == 2


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
