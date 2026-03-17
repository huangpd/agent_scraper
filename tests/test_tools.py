"""测试 IteratePagesTool 流式提取"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_scraper.core.models import ExtractionGoal, NavigationStep, PageRules, ParsedTask
from agent_scraper.pipeline.context import AgentContext
from agent_scraper.pipeline.tools import IteratePagesTool


async def _async_gen(items):
    for item in items:
        yield item


def _make_ctx(*, samples=None, browser=None, html="<html/>", page_rules=None):
    """构造测试用 AgentContext"""
    goal = ExtractionGoal(
        fields={"name": "名称", "url": "链接"},
        samples=samples or {"name": ["a.txt"], "url": ["/a"]},
    )
    task = ParsedTask(
        navigation_steps=[
            NavigationStep(action="goto", target="https://example.com", description="open")
        ],
        extraction_goal=goal,
        raw_instruction="test",
    )
    ctx = AgentContext(task=task)
    ctx.browser = browser
    ctx.html = html
    ctx.page_rules = page_rules
    return ctx


class TestIteratePagesTool:
    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_stream_extract_no_html_accumulation(self, MockPageIter):
        """验证页面 HTML 不会累积，extracted_data 有数据"""
        pages = [(f"https://example.com/page{i}", f"<html>page{i}</html>") for i in range(5)]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(
            side_effect=[
                [{"name": f"file{i}.txt", "url": f"/file{i}"}]
                for i in range(5)
            ]
        )

        browser = MagicMock()
        ctx = _make_ctx(browser=browser)

        tool = IteratePagesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        assert result.data["page_count"] == 5
        assert not hasattr(ctx, "html_pages")
        assert len(ctx.extracted_data) == 5
        assert all("_source_url" in r for r in ctx.extracted_data)
        # url 字段已有值时不被覆盖
        assert ctx.extracted_data[0]["url"] == "/file0"

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_empty_url_filled_from_source(self, MockPageIter):
        """url 字段为空时，用详情页地址自动填充"""
        pages = [("https://example.com/detail/1", "<html>d1</html>")]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        # AutoScraper 提取到 name 但 url 为空
        extractor.extract = AsyncMock(return_value=[{"name": "title1", "url": ""}])

        browser = MagicMock()
        ctx = _make_ctx(browser=browser)

        tool = IteratePagesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        assert len(ctx.extracted_data) == 1
        # 空 url 被 _source_url 填充
        assert ctx.extracted_data[0]["url"] == "https://example.com/detail/1"
        assert ctx.extracted_data[0]["_source_url"] == "https://example.com/detail/1"

    @pytest.mark.asyncio
    async def test_single_page_no_browser(self):
        """无浏览器时降级为单页提取"""
        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=[{"name": "x", "url": "/x"}])

        ctx = _make_ctx(html="<html>single</html>")
        tool = IteratePagesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        assert result.data["page_count"] == 1
        extractor.extract.assert_called_once()
        assert ctx.extracted_data == [{"name": "x", "url": "/x"}]

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_extractor_called_per_page(self, MockPageIter):
        """Extractor 对每页调用一次"""
        pages = [("https://example.com/p1", "<html>p1</html>"), ("https://example.com/p2", "<html>p2</html>"), ("https://example.com/p3", "<html>p3</html>")]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=[{"name": "x", "url": "/x"}])

        browser = MagicMock()
        ctx = _make_ctx(browser=browser)

        tool = IteratePagesTool(extractor=extractor)
        await tool.execute(ctx)

        assert extractor.extract.call_count == 3

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_progress_events_emitted(self, MockPageIter):
        """每页发出 progress 事件"""
        pages = [("https://example.com/p1", "<html>p1</html>"), ("https://example.com/p2", "<html>p2</html>")]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=[{"name": "x", "url": "/x"}])

        events = []
        browser = MagicMock()
        ctx = _make_ctx(browser=browser)
        ctx.on_event = lambda t, d: events.append((t, d))

        tool = IteratePagesTool(extractor=extractor)
        await tool.execute(ctx)

        progress_events = [e for e in events if e[0] == "progress"]
        assert len(progress_events) == 2
        assert progress_events[0][1] == {"current": 1, "total": 0}
        assert progress_events[1][1] == {"current": 2, "total": 0}

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_no_extractor_skips_extraction(self, MockPageIter):
        """没传 Extractor 时只遍历不提取"""
        pages = [("https://example.com/p1", "<html>p1</html>"), ("https://example.com/p2", "<html>p2</html>")]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        browser = MagicMock()
        ctx = _make_ctx(browser=browser)

        tool = IteratePagesTool(extractor=None)
        result = await tool.execute(ctx)

        assert result.success
        assert result.data["page_count"] == 2
        assert ctx.extracted_data == []  # 无提取

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_free_mode_skips_inline_extract(self, MockPageIter):
        """无样本（自由模式）时 IteratePagesTool 不做内联提取"""
        pages = [("https://example.com/p1", "<html>p1</html>")]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value=[{"name": "x"}])

        browser = MagicMock()
        # 构造无样本的 ctx
        goal = ExtractionGoal(fields={"name": "名称"})
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=goal,
            raw_instruction="test",
        )
        ctx = AgentContext(task=task)
        ctx.browser = browser
        ctx.html = "<html/>"

        tool = IteratePagesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        # 无样本时不调用 extractor
        extractor.extract.assert_not_called()
        assert ctx.extracted_data == []
