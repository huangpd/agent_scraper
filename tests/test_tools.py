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
        pages = [f"<html>page{i}</html>" for i in range(5)]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(
            side_effect=[
                {"name": [f"file{i}.txt"], "url": [f"/file{i}"]}
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
        assert len(ctx.extracted_data["name"]) == 5
        assert len(ctx.extracted_data["url"]) == 5

    @pytest.mark.asyncio
    async def test_single_page_no_browser(self):
        """无浏览器时降级为单页提取"""
        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value={"name": ["x"], "url": ["/x"]})

        ctx = _make_ctx(html="<html>single</html>")
        tool = IteratePagesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        assert result.data["page_count"] == 1
        extractor.extract.assert_called_once()
        assert ctx.extracted_data == {"name": ["x"], "url": ["/x"]}

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_extractor_called_per_page(self, MockPageIter):
        """Extractor 对每页调用一次"""
        pages = ["<html>p1</html>", "<html>p2</html>", "<html>p3</html>"]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value={"name": ["x"], "url": ["/x"]})

        browser = MagicMock()
        ctx = _make_ctx(browser=browser)

        tool = IteratePagesTool(extractor=extractor)
        await tool.execute(ctx)

        assert extractor.extract.call_count == 3

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_progress_events_emitted(self, MockPageIter):
        """每页发出 progress 事件"""
        pages = ["<html>p1</html>", "<html>p2</html>"]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock(return_value={"name": ["x"], "url": ["/x"]})

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
        pages = ["<html>p1</html>", "<html>p2</html>"]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        browser = MagicMock()
        ctx = _make_ctx(browser=browser)

        tool = IteratePagesTool(extractor=None)
        result = await tool.execute(ctx)

        assert result.success
        assert result.data["page_count"] == 2
        assert ctx.extracted_data == {}  # 无提取

    @pytest.mark.asyncio
    @patch("agent_scraper.browser.page_iterator.PageIterator")
    async def test_no_samples_skips_extraction(self, MockPageIter):
        """无样本时 IteratePagesTool 不调用 extractor（bootstrap 已分离到 BootstrapSamplesTool）"""
        pages = ["<html>p1</html>", "<html>p2</html>"]
        MockPageIter.return_value.iterate = MagicMock(return_value=_async_gen(pages))

        extractor = MagicMock()
        extractor.extract = AsyncMock()

        browser = MagicMock()
        goal = ExtractionGoal(fields={"name": "名称", "url": "链接"})
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
        assert result.data["page_count"] == 2
        # 无样本 → extractor 不被调用
        extractor.extract.assert_not_called()
        assert ctx.extracted_data == {}


class TestBootstrapSamplesTool:
    """测试 BootstrapSamplesTool：从首页 LLM 提取 → 校验 → 生成样本"""

    @pytest.mark.asyncio
    async def test_generates_samples_from_html(self):
        """首页 LLM 提取成功 → 校验通过 → 设置样本"""
        from agent_scraper.pipeline.tools import BootstrapSamplesTool

        extractor = MagicMock()
        extractor._css_rule_cache = {}
        extractor._trained_scraper = None
        extractor.extract = AsyncMock(return_value={
            "name": ["article1", "article2", "article3"],
            "url": ["/a1", "/a2", "/a3"],
        })

        goal = ExtractionGoal(fields={"name": "名称", "url": "链接"})
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=goal,
            raw_instruction="test",
        )
        ctx = AgentContext(task=task)
        ctx.html = '<html><body><a href="/a1">article1</a><a href="/a2">article2</a><a href="/a3">article3</a></body></html>'

        tool = BootstrapSamplesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        extractor.extract.assert_called_once()
        # 样本取前 2 条
        assert goal.samples == {"name": ["article1", "article2"], "url": ["/a1", "/a2"]}
        # 缓存已清除
        assert extractor._trained_scraper is None

    @pytest.mark.asyncio
    async def test_skips_when_samples_exist(self):
        """已有样本时跳过 bootstrap"""
        from agent_scraper.pipeline.tools import BootstrapSamplesTool

        extractor = MagicMock()
        extractor.extract = AsyncMock()

        ctx = _make_ctx(samples={"name": ["existing"]})
        tool = BootstrapSamplesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        assert "跳过" in result.summary
        extractor.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_extraction_no_samples(self):
        """首页 LLM 提取为空 → 不生成样本"""
        from agent_scraper.pipeline.tools import BootstrapSamplesTool

        extractor = MagicMock()
        extractor._css_rule_cache = {}
        extractor._trained_scraper = None
        extractor.extract = AsyncMock(return_value={"name": [], "url": []})

        goal = ExtractionGoal(fields={"name": "名称"})
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=goal,
            raw_instruction="test",
        )
        ctx = AgentContext(task=task)
        ctx.html = "<html><body>empty</body></html>"

        tool = BootstrapSamplesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success  # 不致命
        assert goal.samples is None
        assert "为空" in result.summary

    @pytest.mark.asyncio
    async def test_validation_filters_hallucinated_samples(self):
        """样本校验过滤不存在于 HTML 中的值"""
        from agent_scraper.pipeline.tools import BootstrapSamplesTool

        extractor = MagicMock()
        extractor._css_rule_cache = {}
        extractor._trained_scraper = None
        extractor.extract = AsyncMock(return_value={
            "name": ["real_text", "hallucinated_text"],
            "url": ["/real", "/fake"],
        })

        goal = ExtractionGoal(fields={"name": "名称", "url": "链接"})
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=goal,
            raw_instruction="test",
        )
        ctx = AgentContext(task=task)
        ctx.html = '<html><body><a href="/real">real_text</a></body></html>'

        tool = BootstrapSamplesTool(extractor=extractor)
        result = await tool.execute(ctx)

        assert result.success
        # 只保留存在于 HTML 中的值
        assert goal.samples["name"] == ["real_text"]
        assert goal.samples["url"] == ["/real"]


class TestNavigateToolMultiEntity:
    """测试 NavigateTool 多实体模式：extract_point 分组 + HTML 缓存"""

    def test_split_at_extract_points_no_points(self):
        """无 extract_point 时返回单组"""
        from agent_scraper.pipeline.tools import NavigateTool
        steps = [
            NavigationStep(action="goto", target="https://example.com", description="go"),
            NavigationStep(action="click", target="btn", description="click"),
        ]
        groups = NavigateTool._split_at_extract_points(steps)
        assert len(groups) == 1
        assert groups[0] == steps

    def test_split_at_extract_points_multi(self):
        """有 extract_point 时按提取点分组"""
        from agent_scraper.pipeline.tools import NavigateTool
        steps = [
            NavigationStep(action="goto", target="https://wiki.com", description="go"),
            NavigationStep(action="input", target="search", value="A", description="search A"),
            NavigationStep(action="click", target="article A", description="click A", extract_point=True),
            NavigationStep(action="input", target="search", value="B", description="search B"),
            NavigationStep(action="click", target="article B", description="click B", extract_point=True),
        ]
        groups = NavigateTool._split_at_extract_points(steps)
        assert len(groups) == 2
        assert len(groups[0]) == 3  # goto + input + click(ep)
        assert len(groups[1]) == 2  # input + click(ep)
        assert groups[0][-1].extract_point is True
        assert groups[1][-1].extract_point is True

    @pytest.mark.asyncio
    async def test_multi_entity_caches_html(self):
        """多实体模式：每组执行后缓存 HTML"""
        from agent_scraper.pipeline.tools import NavigateTool

        navigator = MagicMock()
        nav_result_1 = MagicMock()
        nav_result_1.browser = MagicMock()
        nav_result_1.html = "<html>OpenAI page</html>"
        nav_result_2 = MagicMock()
        nav_result_2.browser = nav_result_1.browser
        nav_result_2.html = "<html>Apple page</html>"
        navigator.navigate = AsyncMock(side_effect=[nav_result_1, nav_result_2])

        steps = [
            NavigationStep(action="goto", target="https://wiki.com", description="go"),
            NavigationStep(action="click", target="OpenAI", description="open", extract_point=True),
            NavigationStep(action="click", target="Apple", description="open", extract_point=True),
        ]
        goal = ExtractionGoal(fields={"name": "公司名称"})
        task = ParsedTask(navigation_steps=steps, extraction_goal=goal, raw_instruction="test")
        ctx = AgentContext(task=task)

        # Mock get_current_page_url — called per extract_point group + once at end
        nav_result_1.browser.get_current_page_url = AsyncMock(
            side_effect=["https://wiki.com/OpenAI", "https://wiki.com/Apple", "https://wiki.com/Apple"]
        )

        tool = NavigateTool(navigator)
        result = await tool.execute(ctx)

        assert result.success
        assert len(ctx.html_cache) == 2
        # html_cache stores (url, html) tuples
        assert ctx.html_cache[0][0] == "https://wiki.com/OpenAI"
        assert "OpenAI" in ctx.html_cache[0][1]
        assert ctx.html_cache[1][0] == "https://wiki.com/Apple"
        assert "Apple" in ctx.html_cache[1][1]
        assert navigator.navigate.call_count == 2

    @pytest.mark.asyncio
    async def test_single_entity_no_cache(self):
        """无 extract_point 时不创建缓存"""
        from agent_scraper.pipeline.tools import NavigateTool

        navigator = MagicMock()
        nav_result = MagicMock()
        nav_result.browser = MagicMock()
        nav_result.browser.get_current_page_url = AsyncMock(return_value="https://example.com")
        nav_result.html = "<html>single page</html>"
        navigator.navigate = AsyncMock(return_value=nav_result)

        steps = [
            NavigationStep(action="goto", target="https://example.com", description="go"),
        ]
        goal = ExtractionGoal(fields={"name": "名称"})
        task = ParsedTask(navigation_steps=steps, extraction_goal=goal, raw_instruction="test")
        ctx = AgentContext(task=task)

        tool = NavigateTool(navigator)
        result = await tool.execute(ctx)

        assert result.success
        assert ctx.html_cache == []
        assert ctx.html == "<html>single page</html>"
