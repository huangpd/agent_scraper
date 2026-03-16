"""测试 orchestrator 事件回调机制

新架构事件模型:
- "step"     : Reasoner 每执行一个 Tool 时发出
- "evaluate" : Evaluator 完成评估后发出
- "progress" : IteratePagesTool 每提取一页时发出
- "result"   : Reasoner 完成后发出最终结果
- "error"    : 异常时由 orchestrator 发出
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_scraper.core.models import (
    ExtractionGoal,
    NavigationStep,
    PageRules,
    ParsedTask,
    ScrapedResult,
)
from agent_scraper.pipeline.evaluator import EvalResult


async def _async_gen(items):
    for item in items:
        yield item


def _make_mocks():
    mock_task = ParsedTask(
        navigation_steps=[
            NavigationStep(action="goto", target="https://example.com", description="open")
        ],
        extraction_goal=ExtractionGoal(
            fields={"name": "文件名", "url": "链接"},
            traversal_hints=["load_more"],
            samples={"name": ["a.txt"], "url": ["/a.txt"]},
        ),
        raw_instruction="test",
    )
    mock_nav_result = MagicMock()
    mock_nav_result.browser = MagicMock()
    mock_nav_result.browser.stop = AsyncMock()
    mock_nav_result.html = "<html><body>test</body></html>"

    mock_result = ScrapedResult(
        data=[{"name": "a.txt", "url": "https://example.com/a.txt"}],
        total_count=1,
        source_url="https://example.com",
    )
    return mock_task, mock_nav_result, mock_result


def _patch_all():
    return (
        patch("agent_scraper.core.llm.create_openai_client"),
        patch("agent_scraper.core.llm.get_model_name", return_value="test-model"),
        patch("agent_scraper.pipeline.orchestrator.TaskParser"),
        patch("agent_scraper.pipeline.orchestrator.Navigator"),
        patch("agent_scraper.pipeline.orchestrator.RuleDiscoverer"),
        patch("agent_scraper.pipeline.orchestrator.Extractor"),
        patch("agent_scraper.pipeline.orchestrator.Formatter"),
        patch("agent_scraper.pipeline.orchestrator.Evaluator"),
        patch("agent_scraper.browser.page_iterator.PageIterator"),
    )


def _setup_mocks(patches, mock_task, mock_nav_result, mock_result, pages=None):
    (_, _, MockParser, MockNav, MockRuleDisc,
     MockExtractor, MockFormatter, MockEvaluator, MockPageIter) = [p.start() for p in patches]

    MockParser.return_value.parse = AsyncMock(return_value=mock_task)
    MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)
    MockRuleDisc.return_value.discover = AsyncMock(return_value=PageRules())
    MockPageIter.return_value.iterate = MagicMock(
        return_value=_async_gen(pages or ["<html/>"])
    )
    MockExtractor.return_value.extract = AsyncMock(return_value={"name": ["x"], "url": ["/x"]})
    MockFormatter.return_value.format = AsyncMock(return_value=mock_result)
    MockEvaluator.return_value.evaluate = AsyncMock(
        return_value=EvalResult(passed=True, field_check=True, quality_score=0.95)
    )

    return MockParser


class TestEventCallback:
    @pytest.mark.asyncio
    async def test_emits_step_and_result(self):
        """on_event 应收到 step、evaluate、progress 和 result 事件"""
        mock_task, mock_nav_result, mock_result = _make_mocks()
        events = []
        patches = _patch_all()
        _setup_mocks(patches, mock_task, mock_nav_result, mock_result)

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            scraper = AgentScraper(headless=True, on_event=lambda t, d: events.append((t, d)))
            await scraper.run("test")
        finally:
            for p in patches:
                p.stop()

        event_types = [e[0] for e in events]
        assert "step" in event_types       # Reasoner 每步发出
        assert "result" in event_types     # 最终结果
        assert "progress" in event_types   # ExtractTool 每页发出
        assert "evaluate" in event_types   # Evaluator 结果

    @pytest.mark.asyncio
    async def test_progress_events_for_multiple_pages(self):
        """多页面时应发出多条 progress 事件"""
        mock_task, mock_nav_result, mock_result = _make_mocks()
        events = []
        patches = _patch_all()
        _setup_mocks(patches, mock_task, mock_nav_result, mock_result,
                     pages=["<p>1</p>", "<p>2</p>", "<p>3</p>"])

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            scraper = AgentScraper(headless=True, on_event=lambda t, d: events.append((t, d)))
            await scraper.run("test")
        finally:
            for p in patches:
                p.stop()

        progress_events = [e[1] for e in events if e[0] == "progress"]
        assert len(progress_events) == 3
        # 流式模式下 total 未知（0）
        assert progress_events[0] == {"current": 1, "total": 0}
        assert progress_events[2] == {"current": 3, "total": 0}

    @pytest.mark.asyncio
    async def test_error_event_on_exception(self):
        """解析阶段异常时应发出 error 事件"""
        mock_task, mock_nav_result, _ = _make_mocks()
        events = []
        patches = _patch_all()
        MockParser = _setup_mocks(patches, mock_task, mock_nav_result, None)
        # TaskParser 在 Reasoner 之前执行，异常会直接传播
        MockParser.return_value.parse = AsyncMock(side_effect=RuntimeError("boom"))

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            scraper = AgentScraper(headless=True, on_event=lambda t, d: events.append((t, d)))
            with pytest.raises(RuntimeError):
                await scraper.run("test")
        finally:
            for p in patches:
                p.stop()

        error_events = [e for e in events if e[0] == "error"]
        assert len(error_events) == 1
        assert "boom" in error_events[0][1]["message"]

    @pytest.mark.asyncio
    async def test_result_event_contains_data(self):
        """result 事件应包含 data, total, source_url"""
        mock_task, mock_nav_result, mock_result = _make_mocks()
        events = []
        patches = _patch_all()
        _setup_mocks(patches, mock_task, mock_nav_result, mock_result)

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            scraper = AgentScraper(headless=True, on_event=lambda t, d: events.append((t, d)))
            await scraper.run("test")
        finally:
            for p in patches:
                p.stop()

        result_events = [e[1] for e in events if e[0] == "result"]
        assert len(result_events) == 1
        r = result_events[0]
        assert r["total"] == 1
        assert r["source_url"] == "https://example.com"
        assert isinstance(r["data"], list)

    @pytest.mark.asyncio
    async def test_no_event_callback_does_not_crash(self):
        """不传 on_event 时不应报错"""
        mock_task, mock_nav_result, mock_result = _make_mocks()
        patches = _patch_all()
        _setup_mocks(patches, mock_task, mock_nav_result, mock_result)

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            scraper = AgentScraper(headless=True)
            result = await scraper.run("test")
            assert result.total_count == 1
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_step_events_include_tool_names(self):
        """step 事件应包含每个 Tool 的名称"""
        mock_task, mock_nav_result, mock_result = _make_mocks()
        events = []
        patches = _patch_all()
        _setup_mocks(patches, mock_task, mock_nav_result, mock_result)

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            scraper = AgentScraper(headless=True, on_event=lambda t, d: events.append((t, d)))
            await scraper.run("test")
        finally:
            for p in patches:
                p.stop()

        step_events = [e[1] for e in events if e[0] == "step"]
        tool_names = {e["tool"] for e in step_events}
        # extract 模式应包含这些 tool
        assert "navigate" in tool_names
        assert "discover_rules" in tool_names
        assert "extract" in tool_names
