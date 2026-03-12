"""测试 orchestrator 事件回调机制
orchestrator 只 emit 三种事件: progress, result, error
其余日志通过 print() 输出（由 server 的 PrintCapture 转发到前端）
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


def _make_mocks():
    mock_task = ParsedTask(
        navigation_steps=[
            NavigationStep(action="goto", target="https://example.com", description="open")
        ],
        extraction_goal=ExtractionGoal(
            fields={"name": "文件名", "url": "链接"},
            traversal_hints=["load_more"],
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
        patch("agent_scraper.pipeline.orchestrator.TaskParser"),
        patch("agent_scraper.pipeline.orchestrator.Navigator"),
        patch("agent_scraper.pipeline.orchestrator.RuleDiscoverer"),
        patch("agent_scraper.pipeline.orchestrator.Extractor"),
        patch("agent_scraper.pipeline.orchestrator.Formatter"),
        patch("agent_scraper.pipeline.orchestrator.PageIterator"),
    )


def _setup_mocks(patches, mock_task, mock_nav_result, mock_result, pages=None):
    (_, MockParser, MockNav, MockRuleDisc,
     MockExtractor, MockFormatter, MockPageIter) = [p.start() for p in patches]

    MockParser.return_value.parse = AsyncMock(return_value=mock_task)
    MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)
    MockRuleDisc.return_value.discover = AsyncMock(return_value=PageRules())
    MockPageIter.return_value.iterate = AsyncMock(return_value=pages or ["<html/>"])
    MockExtractor.return_value.extract = AsyncMock(return_value={"name": ["x"]})
    MockFormatter.return_value.format = AsyncMock(return_value=mock_result)

    return MockRuleDisc


class TestEventCallback:
    @pytest.mark.asyncio
    async def test_emits_progress_and_result(self):
        """on_event 应收到 progress 和 result 事件"""
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
        assert "progress" in event_types
        assert "result" in event_types
        # step/log 不再 emit，由 print 输出
        assert "step" not in event_types
        assert "log" not in event_types

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
        assert progress_events[0] == {"current": 1, "total": 3}
        assert progress_events[2] == {"current": 3, "total": 3}

    @pytest.mark.asyncio
    async def test_error_event_on_exception(self):
        """异常时应发出 error 事件"""
        mock_task, mock_nav_result, _ = _make_mocks()
        events = []
        patches = _patch_all()
        MockRuleDisc = _setup_mocks(patches, mock_task, mock_nav_result, None)
        MockRuleDisc.return_value.discover = AsyncMock(side_effect=RuntimeError("boom"))

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
    async def test_log_output_contains_all_steps(self, caplog):
        """日志应包含所有 6 个步骤"""
        import logging
        mock_task, mock_nav_result, mock_result = _make_mocks()
        patches = _patch_all()
        _setup_mocks(patches, mock_task, mock_nav_result, mock_result)

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper
            with caplog.at_level(logging.INFO, logger="agent_scraper.pipeline.orchestrator"):
                scraper = AgentScraper(headless=True)
                await scraper.run("test")
        finally:
            for p in patches:
                p.stop()

        output = caplog.text
        for step_num in range(1, 7):
            assert f"步骤{step_num}" in output
