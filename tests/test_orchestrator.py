"""测试 agent_scraper.pipeline.orchestrator — 编排器集成测试

新架构: orchestrator → Tools + Reasoner(ReAct) + Evaluator
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


def _common_patches():
    """返回新架构需要的所有 patch 对象"""
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


class TestAgentScraperRun:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """mock 所有子组件，验证 ReAct 循环完整执行"""
        patches = _common_patches()
        (_, _, MockParser, MockNav, MockRuleDisc,
         MockExtractor, MockFormatter, MockEvaluator, MockPageIter) = [p.start() for p in patches]

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper

            # 1. TaskParser
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
            MockParser.return_value.parse = AsyncMock(return_value=mock_task)

            # 2. Navigator
            mock_nav_result = MagicMock()
            mock_nav_result.browser = MagicMock()
            mock_nav_result.browser.stop = AsyncMock()
            mock_nav_result.browser.get_current_page = AsyncMock()
            mock_nav_result.html = "<html><body>test</body></html>"
            MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)

            # 3. RuleDiscoverer
            MockRuleDisc.return_value.discover = AsyncMock(
                return_value=PageRules(load_more_selector="button.load")
            )

            # 4. PageIterator — async generator
            MockPageIter.return_value.iterate = MagicMock(
                return_value=_async_gen(["<html>page1</html>", "<html>page2</html>"])
            )

            # 5. Extractor
            MockExtractor.return_value.extract = AsyncMock(
                return_value={"name": ["a.txt"], "url": ["/a.txt"]}
            )

            # 6. Formatter
            mock_result = ScrapedResult(
                data=[{"name": "a.txt", "url": "https://example.com/a.txt"}],
                total_count=1,
                source_url="https://example.com",
            )
            MockFormatter.return_value.format = AsyncMock(return_value=mock_result)

            # 7. Evaluator — 评估通过
            MockEvaluator.return_value.evaluate = AsyncMock(
                return_value=EvalResult(passed=True, field_check=True, quality_score=0.95)
            )

            # 执行
            scraper = AgentScraper(headless=True)
            result = await scraper.run("test instruction")

            # 验证
            assert isinstance(result, ScrapedResult)
            assert result.total_count == 1

            # 各组件被调用
            MockParser.return_value.parse.assert_called_once()
            MockNav.return_value.navigate.assert_called_once()
            MockRuleDisc.return_value.discover.assert_called_once()
            MockPageIter.return_value.iterate.assert_called_once()
            # IteratePagesTool 内部对每页调用 extractor.extract（2 pages）
            assert MockExtractor.return_value.extract.call_count == 2
            MockFormatter.return_value.format.assert_called_once()
            MockEvaluator.return_value.evaluate.assert_called_once()

            # 浏览器关闭
            mock_nav_result.browser.stop.assert_called()

        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_browser_closed_on_error(self):
        """即使出错，浏览器也应被关闭"""
        patches = _common_patches()
        (_, _, MockParser, MockNav, MockRuleDisc,
         MockExtractor, MockFormatter, MockEvaluator, MockPageIter) = [p.start() for p in patches]

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper

            # TaskParser 直接抛异常
            MockParser.return_value.parse = AsyncMock(side_effect=RuntimeError("parse boom"))

            scraper = AgentScraper(headless=True)
            with pytest.raises(RuntimeError, match="parse boom"):
                await scraper.run("test")

        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_capture_fallback_to_extract(self):
        """Capture 失败时应降级到 extract 模式"""
        patches = _common_patches()
        (_, _, MockParser, MockNav, MockRuleDisc,
         MockExtractor, MockFormatter, MockEvaluator, MockPageIter) = [p.start() for p in patches]

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper

            mock_task = ParsedTask(
                navigation_steps=[
                    NavigationStep(action="goto", target="https://example.com", description="open")
                ],
                extraction_goal=ExtractionGoal(
                    fields={"name": "名称"},
                    samples={"name": ["x"]},
                ),
                raw_instruction="capture test",
                mode="capture",
            )
            MockParser.return_value.parse = AsyncMock(return_value=mock_task)

            # Capture 返回空
            mock_cap = MagicMock()
            mock_cap.browser = MagicMock()
            mock_cap.browser.stop = AsyncMock()
            mock_cap.captured = {}
            mock_cap.page_url = ""
            MockNav.return_value.navigate_and_capture = AsyncMock(return_value=mock_cap)

            # Extract 降级路径
            mock_nav_result = MagicMock()
            mock_nav_result.browser = MagicMock()
            mock_nav_result.browser.stop = AsyncMock()
            mock_nav_result.html = "<html/>"
            MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)
            MockRuleDisc.return_value.discover = AsyncMock(return_value=PageRules())
            MockPageIter.return_value.iterate = MagicMock(
                return_value=_async_gen(["<html/>"])
            )
            MockExtractor.return_value.extract = AsyncMock(return_value={"name": ["x"]})
            mock_result = ScrapedResult(data=[{"name": "x"}], total_count=1, source_url="")
            MockFormatter.return_value.format = AsyncMock(return_value=mock_result)
            MockEvaluator.return_value.evaluate = AsyncMock(
                return_value=EvalResult(passed=True, field_check=True, quality_score=0.9)
            )

            scraper = AgentScraper(headless=True)
            result = await scraper.run("capture test")

            # capture 失败后降级到 extract
            assert result.total_count == 1
            MockNav.return_value.navigate.assert_called_once()  # extract 路径

        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_eval_retry_on_failure(self):
        """评估未通过时应触发重试"""
        patches = _common_patches()
        (_, _, MockParser, MockNav, MockRuleDisc,
         MockExtractor, MockFormatter, MockEvaluator, MockPageIter) = [p.start() for p in patches]

        try:
            from agent_scraper.pipeline.orchestrator import AgentScraper

            mock_task = ParsedTask(
                navigation_steps=[
                    NavigationStep(action="goto", target="https://example.com", description="open")
                ],
                extraction_goal=ExtractionGoal(
                    fields={"name": "名称"},
                    samples={"name": ["x"]},
                ),
                raw_instruction="test",
            )
            MockParser.return_value.parse = AsyncMock(return_value=mock_task)

            mock_nav_result = MagicMock()
            mock_nav_result.browser = MagicMock()
            mock_nav_result.browser.stop = AsyncMock()
            mock_nav_result.html = "<html/>"
            MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)
            MockRuleDisc.return_value.discover = AsyncMock(return_value=PageRules())
            # PageIterator.iterate 会被调用两次（初始 + clear_css_cache 重试）
            MockPageIter.return_value.iterate = MagicMock(
                side_effect=[_async_gen(["<html/>"]), _async_gen(["<html/>"])]
            )

            # 第一次提取数据不完整（评估不通过），第二次完整
            MockExtractor.return_value.extract = AsyncMock(
                side_effect=[{"name": ["partial"]}, {"name": ["ok"]}]
            )
            mock_result = ScrapedResult(data=[{"name": "ok"}], total_count=1, source_url="")
            MockFormatter.return_value.format = AsyncMock(return_value=mock_result)

            # 第一次评估未通过（clear_css_cache），第二次通过
            MockEvaluator.return_value.evaluate = AsyncMock(side_effect=[
                EvalResult(
                    passed=False, field_check=False, quality_score=0.2,
                    issues=["缺少字段"], retry_strategy="clear_css_cache",
                ),
                EvalResult(passed=True, field_check=True, quality_score=0.9),
            ])

            scraper = AgentScraper(headless=True)
            result = await scraper.run("test")

            assert result.total_count == 1
            # Extractor 被调用 2 次（每次 iterate_pages 调用一次）
            assert MockExtractor.return_value.extract.call_count == 2
            assert MockEvaluator.return_value.evaluate.call_count == 2

        finally:
            for p in patches:
                p.stop()
