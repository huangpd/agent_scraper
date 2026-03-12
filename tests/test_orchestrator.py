"""测试 agent_scraper.pipeline.orchestrator — 编排器集成测试"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agent_scraper.core.models import (
    ExtractionGoal,
    NavigationStep,
    PageRules,
    ParsedTask,
    ScrapedResult,
)


class TestAgentScraperRun:
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """mock 所有子组件，验证流水线完整执行"""
        # 避免直接 import AgentScraper（会触发 openai import），用 patch 替代
        with patch("agent_scraper.core.llm.create_openai_client"), \
             patch("agent_scraper.pipeline.orchestrator.TaskParser") as MockParser, \
             patch("agent_scraper.pipeline.orchestrator.Navigator") as MockNav, \
             patch("agent_scraper.pipeline.orchestrator.RuleDiscoverer") as MockRuleDisc, \
             patch("agent_scraper.pipeline.orchestrator.Extractor") as MockExtractor, \
             patch("agent_scraper.pipeline.orchestrator.Formatter") as MockFormatter, \
             patch("agent_scraper.pipeline.orchestrator.PageIterator") as MockPageIter:

            from agent_scraper.pipeline.orchestrator import AgentScraper

            # 1. TaskParser mock
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
            MockParser.return_value.parse = AsyncMock(return_value=mock_task)

            # 2. Navigator mock
            mock_nav_result = MagicMock()
            mock_nav_result.browser = MagicMock()
            mock_nav_result.browser.stop = AsyncMock()
            mock_nav_result.page = MagicMock()
            mock_nav_result.html = "<html><body>test</body></html>"
            MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)

            # 3. RuleDiscoverer mock
            MockRuleDisc.return_value.discover = AsyncMock(
                return_value=PageRules(load_more_selector="button.load")
            )

            # 4. PageIterator mock
            MockPageIter.return_value.iterate = AsyncMock(
                return_value=["<html>page1</html>", "<html>page2</html>"]
            )

            # 5. Extractor mock
            MockExtractor.return_value.extract = AsyncMock(
                return_value={"name": ["a.txt"], "url": ["/a.txt"]}
            )

            # 6. Formatter mock
            mock_result = ScrapedResult(
                data=[{"name": "a.txt", "url": "https://example.com/a.txt"}],
                total_count=1,
                source_url="https://example.com",
            )
            MockFormatter.return_value.format = AsyncMock(return_value=mock_result)

            # 执行
            scraper = AgentScraper(headless=True)
            result = await scraper.run("test instruction")

            # 验证
            assert isinstance(result, ScrapedResult)
            assert result.total_count == 1

            # 验证各组件被调用
            MockParser.return_value.parse.assert_called_once()
            MockNav.return_value.navigate.assert_called_once()
            MockRuleDisc.return_value.discover.assert_called_once()
            MockPageIter.return_value.iterate.assert_called_once()
            # extractor 对每个 HTML 页面调用一次
            assert MockExtractor.return_value.extract.call_count == 2
            MockFormatter.return_value.format.assert_called_once()
            # 浏览器应被关闭
            mock_nav_result.browser.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_browser_closed_on_error(self):
        """即使提取出错，浏览器也应被关闭"""
        with patch("agent_scraper.core.llm.create_openai_client"), \
             patch("agent_scraper.pipeline.orchestrator.TaskParser") as MockParser, \
             patch("agent_scraper.pipeline.orchestrator.Navigator") as MockNav, \
             patch("agent_scraper.pipeline.orchestrator.RuleDiscoverer") as MockRuleDisc, \
             patch("agent_scraper.pipeline.orchestrator.PageIterator"), \
             patch("agent_scraper.pipeline.orchestrator.Extractor"), \
             patch("agent_scraper.pipeline.orchestrator.Formatter"):

            from agent_scraper.pipeline.orchestrator import AgentScraper

            mock_task = ParsedTask(
                navigation_steps=[
                    NavigationStep(action="goto", target="https://example.com", description="open")
                ],
                extraction_goal=ExtractionGoal(fields={"name": "名称"}),
                raw_instruction="test",
            )
            MockParser.return_value.parse = AsyncMock(return_value=mock_task)

            mock_nav_result = MagicMock()
            mock_nav_result.browser = MagicMock()
            mock_nav_result.browser.stop = AsyncMock()
            mock_nav_result.html = "<html/>"
            MockNav.return_value.navigate = AsyncMock(return_value=mock_nav_result)

            # RuleDiscoverer 抛出异常
            MockRuleDisc.return_value.discover = AsyncMock(side_effect=RuntimeError("boom"))

            scraper = AgentScraper(headless=True)
            with pytest.raises(RuntimeError):
                await scraper.run("test")

            # 即使出错，浏览器也应关闭
            mock_nav_result.browser.stop.assert_called_once()
