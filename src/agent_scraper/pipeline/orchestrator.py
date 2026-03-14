"""主编排器: Navigator → RuleDiscoverer → PageIterator → Extractor → Formatter"""

import logging
from typing import Any, Callable

from agent_scraper.browser.navigator import Navigator
from agent_scraper.browser.page_iterator import PageIterator
from agent_scraper.core.models import ScrapedResult
from agent_scraper.extraction.extractor import Extractor
from agent_scraper.extraction.formatter import Formatter
from agent_scraper.extraction.rule_discoverer import RuleDiscoverer
from agent_scraper.pipeline.task_parser import TaskParser

logger = logging.getLogger(__name__)


class AgentScraper:
    def __init__(self, headless: bool = False, on_event: Callable[[str, dict[str, Any]], Any] | None = None):
        self.on_event = on_event or (lambda *a: None)
        from agent_scraper.core.llm import create_openai_client
        client = create_openai_client()
        self.task_parser = TaskParser(client=client)
        self.navigator = Navigator(headless=headless)
        self.rule_discoverer = RuleDiscoverer(client=client)
        self.extractor = Extractor(client=client)
        self.formatter = Formatter()

    def _emit(self, event_type: str, data: dict[str, Any]):
        self.on_event(event_type, data)

    async def run(self, instruction: str, images: list[str] | None = None) -> ScrapedResult:
        self._images = images or []
        # 1. 解析自然语言 → 结构化任务

        logger.info("[AgentScraper] 步骤1: 解析指令...")
        task = await self.task_parser.parse(instruction)
        fields = list(task.extraction_goal.fields.keys())
        logger.info(f"  模式: {task.mode}")
        logger.info(f"  导航步骤: {len(task.navigation_steps)} 步")
        logger.info(f"  提取字段: {fields}")
        if task.extraction_goal.samples:
            sample_count = len(list(task.extraction_goal.samples.values())[0])
            logger.info(f"  用户样本: {sample_count} 条")
        else:
            logger.info("  用户样本: 无（建议在指令中提供 JSON 样本以提高准确率）")
        if task.extraction_goal.traversal_hints:
            logger.info(f"  遍历模式: {task.extraction_goal.traversal_hints}")
        else:
            logger.info("  遍历模式: 单页（无遍历）")

        # 推断 source_url
        source_url = ""
        for step in task.navigation_steps:
            if step.action == "goto":
                source_url = step.target
                break

        # 根据模式分流
        if task.mode == "capture":
            return await self._run_capture(task, source_url)
        else:
            return await self._run_extract(task, source_url)

    async def _run_capture(self, task, source_url: str) -> ScrapedResult:
        """Capture 模式：浏览器导航 + 直接捕获值，跳过 HTML 提取流水线"""
        logger.info("[AgentScraper] Capture 模式: Agent 导航并捕获值...")

        cap = await self.navigator.navigate_and_capture(
            task.navigation_steps,
            task.extraction_goal.fields,
            raw_instruction=task.raw_instruction,
            images=self._images,
        )

        try:
            if cap.captured:
                logger.info(f"  >>> 捕获到 {len(cap.captured)} 个字段: {list(cap.captured.keys())}")
                data = [cap.captured]
                result = ScrapedResult(
                    data=data,
                    total_count=1,
                    source_url=cap.page_url or source_url,
                )
            else:
                logger.info("  >>> 未捕获到任何值，尝试 extract 模式兜底...")
                return await self._run_extract(task, source_url)

            logger.info(f"[AgentScraper] Capture 完成! 捕获 {len(cap.captured)} 个字段")
            self._emit("result", {"data": result.data, "total": result.total_count, "source_url": result.source_url})
            return result

        except Exception as e:
            self._emit("error", {"message": str(e)})
            raise

        finally:
            try:
                await cap.browser.stop()
            except Exception:
                pass

    async def _run_extract(self, task, source_url: str) -> ScrapedResult:
        """Extract 模式：完整 6 步流水线"""
        # 2. Agent 首次导航 → browser + page + HTML
        logger.info("[AgentScraper] 步骤2: Agent 首次导航...")
        nav = await self.navigator.navigate(task.navigation_steps, images=self._images)

        try:
            # 3. AI 发现页面遍历规则（仅用户要求遍历时才调用 LLM）
            logger.info("[AgentScraper] 步骤3: 分析页面规则...")
            rules = await self.rule_discoverer.discover(
                nav.html, source_url, task.extraction_goal.traversal_hints
            )
            rules_info = []
            if rules.load_more_selector:
                rules_info.append(f"load_more='{rules.load_more_selector}'")
            if rules.sub_page_selector:
                rules_info.append(f"sub_page='{rules.sub_page_selector}' (attr={rules.sub_page_url_attr}, recursive={rules.sub_page_recursive})")
            if rules.next_button_selector:
                rules_info.append(f"next_button='{rules.next_button_selector}'")
            if rules.pagination_url:
                rules_info.append(f"pagination='{rules.pagination_url}' (max={rules.pagination_max})")
            rules_summary = ', '.join(rules_info) if rules_info else '无(单页模式)'
            logger.info(f"  >>> 规则: {rules_summary}")

            # 4. 代码执行规则，遍历所有页面
            logger.info("[AgentScraper] 步骤4: 代码执行遍历规则...")
            iterator = PageIterator(nav.browser)
            all_htmls = await iterator.iterate(nav.html, rules, source_url)

            # 5. 对每个 HTML 提取数据
            logger.info(f"[AgentScraper] 步骤5: 提取数据 ({len(all_htmls)} 个页面)...")
            all_data: dict[str, list] = {}
            for i, html in enumerate(all_htmls):
                logger.info(f"  提取页面 [{i + 1}/{len(all_htmls)}] ({len(html)/1024:.0f}KB)...")
                self._emit("progress", {"current": i + 1, "total": len(all_htmls)})
                page_data = await self.extractor.extract(html, task.extraction_goal)
                for key, values in page_data.items():
                    all_data.setdefault(key, []).extend(values)
                logger.info(f"    → 字段: { {k: len(v) for k, v in page_data.items()} }")

            logger.info(f"  >>> 总计: { {k: len(v) for k, v in all_data.items()} }")

            # 6. 格式化输出
            logger.info("[AgentScraper] 步骤6: 格式化输出...")
            result = await self.formatter.format(all_data, task.extraction_goal, source_url)

            logger.info(f"[AgentScraper] 完成! 提取 {result.total_count} 条数据")
            self._emit("result", {"data": result.data, "total": result.total_count, "source_url": result.source_url})
            return result

        except Exception as e:
            self._emit("error", {"message": str(e)})
            raise

        finally:
            try:
                await nav.browser.stop()
            except Exception:
                pass
