"""主编排器: 组装 Tools → 构建 Reasoner → 运行 ReAct 循环

对外 API 不变:
    scraper = AgentScraper(headless=False, on_event=callback)
    result  = await scraper.run(instruction, images=[...])
"""

import logging
from typing import Any, Callable

from agent_scraper.browser.navigator import Navigator
from agent_scraper.core.models import ScrapedResult
from agent_scraper.extraction.extractor import Extractor
from agent_scraper.extraction.formatter import Formatter
from agent_scraper.extraction.rule_discoverer import RuleDiscoverer
from agent_scraper.pipeline.context import AgentContext
from agent_scraper.pipeline.evaluator import Evaluator
from agent_scraper.pipeline.reasoner import Reasoner
from agent_scraper.pipeline.task_parser import TaskParser
from agent_scraper.pipeline.tools import (
    CaptureNavigateTool,
    DiscoverRulesTool,
    ExtractTool,
    FormatTool,
    IteratePagesTool,
    NavigateTool,
    ToolRegistry,
)

logger = logging.getLogger(__name__)


class AgentScraper:
    def __init__(
        self,
        headless: bool = False,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        self.on_event = on_event or (lambda *a: None)

        from agent_scraper.core.llm import create_openai_client, get_model_name
        client = create_openai_client()
        model = get_model_name()

        # ── 1. 底层组件（不改动）──
        self.task_parser = TaskParser(client=client)
        navigator = Navigator(headless=headless)
        rule_discoverer = RuleDiscoverer(client=client)
        extractor = Extractor(client=client)
        formatter = Formatter()

        # ── 2. 注册 Tools ──
        self.registry = ToolRegistry()
        self.registry.register(NavigateTool(navigator))
        self.registry.register(CaptureNavigateTool(navigator))
        self.registry.register(DiscoverRulesTool(rule_discoverer))
        self.registry.register(IteratePagesTool(extractor))
        self.registry.register(ExtractTool(extractor))
        self.registry.register(FormatTool(formatter))

        # ── 3. 评估器 + 推理器 ──
        evaluator = Evaluator(client=client, model=model)
        self.reasoner = Reasoner(
            tools=self.registry,
            evaluator=evaluator,
            on_event=self.on_event,
        )

    async def run(
        self, instruction: str, images: list[str] | None = None,
    ) -> ScrapedResult:
        ctx: AgentContext | None = None
        try:
            # 1. 解析自然语言 → 结构化任务
            logger.info("[AgentScraper] 解析指令...")
            task = await self.task_parser.parse(instruction)

            fields = list(task.extraction_goal.fields.keys())
            logger.info("  模式: %s", task.mode)
            logger.info("  导航步骤: %d 步", len(task.navigation_steps))
            logger.info("  提取字段: %s", fields)
            if task.extraction_goal.samples:
                sample_count = len(next(iter(task.extraction_goal.samples.values())))
                logger.info("  用户样本: %d 条", sample_count)
            else:
                logger.info("  用户样本: 无（建议提供 JSON 样本以提高准确率）")
            if task.extraction_goal.traversal_hints:
                logger.info("  遍历模式: %s", task.extraction_goal.traversal_hints)
            else:
                logger.info("  遍历模式: 单页（无遍历）")

            # 2. 构建共享上下文
            ctx = AgentContext(
                task=task,
                images=images or [],
                on_event=self.on_event,
            )

            # 3. ReAct 循环
            return await self.reasoner.run(ctx)

        except Exception as e:
            self.on_event("error", {"message": str(e)})
            raise
        finally:
            if ctx and ctx.browser:
                try:
                    await ctx.browser.stop()
                except Exception:
                    pass
