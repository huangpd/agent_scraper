"""主编排器: 组装 Tools → 构建 Reasoner → 运行 ReAct 循环"""

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
from agent_scraper.core.llm import LLMService
from agent_scraper.core.trace import trace_scope, get_llm_count
from agent_scraper.pipeline.tools import (
    CaptureNavigateTool,
    DiscoverRulesTool,
    ExtractTool,
    FormatTool,
    IteratePagesTool,
    NavigateTool,
    ToolRegistry,
    VisionSampleTool,
)

logger = logging.getLogger(__name__)


class AgentScraper:
    def __init__(
        self,
        headless: bool = False,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        self.on_event = on_event or (lambda *a: None)

        # ── 1. 初始化统一 LLM 服务 (AOP 层) ──
        self.llm_service = LLMService()

        # ── 2. 底层组件 (注入 llm_service) ──
        self.task_parser = TaskParser(llm_service=self.llm_service)
        navigator = Navigator(headless=headless)
        rule_discoverer = RuleDiscoverer(llm_service=self.llm_service)
        extractor = Extractor(llm_service=self.llm_service)
        formatter = Formatter()

        # ── 3. 注册 Tools ──
        self.registry = ToolRegistry()
        self.registry.register(NavigateTool(navigator))
        self.registry.register(CaptureNavigateTool(navigator))
        self.registry.register(DiscoverRulesTool(rule_discoverer))
        self.registry.register(IteratePagesTool(extractor))
        self.registry.register(ExtractTool(extractor))
        self.registry.register(VisionSampleTool(self.llm_service))
        self.registry.register(FormatTool(formatter))

        # ── 4. 评估器 + 推理器 ──
        evaluator = Evaluator(llm_service=self.llm_service)
        self.reasoner = Reasoner(
            tools=self.registry,
            evaluator=evaluator,
            on_event=self.on_event,
        )

    async def run(
        self, instruction: str, images: list[str] | None = None,
    ) -> ScrapedResult:
        # ── 开启追踪作用域：本次任务的所有日志都会带上同一个 ID ──
        with trace_scope() as tid:
            ctx: AgentContext | None = None
            try:
                # 1. 解析自然语言 → 结构化任务
                logger.info("[AgentScraper][#%s] 开始新任务解析...", tid)
                task = await self.task_parser.parse(instruction)

                fields = list(task.extraction_goal.fields.keys())
                logger.info("  模式: %s", task.mode)
                logger.info("  导航步骤: %d 步", len(task.navigation_steps))
                logger.info("  提取字段: %s", fields)

                # 2. 构建共享上下文
                ctx = AgentContext(
                    task=task,
                    images=images or [],
                    on_event=self.on_event,
                )

                # 3. ReAct 循环
                result = await self.reasoner.run(ctx)

                # 4. 打印统计信息
                total_calls = get_llm_count()
                logger.info("\n" + "="*50)
                logger.info(f"[#{tid}] 任务完成！")
                logger.info(f"总计 LLM 调用次数: {total_calls}")
                logger.info("="*50 + "\n")

                return result
            except Exception as e:
                total_calls = get_llm_count()
                logger.error(f"[#%s] 任务执行失败 (已调用 LLM %d 次): %s", tid, total_calls, str(e), exc_info=True)
                self.on_event("error", {"message": str(e)})
                raise
            finally:
                if ctx and ctx.browser:
                    try:
                        await ctx.browser.stop()
                    except Exception:
                        pass
