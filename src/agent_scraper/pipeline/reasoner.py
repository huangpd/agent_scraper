"""ReAct 推理循环

正常路径程序化执行（零额外 LLM 调用），仅在失败/评估不通过时调用 LLM Replanner。

循环结构:
    Plan（程序化）→ Execute → Observe → Evaluate → Pass? → Format → Done
                                                  → Fail? → Replan（LLM）→ 新计划 → Execute …
"""

import logging
from typing import Any, Callable

from agent_scraper.core.models import ScrapedResult
from agent_scraper.pipeline.context import AgentContext, StepRecord
from agent_scraper.pipeline.evaluator import Evaluator
from agent_scraper.pipeline.tools import ExtractTool, ToolRegistry

logger = logging.getLogger(__name__)


class Reasoner:
    """ReAct 推理器：默认计划 + 按步执行 + 评估 + 重规划"""

    def __init__(
        self,
        tools: ToolRegistry,
        evaluator: Evaluator,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        self.tools = tools
        self.evaluator = evaluator
        self.on_event = on_event or (lambda *a: None)

    def _emit(self, event_type: str, data: dict):
        self.on_event(event_type, data)

    # ── 主循环 ────────────────────────────────────────────

    async def run(self, ctx: AgentContext) -> ScrapedResult:
        plan = self._create_default_plan(ctx.task, has_images=bool(ctx.images))
        logger.info("[Reasoner] 初始计划: %s", [s["tool"] for s in plan])

        for attempt in range(ctx.max_retries):
            if attempt > 0:
                logger.info("[Reasoner] ═══ 重试 #%d ═══", attempt)
            ctx.retry_count = attempt

            # ── Execute: 按计划逐步执行 ──
            plan_ok = await self._execute_plan(ctx, plan)

            if not plan_ok:
                # 计划执行中断（如 capture 降级），外层 continue 自动进入下一轮
                continue

            # ── Evaluate: 仅 extract 模式需要评估 ──
            if ctx.extracted_data and not ctx.captured:
                eval_result = await self.evaluator.evaluate(ctx)
                self._emit("evaluate", {
                    "passed": eval_result.passed,
                    "quality": eval_result.quality_score,
                    "issues": eval_result.issues,
                })

                if eval_result.passed:
                    logger.info(
                        "[Reasoner] 评估通过 (quality=%.2f)", eval_result.quality_score,
                    )
                    break

                # 评估未通过
                logger.warning(
                    "[Reasoner] 评估未通过 (quality=%.2f, issues=%s)",
                    eval_result.quality_score, eval_result.issues,
                )
                if attempt < ctx.max_retries - 1 and eval_result.retry_strategy:
                    if eval_result.retry_strategy == "skip":
                        logger.info("[Reasoner] Replanner 建议 skip，接受当前结果")
                        break
                    plan = self._apply_retry_strategy(ctx, eval_result.retry_strategy)
                    logger.info(
                        "[Reasoner] 重试策略: %s → 新计划: %s",
                        eval_result.retry_strategy, [s["tool"] for s in plan],
                    )
                    continue
                else:
                    logger.info("[Reasoner] 无重试余量，接受当前结果")
                    break
            else:
                # Capture 模式或空数据，不需评估
                break

        # ── Format: 最终格式化 ──
        fmt_tool = self.tools.get("format")
        if fmt_tool:
            logger.info("[Reasoner] 格式化输出...")
            fmt_result = await fmt_tool.execute(ctx)
            ctx.steps.append(StepRecord(tool_name="format", params={}, result=fmt_result))
            if fmt_result.success:
                logger.info("  ✓ %s", fmt_result.summary)
            else:
                logger.warning("  ✗ %s", fmt_result.summary)

        # 确保 result 存在
        if not ctx.result:
            ctx.result = ScrapedResult(data=[], total_count=0, source_url=ctx.source_url)

        # ── Anomaly Detection: 对 URL 字段运行异常检测 ──
        if ctx.result.data and len(ctx.result.data) > 3:
            self._run_anomaly_detection(ctx)

        self._emit("result", {
            "data": ctx.result.data,
            "total": ctx.result.total_count,
            "source_url": ctx.result.source_url,
        })
        return ctx.result

    # ── 异常检测 ──────────────────────────────────────────

    def _run_anomaly_detection(self, ctx: AgentContext):
        """对结果中的 URL 类字段运行异常检测，有异常则发送 anomaly 事件。"""
        from agent_scraper.extraction.anomaly import detect_anomalies

        url_indicators = ("url", "链接", "link", "href")
        fields = ctx.task.extraction_goal.fields

        for field_name, field_desc in fields.items():
            text = (field_name + " " + field_desc).lower()
            if not any(kw in text for kw in url_indicators):
                continue
            entries = [
                str(r[field_name]) for r in ctx.result.data
                if r.get(field_name)
            ]
            if len(entries) <= 3:
                continue
            try:
                result = detect_anomalies(entries)
            except Exception as e:
                logger.warning("[Reasoner] 异常检测失败: %s", e)
                continue
            if result["anomaly_count"] > 0:
                result["field"] = field_name
                self._emit("anomaly", result)
                logger.info(
                    "[Reasoner] 异常检测: 字段 '%s' 发现 %d 条异常",
                    field_name, result["anomaly_count"],
                )

    # ── 计划执行 ──────────────────────────────────────────

    async def _execute_plan(self, ctx: AgentContext, plan: list[dict]) -> bool:
        """执行计划中的每一步。返回 True=全部完成, False=中断需要重规划。"""
        for step_def in plan:
            tool_name = step_def["tool"]
            params = step_def.get("params", {})
            tool = self.tools.get(tool_name)
            if not tool:
                logger.warning("未知工具: %s, 跳过", tool_name)
                continue

            logger.info("[Reasoner] 执行: %s", tool_name)
            self._emit("step", {"tool": tool_name, "status": "running"})

            result = await tool.execute(ctx, **params)
            ctx.steps.append(StepRecord(tool_name=tool_name, params=params, result=result))

            self._emit("step", {
                "tool": tool_name,
                "status": "success" if result.success else "failed",
                "summary": result.summary,
            })

            if result.success:
                logger.info("  ✓ %s", result.summary)
            else:
                logger.warning("  ✗ %s", result.summary)

                # Capture 失败 → 降级到 extract 模式，中断当前计划
                if tool_name == "capture_navigate":
                    logger.info("[Reasoner] Capture 失败，降级到 extract 模式")
                    ctx.captured = {}  # 清空残留的部分捕获值，防止 format 走错分支
                    if ctx.browser:
                        try:
                            await ctx.browser.stop()
                        except Exception:
                            pass
                        ctx.browser = None
                    # 用 extract 计划替换，让外层循环的下一轮执行
                    plan[:] = self._create_extract_plan()
                    return False  # 中断，外层会 continue

                # 其他工具失败：如果还有重试余量，中断计划让外层重试
                if ctx.retry_count < ctx.max_retries - 1:
                    return False

                # 最后一次重试也失败 → 继续执行后续步骤（尽力而为）
                logger.warning("  最后一轮重试，继续执行...")

        return True

    # ── 计划生成 ──────────────────────────────────────────

    @staticmethod
    def _create_default_plan(task, has_images: bool = False) -> list[dict]:
        """根据任务类型生成默认执行计划（不含 format，format 在循环外统一执行）"""
        if task.mode == "capture":
            return [{"tool": "capture_navigate"}]

        # 无样本 + 有截图 → 用 VLM 从截图生成样本，然后走 AutoScraper 路径
        if not task.extraction_goal.samples and has_images:
            return [
                {"tool": "navigate"},
                {"tool": "vision_sample"},
                {"tool": "discover_rules"},
                {"tool": "iterate_pages"},
                {"tool": "extract"},
            ]

        # 自由模式（无样本、无截图）: browser-use Agent 提取
        if not task.extraction_goal.samples:
            # 有遍历提示时仍需发现规则和翻页（每页由 Agent 提取）
            if task.extraction_goal.traversal_hints:
                return [
                    {"tool": "navigate"},
                    {"tool": "discover_rules"},
                    {"tool": "iterate_pages"},
                    {"tool": "extract"},
                ]
            return [
                {"tool": "navigate"},
                {"tool": "extract"},
            ]
        return Reasoner._create_extract_plan()

    @staticmethod
    def _create_extract_plan() -> list[dict]:
        return [
            {"tool": "navigate"},
            {"tool": "discover_rules"},
            {"tool": "iterate_pages"},
            {"tool": "extract"},
        ]

    # ── 重试策略 → 新计划 ────────────────────────────────

    def _apply_retry_strategy(self, ctx: AgentContext, strategy: str) -> list[dict]:
        """根据 Replanner 的策略生成新执行计划"""

        if strategy == "clear_css_cache":
            # 清除 Extractor 缓存，重新遍历+提取（HTML 不再缓存，需重新访问）
            extract_tool = self.tools.get("extract")
            if isinstance(extract_tool, ExtractTool):
                extract_tool.clear_cache()
            ctx.extracted_data = {}  # 清空旧数据
            return [{"tool": "iterate_pages"}]

        if strategy == "retry_navigate":
            # 完全重来：重新导航 + 发现规则 + 遍历 + 提取
            extract_tool = self.tools.get("extract")
            if isinstance(extract_tool, ExtractTool):
                extract_tool.clear_cache()
            return self._create_extract_plan()

        # "skip" 或未知策略 → 不执行任何工具，直接进入 format
        return []
