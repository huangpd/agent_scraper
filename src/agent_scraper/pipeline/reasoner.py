"""Explicit FSM orchestrator for the scraping pipeline.

Keeps tool logic unchanged while making state transitions declarative and observable.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Callable
import warnings

from agent_scraper.core.models import ScrapedResult
from agent_scraper.pipeline.context import AgentContext, StepRecord
from agent_scraper.pipeline.evaluator import Evaluator
from agent_scraper.pipeline.rule_cache import RuleCache
from agent_scraper.pipeline.tools import ExtractTool, ToolRegistry
from agent_scraper.extraction.anomaly_detector import detect_anomalies

logger = logging.getLogger(__name__)


class State(Enum):
    ROUTING = auto()
    CAPTURING = auto()
    NAVIGATING = auto()
    DISCOVERING = auto()
    ITERATING = auto()
    EXTRACTING = auto()
    EVALUATING = auto()
    FORMATTING = auto()
    DONE = auto()


class Event(Enum):
    # --- 成功 ---
    OK = auto()
    OK_CAPTURE = auto()       # routing 选择 capture 模式
    OK_FAST = auto()          # 跳过 discover/iterate 快速路径
    # --- 失败（handler 只返回语义事件，由主循环统一判断是否耗尽） ---
    FAIL = auto()
    FAIL_EXHAUSTED = auto()   # 仅由主循环生成，handler 不应直接返回
    # --- 评估 ---
    EVAL_PASS = auto()
    EVAL_FAIL_CLEAR = auto()
    EVAL_FAIL_NAV = auto()
    EVAL_SKIP = auto()
    # --- 降级 ---
    CAPTURE_DEGRADE = auto()


# handler 返回这些事件时消耗一次重试；耗尽时主循环将其替换为 FAIL_EXHAUSTED
_RETRY_EVENTS = frozenset({
    Event.FAIL,
    Event.CAPTURE_DEGRADE,
    Event.EVAL_FAIL_CLEAR,
    Event.EVAL_FAIL_NAV,
})


TRANSITIONS: dict[tuple[State, Event], State] = {
    # ROUTING — 进入表，不再需要 _next_state 特殊处理
    (State.ROUTING, Event.OK): State.NAVIGATING,
    (State.ROUTING, Event.OK_CAPTURE): State.CAPTURING,

    # CAPTURING
    (State.CAPTURING, Event.OK): State.FORMATTING,
    (State.CAPTURING, Event.CAPTURE_DEGRADE): State.NAVIGATING,
    (State.CAPTURING, Event.FAIL_EXHAUSTED): State.FORMATTING,

    # NAVIGATING
    (State.NAVIGATING, Event.OK): State.DISCOVERING,
    (State.NAVIGATING, Event.OK_FAST): State.EXTRACTING,    # 跳过 discover+iterate
    (State.NAVIGATING, Event.FAIL): State.NAVIGATING,       # 重试
    (State.NAVIGATING, Event.FAIL_EXHAUSTED): State.FORMATTING,

    # DISCOVERING
    (State.DISCOVERING, Event.OK): State.ITERATING,
    (State.DISCOVERING, Event.OK_FAST): State.EXTRACTING,    # 跳过 iterate
    (State.DISCOVERING, Event.FAIL): State.NAVIGATING,
    (State.DISCOVERING, Event.FAIL_EXHAUSTED): State.FORMATTING,

    # ITERATING
    (State.ITERATING, Event.OK): State.EXTRACTING,
    (State.ITERATING, Event.FAIL): State.EXTRACTING,         # 降级单页提取
    (State.ITERATING, Event.FAIL_EXHAUSTED): State.FORMATTING,

    # EXTRACTING
    (State.EXTRACTING, Event.OK): State.EVALUATING,
    (State.EXTRACTING, Event.EVAL_SKIP): State.FORMATTING,
    (State.EXTRACTING, Event.FAIL): State.NAVIGATING,
    (State.EXTRACTING, Event.FAIL_EXHAUSTED): State.FORMATTING,

    # EVALUATING
    (State.EVALUATING, Event.EVAL_PASS): State.FORMATTING,
    (State.EVALUATING, Event.EVAL_SKIP): State.FORMATTING,
    (State.EVALUATING, Event.EVAL_FAIL_CLEAR): State.ITERATING,
    (State.EVALUATING, Event.EVAL_FAIL_NAV): State.NAVIGATING,
    (State.EVALUATING, Event.FAIL_EXHAUSTED): State.FORMATTING,

    # FORMATTING
    (State.FORMATTING, Event.OK): State.DONE,
    (State.FORMATTING, Event.FAIL): State.DONE,
}


class Reasoner:
    """FSM-driven orchestrator; per-state handlers emit events consumed by TRANSITIONS."""

    def __init__(
        self,
        tools: ToolRegistry,
        evaluator: Evaluator,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
        rule_cache: RuleCache | None = None,
    ):
        self.tools = tools
        self.evaluator = evaluator
        self.on_event = on_event or (lambda *a: None)
        self._rule_cache = rule_cache or RuleCache()
        self._handlers = {
            State.ROUTING: self._do_route,
            State.CAPTURING: self._do_capture,
            State.NAVIGATING: self._do_navigate,
            State.DISCOVERING: self._do_discover,
            State.ITERATING: self._do_iterate,
            State.EXTRACTING: self._do_extract,
            State.EVALUATING: self._do_evaluate,
            State.FORMATTING: self._do_format,
        }
        self._retries_left: int = 0

    def _emit(self, event_type: str, data: dict):
        self.on_event(event_type, data)

    # -- main loop -----------------------------------------------------

    async def run(self, ctx: AgentContext) -> ScrapedResult:
        self._retries_left = ctx.max_retries
        state = State.ROUTING
        safety_counter = 0
        SAFETY_LIMIT = 64

        logger.info("[Reasoner] FSM start, retries=%d", self._retries_left)

        while state != State.DONE:
            if safety_counter >= SAFETY_LIMIT:
                raise RuntimeError("FSM safety limit exceeded")
            safety_counter += 1

            handler = self._handlers[state]
            event = await handler(ctx)

            # 统一 retry 判断：handler 只返回语义事件，此处决定是否耗尽
            if event in _RETRY_EVENTS:
                if self._retries_left > 0:
                    self._retries_left -= 1
                    ctx.retry_count = ctx.max_retries - self._retries_left
                else:
                    event = Event.FAIL_EXHAUSTED

            next_state = self._next_state(state, event)
            self._emit("transition", {
                "from": state.name, "event": event.name, "to": next_state.name,
                "retries_left": self._retries_left,
            })
            state = next_state

        if not ctx.result:
            ctx.result = ScrapedResult(data=[], total_count=0, source_url=ctx.source_url)

        self._emit("result", {
            "data": ctx.result.data,
            "total": ctx.result.total_count,
            "source_url": ctx.result.source_url,
        })
        return ctx.result

    # -- transition helpers -------------------------------------------

    def _next_state(self, state: State, event: Event) -> State:
        """纯表查询，无特殊分支。"""
        key = (state, event)
        if key not in TRANSITIONS:
            raise RuntimeError(f"Illegal transition: {state} + {event}")
        return TRANSITIONS[key]

    async def _run_tool(self, name: str, ctx: AgentContext, **params) -> Any:
        tool = self.tools.get(name)
        if not tool:
            return None
        self._emit("step", {"tool": name, "status": "running"})
        result = await tool.execute(ctx, **params)
        ctx.steps.append(StepRecord(tool_name=name, params=params, result=result))
        self._emit("step", {
            "tool": name,
            "status": "success" if result.success else "failed",
            "summary": result.summary,
        })
        if result.success:
            logger.info("  %s", result.summary)
        else:
            logger.warning("  %s", result.summary)
        return result

    # -- state handlers -----------------------------------------------

    async def _do_route(self, ctx: AgentContext) -> Event:
        task = ctx.task
        ctx.skip_discover = False
        ctx.skip_iterate = False

        if task.mode == "capture":
            return Event.OK_CAPTURE

        # 记录初始是否有样本（用于后续判断是否触发缓存保存）
        ctx.had_samples_initially = bool(task.extraction_goal.samples)

        # 无样本时查缓存
        if not task.extraction_goal.samples:
            source_url = task.navigation_steps[0].target if task.navigation_steps else ""
            hit = self._rule_cache.lookup(source_url, task.extraction_goal.fields)
            if hit:
                task.extraction_goal.samples = hit.samples
                if hit.page_rules:
                    ctx.page_rules = hit.page_rules
                ctx.skip_discover = True
                logger.info("[Reasoner] 缓存命中，注入 %d 个字段的样本",
                            len(hit.samples))
                return Event.OK

        if not task.extraction_goal.samples and not task.extraction_goal.traversal_hints:
            ctx.skip_discover = True
            ctx.skip_iterate = True
        return Event.OK

    async def _do_capture(self, ctx: AgentContext) -> Event:
        result = await self._run_tool("capture_navigate", ctx)
        if result and result.success:
            return Event.OK

        ctx.captured = {}
        if ctx.browser:
            try:
                await ctx.browser.stop()
            except Exception:
                pass
            ctx.browser = None

        logger.info("[Reasoner] capture failed, degrade to extract mode")
        return Event.CAPTURE_DEGRADE

    async def _do_navigate(self, ctx: AgentContext) -> Event:
        result = await self._run_tool("navigate", ctx)
        if not result or not result.success:
            return Event.FAIL
        if ctx.skip_discover:
            return Event.OK_FAST
        return Event.OK

    async def _do_discover(self, ctx: AgentContext) -> Event:
        result = await self._run_tool("discover_rules", ctx)
        if not result or not result.success:
            return Event.FAIL
        if ctx.skip_iterate:
            return Event.OK_FAST
        return Event.OK

    async def _do_iterate(self, ctx: AgentContext) -> Event:
        result = await self._run_tool("iterate_pages", ctx)
        if result and result.success:
            return Event.OK
        return Event.FAIL

    async def _do_extract(self, ctx: AgentContext) -> Event:
        result = await self._run_tool("extract", ctx)
        if result and result.success:
            if ctx.captured or not ctx.extracted_data:
                return Event.EVAL_SKIP
            return Event.OK
        return Event.FAIL

    async def _do_evaluate(self, ctx: AgentContext) -> Event:
        if ctx.captured or not ctx.extracted_data:
            return Event.EVAL_SKIP

        eval_result = await self.evaluator.evaluate(ctx)
        self._emit("evaluate", {
            "passed": eval_result.passed,
            "quality": eval_result.quality_score,
            "issues": eval_result.issues,
        })

        if eval_result.passed:
            logger.info("[Reasoner] evaluation passed (quality=%.2f)", eval_result.quality_score)
            return Event.EVAL_PASS

        strategy = eval_result.retry_strategy
        if strategy == "skip":
            logger.info("[Reasoner] replanner suggests skip; accept current result")
            return Event.EVAL_SKIP
        if strategy == "clear_css_cache":
            self._clear_extract_cache(ctx)
            ctx.extracted_data = []
            return Event.EVAL_FAIL_CLEAR
        if strategy == "retry_navigate":
            self._clear_extract_cache(ctx)
            return Event.EVAL_FAIL_NAV

        return Event.FAIL

    async def _do_format(self, ctx: AgentContext) -> Event:
        result = await self._run_tool("format", ctx)
        if result and result.success:
            # 自由模式提取成功 → 训练 AutoScraper → 缓存
            if not ctx.had_samples_initially and ctx.extracted_data:
                self._learn_and_cache(ctx)
            self._maybe_emit_anomalies(ctx)
            return Event.OK
        return Event.FAIL

    def _clear_extract_cache(self, ctx: AgentContext):
        extract_tool = self.tools.get("extract")
        if isinstance(extract_tool, ExtractTool):
            extract_tool.clear_cache()

    # -- rule cache ----------------------------------------------------

    def _learn_and_cache(self, ctx: AgentContext):
        """自由模式提取成功后：训练 AutoScraper → 验证 → 保存缓存。"""
        from agent_scraper.rule_learner import AutoScraper

        if not ctx.html or not ctx.extracted_data:
            return

        task = ctx.task
        source_url = ctx.source_url or (
            task.navigation_steps[0].target if task.navigation_steps else ""
        )

        # 1. extracted_data[:3] → wanted_dict
        preview = ctx.extracted_data[:3]
        wanted_dict: dict[str, list[str]] = {}
        for row in preview:
            for k, v in row.items():
                if k not in wanted_dict:
                    wanted_dict[k] = []
                wanted_dict[k].append(str(v))

        if not wanted_dict:
            return

        # 2. 训练 AutoScraper
        try:
            scraper = AutoScraper()
            scraper.build(html=ctx.html, wanted_dict=wanted_dict)
        except Exception as e:
            logger.warning("[Reasoner] 规则学习失败: %s", e)
            return

        if not scraper.stack_list:
            logger.info("[Reasoner] 规则学习未产出 stack，跳过缓存")
            return

        # 3. 验证：用训练好的 scraper 在同一 HTML 上提取，检查结果数量
        try:
            verify = scraper.get_result_similar(html=ctx.html, group_by_alias=True)
            if not verify:
                logger.info("[Reasoner] 规则验证无结果，跳过缓存")
                return
            # 取最长列的结果数，对比原始数据量
            max_verify = max(len(v) for v in verify.values()) if verify else 0
            original_count = len(ctx.extracted_data)
            if max_verify < original_count * 0.5:
                logger.info(
                    "[Reasoner] 规则验证不足: %d/%d (< 50%%)，跳过缓存",
                    max_verify, original_count,
                )
                return
        except Exception as e:
            logger.warning("[Reasoner] 规则验证异常: %s", e)
            return

        # 4. 保存缓存
        try:
            self._rule_cache.save(
                url=source_url,
                fields=task.extraction_goal.fields,
                scraper=scraper,
                page_rules=ctx.page_rules,
                sample_data=ctx.extracted_data,
            )
            self._emit("cache_saved", {
                "source_url": source_url,
                "fields": list(task.extraction_goal.fields.keys()),
                "stack_count": len(scraper.stack_list),
            })
        except Exception as e:
            logger.warning("[Reasoner] 缓存保存失败: %s", e)

    # -- anomaly hint --------------------------------------------------

    def _maybe_emit_anomalies(self, ctx: AgentContext):
        """Run anomaly ensemble on URL fields and emit hint event if suspicious."""
        records = ctx.result.data if ctx.result else []
        if not records:
            return

        # Collect URL-like fields
        urls: list[str] = []
        for row in records:
            for k, v in row.items():
                if not isinstance(v, str):
                    continue
                if "url" in k.lower():
                    urls.append(v)
        if len(urls) <= 3:
            return

        n = len(urls)
        top_k = min(50, max(5, int(n * 0.01)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="sklearn.neighbors._lof",
            )
            report = detect_anomalies(urls, top_k=top_k)

        if report["anomaly_count"] > 0:
            logger.info("[Reasoner] anomaly detected: %d items", report["anomaly_count"])
            self._emit("anomaly", {
                "summary": report["summary"],
                "anomaly_count": report["anomaly_count"],
                "categories": report["categories"],
                "items": report["items"],
            })
        else:
            logger.info("[Reasoner] anomaly check complete: none")
