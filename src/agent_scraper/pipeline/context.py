"""ReAct 循环共享上下文：ToolResult / StepRecord / AgentContext"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_scraper.core.models import PageRules, ParsedTask, ScrapedResult
from agent_scraper.pipeline.retry_escalator import FailureMemory

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: str | None = None
    summary: str = ""


@dataclass
class StepRecord:
    """单步执行记录"""
    tool_name: str
    params: dict
    result: ToolResult


@dataclass
class AgentContext:
    """ReAct 循环的共享状态，所有 Tool 通过它读写数据"""
    task: ParsedTask
    images: list[str] = field(default_factory=list)
    on_event: Callable[[str, dict[str, Any]], Any] = field(default=lambda *a: None)

    # ── 可变状态（由 Tool 写入）──
    browser: Any = None
    html: str = ""
    page_rules: PageRules | None = None
    extracted_data: dict[str, list] = field(default_factory=dict)
    result: ScrapedResult | None = None
    source_url: str = ""

    # Capture 模式专用
    captured: dict[str, str] = field(default_factory=dict)

    # 多实体模式：缓存每个 extract_point 的 (url, html)
    html_cache: list[tuple[str, str]] = field(default_factory=list)

    # ── 执行历史 ──
    steps: list[StepRecord] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 4  # 支持 L0-L2 三级重试

    # ── 重试记忆 ──
    failure_memory: FailureMemory = field(default_factory=FailureMemory)

    # ── 安全的状态修改方法 ──

    def set_samples(self, samples: dict[str, list[str]]):
        """安全设置样本并发出事件（替代直接操作 goal.samples）"""
        self.task.extraction_goal.samples = samples
        self.on_event("state_change", {"field": "samples", "value": {k: len(v) for k, v in samples.items()}})
        logger.info("[AgentContext] 设置样本: %s", {k: len(v) for k, v in samples.items()})

    def checkpoint(self) -> dict:
        """快照当前可恢复状态（重试前调用）"""
        return {
            "extracted_data": copy.deepcopy(self.extracted_data),
            "samples": copy.deepcopy(self.task.extraction_goal.samples),
        }

    def rollback(self, snapshot: dict):
        """回滚到快照（重试失败时恢复）"""
        self.extracted_data = snapshot["extracted_data"]
        self.task.extraction_goal.samples = snapshot["samples"]
        logger.info("[AgentContext] 状态已回滚")
