"""ReAct 循环共享上下文：ToolResult / StepRecord / AgentContext"""

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_scraper.core.models import PageRules, ParsedTask, ScrapedResult


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

    # ── 执行历史 ──
    steps: list[StepRecord] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
