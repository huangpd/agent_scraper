import contextvars
import uuid
from contextlib import contextmanager

# 异步上下文变量
_trace_id_var = contextvars.ContextVar("trace_id", default="system")
_llm_call_count_var = contextvars.ContextVar("llm_call_count", default=0)

def get_trace_id() -> str:
    return _trace_id_var.get()

def increment_llm_count():
    """累加 LLM 调用次数"""
    current = _llm_call_count_var.get()
    _llm_call_count_var.set(current + 1)

def get_llm_count() -> int:
    """获取当前任务的 LLM 调用总数"""
    return _llm_call_count_var.get()

@contextmanager
def trace_scope(trace_id: str | None = None):
    """开启一个追踪范围，并重置计数器"""
    token_id = _trace_id_var.set(trace_id or str(uuid.uuid4())[:8])
    token_count = _llm_call_count_var.set(0) # 每个新任务从 0 开始计数
    try:
        yield _trace_id_var.get()
    finally:
        _trace_id_var.reset(token_id)
        _llm_call_count_var.reset(token_count)
