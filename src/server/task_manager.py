"""内存任务管理器"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    task_id: str
    instruction: str
    headless: bool
    status: TaskStatus = TaskStatus.PENDING
    result: dict | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _task: asyncio.Task | None = field(default=None, repr=False)


class TaskManager:
    def __init__(self):
        self.tasks: dict[str, TaskInfo] = {}

    def create(self, instruction: str, headless: bool = True) -> str:
        task_id = uuid.uuid4().hex[:8]
        self.tasks[task_id] = TaskInfo(
            task_id=task_id,
            instruction=instruction,
            headless=headless,
        )
        return task_id

    def get(self, task_id: str) -> TaskInfo | None:
        return self.tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        info = self.tasks.get(task_id)
        if info and info._task and not info._task.done():
            info._task.cancel()
            info.status = TaskStatus.CANCELLED
            return True
        return False
