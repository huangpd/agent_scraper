"""测试 server.task_manager — 内存任务管理器"""

import asyncio

import pytest
from server.task_manager import TaskInfo, TaskManager, TaskStatus


class TestTaskManager:
    def test_create_returns_unique_ids(self):
        tm = TaskManager()
        id1 = tm.create("instruction 1")
        id2 = tm.create("instruction 2")
        assert id1 != id2
        assert len(id1) == 8
        assert len(id2) == 8

    def test_get_returns_task_info(self):
        tm = TaskManager()
        task_id = tm.create("test instruction", headless=False)
        info = tm.get(task_id)
        assert info is not None
        assert info.task_id == task_id
        assert info.instruction == "test instruction"
        assert info.headless is False
        assert info.status == TaskStatus.PENDING
        assert info.result is None
        assert info.error is None
        assert info.events == []

    def test_get_nonexistent_returns_none(self):
        tm = TaskManager()
        assert tm.get("nonexistent") is None

    def test_create_default_headless_true(self):
        tm = TaskManager()
        task_id = tm.create("test")
        info = tm.get(task_id)
        assert info.headless is True

    def test_cancel_without_async_task(self):
        """没有关联 asyncio.Task 时 cancel 返回 False"""
        tm = TaskManager()
        task_id = tm.create("test")
        assert tm.cancel(task_id) is False

    def test_cancel_nonexistent_task(self):
        tm = TaskManager()
        assert tm.cancel("nonexistent") is False

    @pytest.mark.asyncio
    async def test_cancel_running_task(self):
        """关联了 asyncio.Task 后 cancel 应成功"""
        tm = TaskManager()
        task_id = tm.create("test")
        info = tm.get(task_id)

        async def long_running():
            await asyncio.sleep(100)

        info._task = asyncio.create_task(long_running())
        assert tm.cancel(task_id) is True
        assert info.status == TaskStatus.CANCELLED
        # 等待 task 被 cancel 完成
        with pytest.raises(asyncio.CancelledError):
            await info._task
        assert info._task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_already_done_task(self):
        """已完成的 asyncio.Task cancel 应返回 False"""
        tm = TaskManager()
        task_id = tm.create("test")
        info = tm.get(task_id)

        async def fast():
            return 42

        info._task = asyncio.create_task(fast())
        await info._task  # 等待完成
        assert tm.cancel(task_id) is False

    def test_multiple_tasks_independent(self):
        tm = TaskManager()
        id1 = tm.create("task A")
        id2 = tm.create("task B")
        info1 = tm.get(id1)
        info2 = tm.get(id2)
        assert info1.instruction == "task A"
        assert info2.instruction == "task B"
        # 修改一个不影响另一个
        info1.status = TaskStatus.RUNNING
        assert info2.status == TaskStatus.PENDING


class TestTaskInfo:
    def test_events_append(self):
        info = TaskInfo(task_id="abc", instruction="test", headless=True)
        info.events.append({"type": "step", "data": {"step": 1}})
        assert len(info.events) == 1

    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
