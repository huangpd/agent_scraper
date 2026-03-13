"""FastAPI 主应用 + WebSocket"""

import asyncio
import io
import json
import logging
import sys
import threading
from pathlib import Path

# Windows 必须用 ProactorEventLoop，browser-use 依赖 asyncio.create_subprocess_exec
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_scraper.pipeline.orchestrator import AgentScraper
from server.task_manager import TaskManager, TaskStatus

load_dotenv()

app = FastAPI(title="Agent Scraper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = TaskManager()

# WebSocket 连接池: task_id -> set of websockets
ws_connections: dict[str, set[WebSocket]] = {}


async def broadcast(task_id: str, event_type: str, data: dict):
    """向所有订阅该任务的 WebSocket 广播消息"""
    msg = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    for ws in list(ws_connections.get(task_id, [])):
        try:
            await ws.send_text(msg)
        except Exception:
            ws_connections[task_id].discard(ws)


class PrintCapture(io.TextIOBase):
    """拦截 print() 输出，1:1 转发为 WebSocket log 事件，同时保留原始终端输出。"""

    def __init__(self, original_stdout, loop, manager_ref):
        self._original = original_stdout
        self._loop = loop
        self._manager = manager_ref
        self._task_id: str | None = None

    def bind(self, task_id: str):
        self._task_id = task_id

    def write(self, text: str):
        self._original.write(text)
        if not self._task_id:
            return len(text)
        line = text.rstrip("\n\r")
        if not line:
            return len(text)
        info = self._manager.get(self._task_id)
        if info:
            event = {"type": "log", "data": {"message": line}}
            info.events.append(event)
            try:
                self._loop.create_task(
                    broadcast(self._task_id, "log", {"message": line})
                )
            except RuntimeError:
                pass
        return len(text)

    def flush(self):
        self._original.flush()

    @property
    def encoding(self):
        return self._original.encoding


class CreateTaskRequest(BaseModel):
    instruction: str
    headless: bool = False
    images: list[str] = []  # base64 data URL 列表（用户标注的参考截图）


@app.post("/api/tasks")
async def create_task(req: CreateTaskRequest):
    task_id = manager.create(req.instruction, req.headless)
    info = manager.get(task_id)

    async def run_scraper():
        info.status = TaskStatus.RUNNING

        loop = asyncio.get_running_loop()

        # 1) print 拦截：所有 print() → 终端 + WebSocket
        capture = PrintCapture(sys.__stdout__, loop, manager)
        capture.bind(task_id)
        old_stdout = sys.stdout
        sys.stdout = capture

        # 2) logging 拦截：劫持所有已有 StreamHandler 的输出流 → PrintCapture
        #    browser-use 等库在导入时就注册了自己的 StreamHandler（指向原始 stdout/stderr），
        #    需要把它们的输出流也劫持到 PrintCapture，才能转发到 WebSocket。
        root_logger = logging.getLogger()
        old_log_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        # 确保 root logger 有一个 handler（agent_scraper 的日志靠它向上传播）
        log_handler: logging.StreamHandler | None = None
        if not root_logger.handlers:
            log_handler = logging.StreamHandler(sys.stdout)
            log_handler.setFormatter(logging.Formatter("%(message)s"))
            log_handler.setLevel(logging.INFO)
            root_logger.addHandler(log_handler)

        # 劫持所有已有 StreamHandler 的输出流 → PrintCapture
        saved_streams: list[tuple[logging.StreamHandler, object]] = []
        all_loggers = [logging.getLogger()] + [
            logging.getLogger(name)
            for name, obj in logging.Logger.manager.loggerDict.items()
            if isinstance(obj, logging.Logger)
        ]
        for lg in all_loggers:
            for h in lg.handlers:
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                    saved_streams.append((h, h.stream))
                    h.stream = sys.stdout  # → PrintCapture

        try:
            def on_event(event_type: str, data: dict):
                info.events.append({"type": event_type, "data": data})
                loop.create_task(broadcast(task_id, event_type, data))

            scraper = AgentScraper(headless=req.headless, on_event=on_event)
            result = await scraper.run(req.instruction, images=req.images)
            info.status = TaskStatus.COMPLETED
            info.result = result.model_dump()
            await broadcast(task_id, "done", {"message": "任务完成"})
        except asyncio.CancelledError:
            info.status = TaskStatus.CANCELLED
            await broadcast(task_id, "error", {"message": "任务已取消"})
        except Exception as e:
            info.status = TaskStatus.FAILED
            info.error = str(e)
            await broadcast(task_id, "error", {"message": str(e)})
        finally:
            # 恢复所有被劫持的 StreamHandler 输出流
            for h, original_stream in saved_streams:
                h.stream = original_stream
            sys.stdout = old_stdout
            if log_handler:
                root_logger.removeHandler(log_handler)
            root_logger.setLevel(old_log_level)

    info._task = asyncio.create_task(run_scraper())
    return {"task_id": task_id}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    info = manager.get(task_id)
    if not info:
        return {"error": "not found"}, 404
    return {
        "task_id": info.task_id,
        "instruction": info.instruction,
        "status": info.status.value,
        "result": info.result,
        "error": info.error,
        "event_count": len(info.events),
    }


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = manager.cancel(task_id)
    return {"cancelled": ok}


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    ws_connections.setdefault(task_id, set()).add(websocket)

    # 发送已有的历史事件（用于重连场景）
    info = manager.get(task_id)
    if info:
        for event in info.events:
            await websocket.send_text(json.dumps(event, ensure_ascii=False))

    try:
        while True:
            await websocket.receive_text()  # 保持连接
    except WebSocketDisconnect:
        ws_connections.get(task_id, set()).discard(websocket)


# 生产模式: serve 前端静态文件
dist_path = Path(__file__).parent.parent.parent / "web" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")
