"""一键启动 Agent Scraper Web UI"""

import asyncio
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# Windows 必须用 ProactorEventLoop 才能支持 asyncio.create_subprocess_exec
# browser-use 库依赖此 API 启动浏览器进程
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def main():
    parser = argparse.ArgumentParser(description="Agent Scraper Web UI")
    parser.add_argument("--prod", action="store_true", help="生产模式 (serve 静态文件)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    web_dir = ROOT / "web"

    if args.prod:
        # 生产模式: 先 build 前端，再启动 FastAPI
        print("[*] 构建前端...")
        subprocess.run(["npm", "run", "build"], cwd=str(web_dir), check=True, shell=True)
        print(f"[*] 启动 FastAPI on {args.host}:{args.port}")
        import uvicorn
        uvicorn.run("server.app:app", host=args.host, port=args.port)
    else:
        # 开发模式: 同时启动 Vite dev server + FastAPI
        # 注意: 不使用 reload=True，因为 reload 会 spawn 子进程并重置事件循环策略
        print("[*] 开发模式: Vite(5173) + FastAPI(8000)")
        vite_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(web_dir),
            shell=True,
        )
        try:
            import uvicorn
            uvicorn.run("server.app:app", host=args.host, port=args.port)
        finally:
            vite_proc.terminate()


if __name__ == "__main__":
    main()
