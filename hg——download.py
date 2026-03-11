import asyncio
import json
import os
from pydantic import BaseModel
from browser_use import Agent, Browser, BrowserProfile, Controller
from browser_use.llm import ChatOpenAI
from playwright.async_api import Page
from stagehand import Stagehand

# ============================================================
# Task 统一管理
# ============================================================
TASK = {
    "url": "https://huggingface.co/Qwen/Qwen3.5-397B-A17B",
    "agent_task": """
        步骤1: 打开网址 https://huggingface.co/Qwen/Qwen3.5-397B-A17B
        步骤2: 找到并点击 "Files and versions" 标签页
        步骤3: 等待文件列表加载完成
        步骤4: 如果页面有 "Load more files" 请一直点击，直到按钮消失
        步骤5: 提取文件列表中所有文件的文件名、完整下载URL（resolve/main格式）用json格式 ,https://huggingface.co/Qwen/模型名/resolve/main/文件名
        步骤6: 确认 action 返回了 HTML 大小信息后，标记任务完成
    """,
    "extract_instruction": "提取文件列表中所有文件的文件名、完整下载URL（resolve/main格式）和文件大小",
    "total_size_gb": 807,
}

# ============================================================
# Schema
# ============================================================
class FileItem(BaseModel):
    filename: str
    url: str
    size: str | None = None

class FileList(BaseModel):
    files: list[FileItem]

# ============================================================
# Step 1: browser-use 负责导航 + Load more
#         Custom Action 拿完整渲染 HTML
# ============================================================
controller = Controller()
rendered_html = None  # 全局存储渲染结果

@controller.action("获取渲染HTML")
async def get_rendered_html(page: Page):
    global rendered_html
    # 循环点击 Load more 确保全部加载
    click_count = 0
    while True:
        try:
            btn = await page.query_selector("button:has-text('Load more'), a:has-text('Load more')")
            if not btn:
                print(f"  ✅ Load more 完毕，共点击 {click_count} 次")
                break
            await btn.scroll_into_view_if_needed()
            await btn.click()
            await page.wait_for_timeout(1500)
            click_count += 1
            print(f"  🔄 第 {click_count} 次 Load more...")
        except Exception:
            break

    rendered_html = await page.content()
    print(f"  📄 获取到完整渲染 HTML，大小: {len(rendered_html) / 1024:.1f} KB")
    return f"HTML已获取，大小{len(rendered_html)/1024:.1f}KB"


async def run_browser_agent():
    """browser-use 只负责导航和加载"""
    # ✅ 修复1: 使用 async with 上下文管理器，避免手动 close() 报错
    async with Browser(
        browser_profile=BrowserProfile(
            headless=False,
            wait_between_actions=1.0,
        )
    ) as browser:
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        agent = Agent(
            task=TASK["agent_task"],
            llm=llm,
            browser=browser,
            controller=controller,
        )
        await agent.run()
    # 退出 async with 时自动释放浏览器资源


# ============================================================
# Step 2: Stagehand 负责结构化提取
# ============================================================
async def run_stagehand_extract(html: str) -> list[FileItem]:
    """Stagehand 只负责从 HTML 提取结构化数据"""
    async with Stagehand() as sh:
        await sh.page.set_content(html)
        print("🤖 Stagehand 正在提取结构化数据...")
        result = await sh.page.extract(
            instruction=TASK["extract_instruction"],
            schema=FileList,
        )
        return result.files


# ============================================================
# 主流程
# ============================================================
async def main():
    global rendered_html

    # Step 1: browser-use 导航 + 拿 HTML
    print("🌐 browser-use 开始导航...")
    await run_browser_agent()

    # ✅ 修复2: Agent 未调用自定义 action 时给出明确提示
    if not rendered_html:
        print("❌ 未能获取渲染 HTML —— Agent 可能没有调用 '获取渲染HTML' action")
        print("   建议检查任务描述，或在 get_rendered_html 中添加保底兜底逻辑")
        return

    # Step 2: Stagehand 提取
    files = await run_stagehand_extract(rendered_html)

    output = [f.model_dump() for f in files]
    with open("qwen_files.json", "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)

    # 校验总大小
    total_gb = 0
    for f in files:
        if not f.size:
            continue
        s = f.size.upper()
        try:
            num = float("".join(c for c in s if c.isdigit() or c == "."))
            if "GB" in s:    total_gb += num
            elif "MB" in s:  total_gb += num / 1024
            elif "KB" in s:  total_gb += num / 1024 / 1024
        except Exception:
            pass

    in_range = abs(total_gb - TASK["total_size_gb"]) / TASK["total_size_gb"] < 0.05
    print(f"\n{'='*50}")
    print(f"📁 文件数量  : {len(files)} 个")
    print(f"💾 总大小    : {total_gb:.1f} GB（预期 {TASK['total_size_gb']} GB）")
    print(f"✅ 完整性校验: {'通过 ✅' if in_range else '不匹配 ❌，可能有遗漏'}")
    print(f"💾 已保存到  : qwen_files.json")


if __name__ == "__main__":
    asyncio.run(main())