"""用 browser-use 自定义 action 提取 HuggingFace 页面的下载链接。

使用官方 @tools.action() 模式注册 Playwright 提取工具，
Agent 自主决定 selector 并调用工具完成提取。
"""

import asyncio
import json
import os

from pydantic import BaseModel, Field
from browser_use import ActionResult, Agent, Browser, BrowserProfile, Tools
from browser_use.llm import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class PlaywrightExtractLinksAction(BaseModel):
    """提取链接的参数模型"""
    selector: str = Field(description="用于定位下载链接元素的 CSS 选择器或 XPath (例如 'a[download]', 'xpath=//a[contains(@href,\"resolve\")]')")


class PlaywrightGetTextAction(BaseModel):
    """提取文本的参数模型"""
    selector: str = Field(description="用于定位元素的 CSS 选择器或 XPath")


# ── 收集提取结果 ──────────────────────────────────────────────
collected_links: list[dict] = []


async def test_hf_extract():
    target_url = "https://huggingface.co/openai-community/gpt2/tree/main"

    print(f"\n[*] 目标页面: {target_url}")

    # ── 注册自定义 action ─────────────────────────────────────
    tools = Tools()

    @tools.action(
        "用 Playwright 选择器提取页面中的所有匹配链接，返回文件名和 href。",
        param_model=PlaywrightExtractLinksAction,
    )
    async def playwright_extract_links(params: PlaywrightExtractLinksAction, browser_session):
        """Agent 调用此工具，传入 selector，批量提取链接。"""
        try:
            page = browser_session.current_page
            if not page:
                return ActionResult(error="浏览器页面不可用")

            elements = page.locator(params.selector)
            count = await elements.count()

            if count == 0:
                return ActionResult(error=f"未找到匹配元素: {params.selector}")

            links = []
            for i in range(count):
                el = elements.nth(i)
                href = await el.get_attribute("href") or ""
                text = (await el.inner_text()).strip()
                tag = await el.evaluate("el => el.tagName")
                links.append({
                    "file_name": text or f"[unnamed-{i}]",
                    "href": href,
                    "tag": tag,
                })

            collected_links.extend(links)

            summary = f"提取到 {len(links)} 个链接"
            return ActionResult(
                extracted_content=json.dumps(links, ensure_ascii=False, indent=2),
                include_in_memory=True,
            )
        except Exception as e:
            return ActionResult(error=f"提取失败: {e}")

    @tools.action(
        "用 Playwright 选择器提取元素文本内容。",
        param_model=PlaywrightGetTextAction,
    )
    async def playwright_get_text(params: PlaywrightGetTextAction, browser_session):
        """Agent 调用此工具，传入 selector，提取文本。"""
        try:
            page = browser_session.current_page
            if not page:
                return ActionResult(error="浏览器页面不可用")

            if params.selector.lower() == "title":
                text = await page.title()
                return ActionResult(extracted_content=f"页面标题: {text}")

            element = page.locator(params.selector).first
            if await element.count() == 0:
                return ActionResult(error=f"未找到元素: {params.selector}")

            text_content = await element.text_content()
            inner_text = await element.inner_text()
            tag_name = await element.evaluate("el => el.tagName")
            is_visible = await element.is_visible()

            result_data = {
                "selector": params.selector,
                "text_content": text_content,
                "inner_text": inner_text,
                "tag_name": tag_name,
                "is_visible": is_visible,
            }
            return ActionResult(
                extracted_content=json.dumps(result_data, ensure_ascii=False),
                include_in_memory=True,
            )
        except Exception as e:
            return ActionResult(error=f"文本提取失败: {e}")

    # ── 创建 Agent ────────────────────────────────────────────
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    browser = Browser(
        browser_profile=BrowserProfile(
            headless=False,
            wait_between_actions=1.0,
            minimum_wait_page_load_time=3.0,
        ),
    )

    task = f"""
    1. 打开 HuggingFace 模型文件页: {target_url}
    2. 等待页面加载完成，观察文件列表结构。
    3. 使用 playwright_extract_links 工具提取所有文件的下载链接。
       - 先观察页面 DOM，找到包含文件链接的 <a> 元素的共同特征（class、href 模式等）
       - 构造合适的 CSS 选择器或 XPath，调用工具批量提取
       - HF 文件页的下载链接通常包含 /resolve/ 路径或指向具体文件的 <a> 标签
    4. 如果页面有 "Load more" 或 "Expand" 按钮，先点击展开再提取。
    5. 提取完成后使用 done 动作结束。
    """

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        tools=tools,
    )

    print("[*] Agent 启动，将自主分析页面并调用 Playwright 工具提取链接...")
    try:
        history = await agent.run(max_steps=20)

        # ── 输出结果 ──────────────────────────────────────────
        print("\n" + "=" * 60)
        if collected_links:
            print(f"[OK] 共提取到 {len(collected_links)} 个链接:\n")
            for item in collected_links:
                print(f"  {item['file_name']:<40s} {item['href']}")

            with open("hf_links.json", "w", encoding="utf-8") as f:
                json.dump(collected_links, f, ensure_ascii=False, indent=2)
            print(f"\n[*] 结果已保存到 hf_links.json")
        else:
            print("[!] 未通过工具提取到链接，检查 Agent 历史...")
            result = history.final_result()
            if result:
                print(f"Agent 输出: {result[:500]}")
        print("=" * 60)
    finally:
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(test_hf_extract())
