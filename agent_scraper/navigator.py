"""Navigator: Agent 只负责首次导航（打开URL、点击标签等需要LLM理解的操作）
返回 browser + page，供后续 PageIterator 使用。
"""

import os

from browser_use import Agent, Browser, BrowserProfile
from browser_use.llm import ChatOpenAI

from agent_scraper.models import NavigationStep


class NavigateResult:
    """导航结果：browser + page + HTML"""
    def __init__(self, browser, page, html: str):
        self.browser = browser
        self.page = page
        self.html = html


class Navigator:
    def __init__(self, headless: bool = False):
        self.headless = headless

    async def navigate(self, steps: list[NavigationStep]) -> NavigateResult:
        """
        Agent 只做需要 LLM 理解的步骤（goto, click, wait）。
        返回 browser + page + 首页 HTML，浏览器保持存活。
        """
        browser = Browser(
            browser_profile=BrowserProfile(
                headless=self.headless,
                wait_between_actions=1.0,
            ),
            keep_alive=True,
        )
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # 只把需要 LLM 理解的步骤交给 Agent
        agent_steps = [s for s in steps if s.action in ("goto", "click", "wait")]
        if agent_steps:
            task_text = self._build_agent_task(agent_steps)
            print(f"[Navigator] Agent 任务:\n{task_text}\n")
            agent = Agent(task=task_text, llm=llm, browser=browser)
            await agent.run()

        page = await browser.get_current_page()
        if not page:
            raise RuntimeError("无法获取浏览器页面")

        html = await page.evaluate("() => document.documentElement.outerHTML")
        print(f"  [Navigator] 首页 HTML: {len(html) / 1024:.1f} KB")

        return NavigateResult(browser=browser, page=page, html=html)

    @staticmethod
    def _build_agent_task(steps: list[NavigationStep]) -> str:
        lines = []
        for i, step in enumerate(steps, 1):
            if step.action == "goto":
                lines.append(f"步骤{i}: 打开网址 {step.target}")
            elif step.action == "click":
                lines.append(f"步骤{i}: 点击 \"{step.target}\"")
            elif step.action == "wait":
                lines.append(f"步骤{i}: 等待{step.description}")
            else:
                lines.append(f"步骤{i}: {step.description}")
        return "\n".join(lines)
