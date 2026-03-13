"""Navigator: Agent 负责浏览器导航和值捕获。

- navigate(): 首次导航，返回 browser + page + HTML
- navigate_and_capture(): 导航 + 直接捕获字段值（capture 模式）
"""

import json
import logging
import os
import re

from browser_use import Agent, Browser, BrowserProfile
from browser_use.llm import ChatOpenAI
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL

from agent_scraper.core.llm import get_model_name
from agent_scraper.core.models import NavigationStep

logger = logging.getLogger(__name__)


class NavigateResult:
    """导航结果：browser + page + HTML"""
    def __init__(self, browser, page, html: str):
        self.browser = browser
        self.page = page
        self.html = html


class CaptureResult:
    """捕获结果：browser + 捕获到的字段值"""
    def __init__(self, browser, captured: dict[str, str], page_url: str):
        self.browser = browser
        self.captured = captured  # {"field_name": "captured_value"}
        self.page_url = page_url


_IMAGE_HINT = "\n\n注意：用户提供了参考截图，红色方框标注的是要操作的目标元素，请据此定位。"


class Navigator:
    def __init__(self, headless: bool = False):
        self.headless = headless

    @staticmethod
    def _convert_images(
        images: list[str] | None,
    ) -> list[ContentPartTextParam | ContentPartImageParam]:
        """将 base64 data URL 列表转为 browser_use 的 ContentPart 类型。"""
        if not images:
            return []
        parts: list[ContentPartTextParam | ContentPartImageParam] = []
        for i, data_url in enumerate(images):
            parts.append(ContentPartTextParam(
                text=f"[参考截图 {i+1}] 红色方框标注的是用户要操作的目标元素位置:",
            ))
            parts.append(ContentPartImageParam(
                image_url=ImageURL(url=data_url, detail="high"),
            ))
        return parts

    def _create_browser(self) -> Browser:
        return Browser(
            browser_profile=BrowserProfile(
                headless=self.headless,
                wait_between_actions=1.0,
            ),
            keep_alive=True,
        )

    def _create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=get_model_name(),
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    async def _run_agent(self, task_text: str, images: list[str] | None = None,
                         max_steps: int | None = None):
        """创建 browser + llm + Agent 并执行，返回 (browser, agent_history)。"""
        browser = self._create_browser()
        llm = self._create_llm()
        sample_images = self._convert_images(images)
        if sample_images:
            task_text += _IMAGE_HINT
        logger.info("Agent 任务:\n%s", task_text)
        agent = Agent(task=task_text, llm=llm, browser=browser, sample_images=sample_images)
        run_kwargs = {"max_steps": max_steps} if max_steps else {}
        history = await agent.run(**run_kwargs)
        return browser, history

    async def navigate(
        self, steps: list[NavigationStep], images: list[str] | None = None
    ) -> NavigateResult:
        """Agent 执行导航步骤，返回 browser + page + 首页 HTML。"""
        agent_steps = [s for s in steps if s.action in ("goto", "click", "wait", "input")]
        if agent_steps:
            task_text = self._format_steps(agent_steps)
            browser, _ = await self._run_agent(task_text, images)
        else:
            browser = self._create_browser()

        page = await browser.get_current_page()
        if not page:
            raise RuntimeError("无法获取浏览器页面")

        html = await page.evaluate("() => document.documentElement.outerHTML")
        logger.info("首页 HTML: %.1f KB", len(html) / 1024)
        return NavigateResult(browser=browser, page=page, html=html)

    async def navigate_and_capture(
        self,
        steps: list[NavigationStep],
        fields: dict[str, str],
        raw_instruction: str = "",
        images: list[str] | None = None,
    ) -> CaptureResult:
        """Capture 模式：导航 + 直接捕获字段值。"""
        if raw_instruction:
            task_text = raw_instruction.strip() + "\n\n" + self._capture_suffix(fields)
        else:
            task_text = self._format_steps(steps) + "\n\n" + self._capture_suffix(fields)

        browser, history = await self._run_agent(task_text, images, max_steps=25)

        captured = self._parse_capture_result(history, fields)
        page_url = await browser.get_current_page_url()
        if not captured:
            captured = self._fallback_capture_from_url(page_url, fields)

        logger.info("捕获结果: %s", captured)
        return CaptureResult(browser=browser, captured=captured, page_url=page_url)

    # ── 任务文本构建 ─────────────────────────────────────

    @staticmethod
    def _format_steps(steps: list[NavigationStep]) -> str:
        """将 NavigationStep 列表转为编号步骤文本。"""
        _actions = {
            "goto": lambda s: f"打开网址 {s.target}",
            "click": lambda s: f"点击 \"{s.target}\"",
            "wait": lambda s: f"等待{s.description}",
            "input": lambda s: f"在 \"{s.target}\" 中输入 \"{s.value}\"",
        }
        lines = []
        for i, step in enumerate(steps, 1):
            fmt = _actions.get(step.action)
            lines.append(f"步骤{i}: {fmt(step) if fmt else step.description}")
        return "\n".join(lines)

    @staticmethod
    def _capture_suffix(fields: dict[str, str]) -> str:
        """生成 capture 模式的尾部指令（读取字段 + done）。"""
        fields_list = "\n".join(f"  - {k}: {v}" for k, v in fields.items())
        json_example = json.dumps({k: f"<{v}>" for k, v in fields.items()}, ensure_ascii=False)
        return f"""完成上述操作后，从页面上读取以下值:
{fields_list}

重要：获取到值后，立刻调用 done 动作完成任务。
在 done 的 text 参数中填入 JSON 格式的结果，例如:
{json_example}"""

    @staticmethod
    def _parse_capture_result(history, fields: dict[str, str]) -> dict[str, str]:
        """从 Agent 执行历史中提取捕获的值。
        优先从 final_result（done 动作）提取 JSON，
        如果 Agent 没有正确 done，则从所有 extracted_content 中搜索。
        """
        # 收集所有可能包含结果的文本（final_result 优先）
        texts = []

        if history and hasattr(history, "final_result"):
            fr = history.final_result()
            if fr:
                texts.append(fr)

        # 从所有步骤的 extracted_content 收集（倒序，最新的优先）
        if history and hasattr(history, "history"):
            for h in reversed(history.history):
                if hasattr(h, "result"):
                    for r in h.result:
                        if hasattr(r, "extracted_content") and r.extracted_content:
                            texts.append(r.extracted_content)

        if not texts:
            texts.append(str(history) if history else "")

        # 策略1: 从文本中尝试提取 JSON 对象
        for text in texts:
            captured = _extract_json_fields(text, fields)
            if captured:
                return captured

        # 策略2: 从文本中直接匹配 URL（兜底，应对 Agent 反复 extract 但不 done 的情况）
        url_fields = {k for k, v in fields.items()
                      if any(kw in k.lower() or kw in v.lower()
                             for kw in ("url", "链接", "link", "地址"))}
        if url_fields:
            for text in texts:
                urls = re.findall(r'https?://\S+', text)
                if urls:
                    captured = {}
                    # 取最长的 URL（通常是完整的下载链接）
                    best_url = max(urls, key=len)
                    for k in url_fields:
                        captured[k] = best_url
                    if captured:
                        logger.info("从 extract 历史中匹配到 URL: %s...", best_url[:80])
                        return captured

        return {}

    @staticmethod
    def _fallback_capture_from_url(page_url: str, fields: dict[str, str]) -> dict[str, str]:
        """兜底捕获：用当前页面 URL 填充 URL 类字段"""
        captured = {}
        if not page_url or page_url == "about:blank":
            return captured
        for field_name, field_desc in fields.items():
            desc_lower = field_desc.lower()
            name_lower = field_name.lower()
            if any(kw in desc_lower or kw in name_lower for kw in ("url", "链接", "link", "地址")):
                captured[field_name] = page_url
        return captured


def _extract_json_fields(text: str, fields: dict[str, str]) -> dict[str, str]:
    """从文本中提取与 fields 匹配的 JSON 对象"""
    try:
        json_matches = []
        for match in re.finditer(r'\{[^{}]*\}', text):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    json_matches.append(obj)
            except json.JSONDecodeError:
                continue

        if json_matches:
            best = max(json_matches, key=lambda d: len(set(d.keys()) & set(fields.keys())))
            captured = {}
            for k in fields:
                if k in best and best[k]:
                    captured[k] = str(best[k])
            if captured:
                return captured
    except Exception:
        pass
    return {}
