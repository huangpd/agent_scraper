"""测试 agent_scraper.navigator — 导航模块"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# mock browser_use 模块（测试环境可能未安装）
if "browser_use" not in sys.modules:
    browser_use_mock = MagicMock()
    sys.modules["browser_use"] = browser_use_mock
    sys.modules["browser_use.llm"] = browser_use_mock.llm

from agent_scraper.browser.navigator import Navigator, NavigateResult, CaptureResult
from agent_scraper.core.models import NavigationStep


class TestFormatSteps:
    def test_goto(self):
        steps = [NavigationStep(action="goto", target="https://example.com", description="打开")]
        text = Navigator._format_steps(steps)
        assert "步骤1" in text
        assert "打开网址" in text
        assert "https://example.com" in text

    def test_click(self):
        steps = [NavigationStep(action="click", target="Files tab", description="点击")]
        text = Navigator._format_steps(steps)
        assert "点击" in text
        assert "Files tab" in text

    def test_wait(self):
        steps = [NavigationStep(action="wait", target="", description="页面加载完成")]
        text = Navigator._format_steps(steps)
        assert "等待" in text
        assert "页面加载完成" in text

    def test_multiple_steps(self):
        steps = [
            NavigationStep(action="goto", target="https://example.com", description="打开"),
            NavigationStep(action="click", target="Login", description="登录"),
            NavigationStep(action="wait", target="", description="加载完成"),
        ]
        text = Navigator._format_steps(steps)
        assert "步骤1" in text
        assert "步骤2" in text
        assert "步骤3" in text

    def test_unknown_action_uses_description(self):
        steps = [NavigationStep(action="scroll", target="", description="向下滚动")]
        text = Navigator._format_steps(steps)
        assert "向下滚动" in text


class TestNavigateResult:
    def test_attributes(self):
        result = NavigateResult(browser="mock_browser", page="mock_page", html="<html/>")
        assert result.browser == "mock_browser"
        assert result.page == "mock_page"
        assert result.html == "<html/>"


class TestCaptureSuffix:
    def test_contains_fields_and_done(self):
        fields = {"url": "下载链接", "name": "文件名"}
        suffix = Navigator._capture_suffix(fields)
        assert "url: 下载链接" in suffix
        assert "name: 文件名" in suffix
        assert "done" in suffix

    def test_json_example(self):
        fields = {"url": "下载链接"}
        suffix = Navigator._capture_suffix(fields)
        assert '"url"' in suffix
        assert "<下载链接>" in suffix


# ── 图片参考功能测试 ──────────────────────────────────────

FAKE_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


class TestConvertImages:
    def test_none_returns_empty(self):
        assert Navigator._convert_images(None) == []

    def test_empty_list_returns_empty(self):
        assert Navigator._convert_images([]) == []

    def test_single_image(self):
        parts = Navigator._convert_images([FAKE_DATA_URL])
        assert len(parts) == 2
        # 第一个是文字说明
        assert parts[0]["type"] == "text"
        assert "参考截图 1" in parts[0]["text"]
        assert "红色方框" in parts[0]["text"]
        # 第二个是图片
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == FAKE_DATA_URL
        assert parts[1]["image_url"]["detail"] == "high"

    def test_multiple_images(self):
        urls = [f"data:image/png;base64,img{i}" for i in range(3)]
        parts = Navigator._convert_images(urls)
        # 每张图 2 个 part（文字 + 图片）
        assert len(parts) == 6
        # 编号递增
        assert "参考截图 1" in parts[0]["text"]
        assert "参考截图 2" in parts[2]["text"]
        assert "参考截图 3" in parts[4]["text"]
        # 图片 URL 正确
        assert parts[1]["image_url"]["url"] == urls[0]
        assert parts[3]["image_url"]["url"] == urls[1]
        assert parts[5]["image_url"]["url"] == urls[2]


class TestImagePassedToAgent:
    """测试图片经 navigate / navigate_and_capture 正确传递到 Agent"""

    @pytest.mark.asyncio
    async def test_navigate_passes_sample_images(self):
        nav = Navigator(headless=True)

        mock_agent_cls = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock()
        mock_agent_cls.return_value = mock_agent_instance

        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value="<html>ok</html>")
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        steps = [NavigationStep(action="goto", target="https://example.com", description="open")]
        images = [FAKE_DATA_URL]

        with patch.object(nav, "_create_browser", return_value=mock_browser), \
             patch.object(nav, "_create_llm", return_value=MagicMock()), \
             patch("agent_scraper.browser.navigator.Agent", mock_agent_cls):
            result = await nav.navigate(steps, images=images)

        # Agent 应被创建且收到 sample_images
        mock_agent_cls.assert_called_once()
        call_kwargs = mock_agent_cls.call_args
        passed_images = call_kwargs.kwargs.get("sample_images") or call_kwargs[1].get("sample_images")
        assert passed_images is not None
        assert len(passed_images) == 2  # 1 text + 1 image_url
        assert passed_images[1]["image_url"]["url"] == FAKE_DATA_URL

        # task_text 应包含图片提示
        task_text = call_kwargs.kwargs.get("task") or call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs["task"]
        assert "参考截图" in task_text

    @pytest.mark.asyncio
    async def test_navigate_no_images(self):
        nav = Navigator(headless=True)

        mock_agent_cls = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock()
        mock_agent_cls.return_value = mock_agent_instance

        mock_browser = MagicMock()
        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock(return_value="<html>ok</html>")
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        steps = [NavigationStep(action="goto", target="https://example.com", description="open")]

        with patch.object(nav, "_create_browser", return_value=mock_browser), \
             patch.object(nav, "_create_llm", return_value=MagicMock()), \
             patch("agent_scraper.browser.navigator.Agent", mock_agent_cls):
            await nav.navigate(steps, images=None)

        call_kwargs = mock_agent_cls.call_args
        passed_images = call_kwargs.kwargs.get("sample_images") or call_kwargs[1].get("sample_images")
        assert passed_images == []

        task_text = call_kwargs.kwargs.get("task") or call_kwargs.kwargs["task"]
        assert "参考截图" not in task_text

    @pytest.mark.asyncio
    async def test_capture_passes_sample_images(self):
        nav = Navigator(headless=True)

        mock_agent_cls = MagicMock()
        mock_agent_instance = MagicMock()
        mock_history = MagicMock()
        mock_history.final_result.return_value = '{"url": "https://example.com/file"}'
        mock_history.history = []
        mock_agent_instance.run = AsyncMock(return_value=mock_history)
        mock_agent_cls.return_value = mock_agent_instance

        mock_browser = MagicMock()
        mock_browser.get_current_page_url = AsyncMock(return_value="https://example.com")

        steps = [NavigationStep(action="goto", target="https://example.com", description="open")]
        fields = {"url": "下载链接"}
        images = [FAKE_DATA_URL, "data:image/jpeg;base64,second"]

        with patch.object(nav, "_create_browser", return_value=mock_browser), \
             patch.object(nav, "_create_llm", return_value=MagicMock()), \
             patch("agent_scraper.browser.navigator.Agent", mock_agent_cls):
            result = await nav.navigate_and_capture(
                steps, fields, raw_instruction="获取下载链接", images=images,
            )

        call_kwargs = mock_agent_cls.call_args
        passed_images = call_kwargs.kwargs.get("sample_images") or call_kwargs[1].get("sample_images")
        assert len(passed_images) == 4  # 2 images × 2 parts each
        assert "参考截图 1" in passed_images[0]["text"]
        assert "参考截图 2" in passed_images[2]["text"]

        task_text = call_kwargs.kwargs.get("task") or call_kwargs.kwargs["task"]
        assert "参考截图" in task_text

        assert isinstance(result, CaptureResult)
