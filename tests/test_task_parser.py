"""测试 agent_scraper.task_parser — 任务解析"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_scraper.pipeline.task_parser import TaskParser


@pytest.fixture
def mock_llm_service():
    service = MagicMock()
    service.call = AsyncMock()
    return service


@pytest.fixture
def parser(mock_llm_service):
    return TaskParser(llm_service=mock_llm_service)


class TestExtractSamples:
    def test_jsonl_extraction(self):
        instruction = """\
打开 https://example.com 提取文件列表
{"file_name": ".gitattributes", "download_url": "https://hf.co/resolve/main/.gitattributes"}
{"file_name": "config.json", "download_url": "https://hf.co/resolve/main/config.json"}
"""
        samples = TaskParser._extract_samples(instruction)
        assert samples is not None
        assert "file_name" in samples
        assert len(samples["file_name"]) == 2
        assert ".gitattributes" in samples["file_name"]
        assert "download_url" in samples
        assert len(samples["download_url"]) == 2

    def test_no_json(self):
        instruction = "打开 example.com 提取标题"
        assert TaskParser._extract_samples(instruction) is None

    def test_invalid_json_skipped(self):
        instruction = """\
some text
{invalid json}
{"name": "valid"}
"""
        samples = TaskParser._extract_samples(instruction)
        assert samples is not None
        assert "name" in samples
        assert len(samples["name"]) == 1

    def test_non_dict_json_skipped(self):
        instruction = """\
[1, 2, 3]
{"name": "test"}
"""
        samples = TaskParser._extract_samples(instruction)
        assert samples is not None
        assert "name" in samples

    def test_inline_json_no_newlines(self):
        """JSON 对象紧挨在一起无换行（textarea 单行输入场景）"""
        instruction = '提取文件 {"file_name":".gitattributes","download_url":"/main/.gitattributes"}{"file_name":"README.md","download_url":"/main/README.md"}'
        samples = TaskParser._extract_samples(instruction)
        assert samples is not None
        assert len(samples["file_name"]) == 2
        assert ".gitattributes" in samples["file_name"]
        assert "README.md" in samples["file_name"]
        assert len(samples["download_url"]) == 2

    def test_inline_json_mixed_with_text(self):
        """JSON 对象散落在文本中"""
        instruction = '步骤1: 打开 https://example.com\n样本: {"name":"a.txt","url":"/a.txt"} 和 {"name":"b.txt","url":"/b.txt"}'
        samples = TaskParser._extract_samples(instruction)
        assert samples is not None
        assert len(samples["name"]) == 2

    def test_single_field_json_ignored_by_regex(self):
        """只有1个字段的 JSON 被正则路径过滤（避免误匹配 CSS 等）"""
        instruction = '使用 {"selector": "div.file"} 选择器'
        samples = TaskParser._extract_samples(instruction)
        assert samples is None


class TestStripTraversalFromSteps:
    """测试 _strip_traversal_from_steps：遍历操作从导航步骤中剥离"""

    def _make_step(self, action="click", target="", description=""):
        from agent_scraper.core.models import NavigationStep
        return NavigationStep(action=action, target=target, description=description)

    def test_target_is_load_more_button(self):
        """target 本身是 load_more 文本 → 采用为 load_more_text"""
        steps = [self._make_step(target="Load more files", description="点击加载更多")]
        clean, hints, text, _ = TaskParser._strip_traversal_from_steps(steps, [])
        assert clean == []
        assert "load_more" in hints
        assert text == "Load more files"

    def test_target_is_nav_tab_not_button(self):
        """target 是导航标签名（如 "Files and versions"），description 含"加载更多"
        → 步骤被剥离，但 load_more_text 不应采用 target"""
        steps = [self._make_step(
            target="Files and versions",
            description="点击 Files and versions 标签查看文件列表并加载更多",
        )]
        clean, hints, text, _ = TaskParser._strip_traversal_from_steps(steps, [])
        assert clean == []
        assert "load_more" in hints
        assert text is None  # 不应把 "Files and versions" 当成按钮文本

    def test_target_chinese_load_more(self):
        """中文 target "加载更多" → 采用"""
        steps = [self._make_step(target="加载更多", description="点击加载按钮")]
        _, _, text, _ = TaskParser._strip_traversal_from_steps(steps, [])
        assert text == "加载更多"

    def test_show_more_target(self):
        """target "Show more" → 采用"""
        steps = [self._make_step(target="Show more", description="展开更多")]
        _, _, text, _ = TaskParser._strip_traversal_from_steps(steps, [])
        assert text == "Show more"

    def test_scroll_load_strips_step(self):
        """scroll + 滚动加载 → 步骤被剥离"""
        steps = [self._make_step(action="scroll", target="", description="下滑加载更多内容")]
        clean, hints, _, _ = TaskParser._strip_traversal_from_steps(steps, [])
        assert clean == []
        assert "load_more" in hints

    def test_next_page_stripped(self):
        """下一页步骤被剥离到 next_button hint，并提取按钮文本"""
        steps = [self._make_step(target="下一页", description="翻页")]
        clean, hints, _, next_text = TaskParser._strip_traversal_from_steps(steps, [])
        assert clean == []
        assert "next_button" in hints
        assert next_text == "下一页"

    def test_normal_step_kept(self):
        """普通导航步骤不应被剥离"""
        steps = [
            self._make_step(action="goto", target="https://example.com", description="打开页面"),
            self._make_step(action="click", target="Files and versions", description="切换标签"),
        ]
        clean, hints, text, _ = TaskParser._strip_traversal_from_steps(steps, [])
        assert len(clean) == 2
        assert hints == []
        assert text is None


class TestExtractQuotedText:
    """测试 _extract_quoted_text：从指令中提取引号文本，优先 load_more"""

    def test_single_load_more_quoted(self):
        instruction = '点击 "Load more files" 加载所有文件'
        assert TaskParser._extract_quoted_text(instruction) == "Load more files"

    def test_load_more_not_first_quoted(self):
        """load_more 文本不是第一个引号文本 → 仍优先返回它"""
        instruction = '进入 "Files and versions" 标签，点击 "Load more files"'
        assert TaskParser._extract_quoted_text(instruction) == "Load more files"

    def test_chinese_load_more_priority(self):
        instruction = '点击 "文件列表" 中的 "加载更多" 按钮'
        assert TaskParser._extract_quoted_text(instruction) == "加载更多"

    def test_no_load_more_returns_first(self):
        """没有 load_more 匹配 → 返回第一个引号文本"""
        instruction = '点击 "Files and versions" 标签'
        assert TaskParser._extract_quoted_text(instruction) == "Files and versions"

    def test_no_quotes(self):
        instruction = "打开页面提取数据"
        assert TaskParser._extract_quoted_text(instruction) is None

    def test_chinese_quotes(self):
        instruction = '点击\u201cLoad more\u201d按钮'
        assert TaskParser._extract_quoted_text(instruction) == "Load more"


class TestEnsureTraversalHints:
    def test_load_more_keywords(self):
        for kw in ["Load more", "加载更多", "全部加载", "加载全部"]:
            hints = TaskParser._ensure_traversal_hints([], f"请{kw}所有文件")
            assert "load_more" in hints

    def test_sub_pages_keywords(self):
        for kw in ["进入每个文件夹", "遍历子页面", "每个文件夹", "子文件夹"]:
            hints = TaskParser._ensure_traversal_hints([], f"请{kw}")
            assert "sub_pages" in hints

    def test_next_button_keywords(self):
        for kw in ["下一页", "next page", "翻页", "所有页", "每一页", "分页"]:
            hints = TaskParser._ensure_traversal_hints([], f"点击{kw}")
            assert "next_button" in hints

    def test_no_duplicate(self):
        """已有的 hint 不重复添加"""
        hints = TaskParser._ensure_traversal_hints(
            ["load_more"], "请加载更多所有文件"
        )
        assert hints.count("load_more") == 1

    def test_multiple_hints(self):
        hints = TaskParser._ensure_traversal_hints(
            [], "加载更多文件，然后进入每个文件夹"
        )
        assert "load_more" in hints
        assert "sub_pages" in hints

    def test_no_keywords(self):
        hints = TaskParser._ensure_traversal_hints([], "打开网站提取数据")
        assert hints == []


class TestParse:
    @pytest.mark.asyncio
    async def test_basic_parse(self, parser, mock_llm_service):
        mock_llm_service.call.return_value = json.dumps({
            "navigation_steps": [
                {"action": "goto", "target": "https://example.com", "description": "打开"}
            ],
            "extraction_goal": {
                "fields": {"title": "标题"},
                "output_format": "json",
                "url_pattern": None,
                "traversal_hints": [],
            }
        })

        task = await parser.parse("打开 example.com 提取标题")
        assert len(task.navigation_steps) == 1
        assert task.navigation_steps[0].action == "goto"
        assert "title" in task.extraction_goal.fields

    @pytest.mark.asyncio
    async def test_parse_with_code_block(self, parser, mock_llm_service):
        """LLM 返回带 ```json 包裹的内容"""
        mock_llm_service.call.return_value = "```json\n" + json.dumps({
            "navigation_steps": [],
            "extraction_goal": {
                "fields": {"name": "名称"},
                "traversal_hints": ["load_more"],
            }
        }) + "\n```"

        task = await parser.parse("加载更多")
        assert "load_more" in task.extraction_goal.traversal_hints

    @pytest.mark.asyncio
    async def test_samples_extracted_from_instruction(self, parser, mock_llm_service):
        """指令中包含 JSONL 样本，应被提取"""
        instruction = """\
打开 https://hf.co 提取文件
{"file_name": "a.txt", "url": "https://hf.co/a.txt"}
"""
        mock_llm_service.call.return_value = json.dumps({
            "navigation_steps": [
                {"action": "goto", "target": "https://hf.co", "description": "open"}
            ],
            "extraction_goal": {
                "fields": {"file_name": "文件名", "url": "链接"},
                "traversal_hints": [],
            }
        })

        task = await parser.parse(instruction)
        assert task.extraction_goal.samples is not None
        assert "file_name" in task.extraction_goal.samples
