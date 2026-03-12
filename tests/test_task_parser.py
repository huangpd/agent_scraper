"""测试 agent_scraper.task_parser — 任务解析"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_scraper.pipeline.task_parser import TaskParser


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def parser(mock_client):
    return TaskParser(client=mock_client)


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


class TestEnsureTraversalHints:
    def test_load_more_keywords(self):
        for kw in ["Load more", "加载更多", "全部加载", "加载全部"]:
            hints = TaskParser._ensure_traversal_hints([], f"请{kw}所有文件")
            assert "load_more" in hints

    def test_sub_pages_keywords(self):
        for kw in ["进入每个文件夹", "遍历子页面", "每个文件夹", "子文件夹"]:
            hints = TaskParser._ensure_traversal_hints([], f"请{kw}")
            assert "sub_pages" in hints

    def test_pagination_keywords(self):
        for kw in ["翻页", "所有页", "每一页", "分页"]:
            hints = TaskParser._ensure_traversal_hints([], f"获取{kw}数据")
            assert "pagination" in hints

    def test_next_button_keywords(self):
        for kw in ["下一页", "next page"]:
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
    async def test_basic_parse(self, parser, mock_client):
        llm_response = json.dumps({
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
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = llm_response
        mock_client.chat.completions.create.return_value = mock_resp

        task = await parser.parse("打开 example.com 提取标题")
        assert len(task.navigation_steps) == 1
        assert task.navigation_steps[0].action == "goto"
        assert "title" in task.extraction_goal.fields

    @pytest.mark.asyncio
    async def test_parse_with_code_block(self, parser, mock_client):
        """LLM 返回带 ```json 包裹的内容"""
        llm_response = "```json\n" + json.dumps({
            "navigation_steps": [],
            "extraction_goal": {
                "fields": {"name": "名称"},
                "traversal_hints": ["load_more"],
            }
        }) + "\n```"
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = llm_response
        mock_client.chat.completions.create.return_value = mock_resp

        task = await parser.parse("加载更多")
        assert "load_more" in task.extraction_goal.traversal_hints

    @pytest.mark.asyncio
    async def test_samples_extracted_from_instruction(self, parser, mock_client):
        """指令中包含 JSONL 样本，应被提取"""
        instruction = """\
打开 https://hf.co 提取文件
{"file_name": "a.txt", "url": "https://hf.co/a.txt"}
"""
        llm_response = json.dumps({
            "navigation_steps": [
                {"action": "goto", "target": "https://hf.co", "description": "open"}
            ],
            "extraction_goal": {
                "fields": {"file_name": "文件名", "url": "链接"},
                "traversal_hints": [],
            }
        })
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = llm_response
        mock_client.chat.completions.create.return_value = mock_resp

        task = await parser.parse(instruction)
        assert task.extraction_goal.samples is not None
        assert "file_name" in task.extraction_goal.samples
