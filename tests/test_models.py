"""测试 agent_scraper.models — Pydantic 数据模型"""

import pytest
from agent_scraper.core.models import (
    NavigationStep,
    ExtractionGoal,
    PageRules,
    ParsedTask,
    ScrapedResult,
)


class TestNavigationStep:
    def test_basic(self):
        step = NavigationStep(action="goto", target="https://example.com", description="打开首页")
        assert step.action == "goto"
        assert step.target == "https://example.com"

    def test_serialization(self):
        step = NavigationStep(action="click", target="Files tab", description="点击文件标签")
        d = step.model_dump()
        assert d == {"action": "click", "target": "Files tab", "value": "", "description": "点击文件标签"}

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            NavigationStep(target="url")  # missing action (required field)


class TestExtractionGoal:
    def test_defaults(self):
        goal = ExtractionGoal(fields={"name": "文件名"})
        assert goal.output_format == "json"
        assert goal.url_pattern is None
        assert goal.samples is None
        assert goal.traversal_hints == []

    def test_with_all_fields(self):
        goal = ExtractionGoal(
            fields={"name": "文件名", "url": "下载链接"},
            output_format="csv",
            url_pattern="https://hf.co/{name}",
            samples={"name": ["a.txt", "b.txt"]},
            traversal_hints=["load_more", "sub_pages"],
        )
        assert goal.output_format == "csv"
        assert len(goal.traversal_hints) == 2
        assert "a.txt" in goal.samples["name"]

    def test_empty_fields_allowed(self):
        goal = ExtractionGoal(fields={})
        assert goal.fields == {}


class TestPageRules:
    def test_all_none_by_default(self):
        rules = PageRules()
        assert rules.load_more_selector is None
        assert rules.next_button_selector is None
        assert rules.pagination_url is None
        assert rules.sub_page_selector is None
        assert rules.sub_page_url_attr == "href"
        assert rules.sub_page_recursive is False

    def test_partial(self):
        rules = PageRules(load_more_selector="button.load-more", sub_page_selector="a.folder")
        assert rules.load_more_selector == "button.load-more"
        assert rules.next_button_selector is None


class TestParsedTask:
    def test_construction(self):
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open"),
            ],
            extraction_goal=ExtractionGoal(fields={"title": "标题"}),
            raw_instruction="打开 example.com 提取标题",
        )
        assert len(task.navigation_steps) == 1
        assert task.extraction_goal.fields["title"] == "标题"
        assert "example.com" in task.raw_instruction


class TestScrapedResult:
    def test_empty(self):
        r = ScrapedResult(data=[], total_count=0, source_url="")
        assert r.data == []

    def test_with_data(self):
        r = ScrapedResult(
            data=[{"name": "a.txt", "url": "/a.txt"}],
            total_count=1,
            source_url="https://example.com",
        )
        assert r.total_count == 1
        assert r.data[0]["name"] == "a.txt"
