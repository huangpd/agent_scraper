"""测试 OutputGuard: LLM 输出护栏的纯函数校验"""

import pytest
from agent_scraper.core.models import ExtractionGoal, NavigationStep, PageRules, ParsedTask
from agent_scraper.pipeline.output_guard import (
    validate_task,
    validate_css_selectors,
    validate_samples,
    validate_page_rules,
)


# ── validate_task ────────────────────────────────────────


class TestValidateTask:
    def test_valid_extract_task(self):
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=ExtractionGoal(fields={"name": "名称"}),
            raw_instruction="test",
        )
        assert validate_task(task) == []

    def test_missing_goto_step(self):
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="click", target="button", description="click")
            ],
            extraction_goal=ExtractionGoal(fields={"name": "名称"}),
            raw_instruction="test",
        )
        issues = validate_task(task)
        assert any("goto" in i for i in issues)

    def test_capture_mode_no_goto_ok(self):
        """capture 模式不需要 goto 步骤"""
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="click", target="button", description="click")
            ],
            extraction_goal=ExtractionGoal(fields={"url": "链接"}),
            raw_instruction="capture",
            mode="capture",
        )
        assert validate_task(task) == []

    def test_empty_fields(self):
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=ExtractionGoal(fields={}),
            raw_instruction="test",
        )
        issues = validate_task(task)
        assert any("fields" in i for i in issues)

    def test_invalid_mode(self):
        task = ParsedTask(
            navigation_steps=[
                NavigationStep(action="goto", target="https://example.com", description="open")
            ],
            extraction_goal=ExtractionGoal(fields={"name": "名称"}),
            raw_instruction="test",
            mode="invalid",
        )
        issues = validate_task(task)
        assert any("mode" in i for i in issues)


# ── validate_css_selectors ───────────────────────────────


_HTML = """
<html><body>
<div class="product-list">
  <div class="item"><a href="/a">Item A</a></div>
  <div class="item"><a href="/b">Item B</a></div>
  <div class="item"><a href="/c">Item C</a></div>
</div>
</body></html>
"""


class TestValidateCssSelectors:
    def test_valid_selectors_pass(self):
        selectors = {
            "name": {"selector": ".item a", "attr": "text"},
            "url": {"selector": ".item a", "attr": "href"},
        }
        result = validate_css_selectors(selectors, _HTML)
        assert "name" in result
        assert "url" in result

    def test_no_match_filtered(self):
        selectors = {
            "name": {"selector": ".nonexistent", "attr": "text"},
        }
        result = validate_css_selectors(selectors, _HTML)
        assert "name" not in result

    def test_deep_nesting_filtered(self):
        # 6 levels of > nesting exceeds _MAX_CSS_DEPTH=5
        deep = "html > body > div > div > div > div > a"
        selectors = {"name": {"selector": deep, "attr": "text"}}
        result = validate_css_selectors(selectors, _HTML)
        assert "name" not in result

    def test_empty_selector_filtered(self):
        selectors = {"name": {"selector": "", "attr": "text"}}
        result = validate_css_selectors(selectors, _HTML)
        assert "name" not in result

    def test_string_selector_format(self):
        """支持简单字符串格式的选择器"""
        selectors = {"name": ".item a"}
        result = validate_css_selectors(selectors, _HTML)
        assert "name" in result


# ── validate_samples ─────────────────────────────────────


_SAMPLE_HTML = """
<html><body>
<a href="/data/file1.csv">report_2024.csv</a>
<a href="/data/file2.csv">summary.csv</a>
<span>Some text here</span>
</body></html>
"""


class TestValidateSamples:
    def test_valid_samples_pass(self):
        samples = {
            "name": ["report_2024.csv", "summary.csv"],
            "url": ["/data/file1.csv"],
        }
        result = validate_samples(samples, _SAMPLE_HTML)
        assert result == samples

    def test_hallucinated_values_removed(self):
        samples = {
            "name": ["report_2024.csv", "nonexistent_file.txt"],
        }
        result = validate_samples(samples, _SAMPLE_HTML)
        assert result["name"] == ["report_2024.csv"]

    def test_all_hallucinated_field_removed(self):
        samples = {"name": ["fake1", "fake2"]}
        result = validate_samples(samples, _SAMPLE_HTML)
        assert "name" not in result

    def test_empty_input(self):
        assert validate_samples({}, _SAMPLE_HTML) == {}
        assert validate_samples({"name": ["a"]}, "") == {}

    def test_href_attribute_matched(self):
        """URL 样本通过 href 属性匹配"""
        samples = {"url": ["/data/file1.csv"]}
        result = validate_samples(samples, _SAMPLE_HTML)
        assert result["url"] == ["/data/file1.csv"]


# ── validate_page_rules ──────────────────────────────────


_RULES_HTML = """
<html><body>
<button class="load-more">Load More</button>
<a class="next-page" href="/page/2">Next</a>
<nav><a class="sub-link" href="/cat/1">Category 1</a></nav>
</body></html>
"""


class TestValidatePageRules:
    def test_valid_rules_pass(self):
        rules = PageRules(
            load_more_selector="button.load-more",
            next_button_selector="a.next-page",
            sub_page_selector="a.sub-link",
        )
        cleaned, failures = validate_page_rules(rules, _RULES_HTML)
        assert failures == []
        assert cleaned.load_more_selector == "button.load-more"
        assert cleaned.next_button_selector == "a.next-page"

    def test_invalid_selector_returns_failure_detail(self):
        rules = PageRules(
            load_more_selector=".nonexistent-button",
            next_button_selector="a.next-page",
        )
        cleaned, failures = validate_page_rules(rules, _RULES_HTML)
        assert len(failures) == 1
        assert failures[0]["field"] == "load_more_selector"
        assert failures[0]["selector"] == ".nonexistent-button"
        assert "无匹配" in failures[0]["reason"]
        assert cleaned.load_more_selector is None
        assert cleaned.next_button_selector == "a.next-page"

    def test_deep_nesting_returns_reason(self):
        rules = PageRules(
            next_button_selector="html > body > div > nav > ul > li > a",
        )
        cleaned, failures = validate_page_rules(rules, _RULES_HTML)
        assert len(failures) == 1
        assert failures[0]["field"] == "next_button_selector"
        assert "嵌套过深" in failures[0]["reason"]
        assert cleaned.next_button_selector is None

    def test_empty_rules_pass(self):
        rules = PageRules()
        cleaned, failures = validate_page_rules(rules, _RULES_HTML)
        assert failures == []

    def test_multiple_failures_all_reported(self):
        rules = PageRules(
            load_more_selector=".fake-load",
            next_button_selector=".fake-next",
            sub_page_selector=".fake-sub",
        )
        cleaned, failures = validate_page_rules(rules, _RULES_HTML)
        assert len(failures) == 3
        failed_fields = {f["field"] for f in failures}
        assert failed_fields == {"load_more_selector", "next_button_selector", "sub_page_selector"}
