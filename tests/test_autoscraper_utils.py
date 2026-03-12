"""测试 agent_scraper.autoscraper.utils — 纯工具函数"""

import pytest
from autoscraper.utils import (
    unique_stack_list,
    unique_hashable,
    normalize,
    text_match,
    get_non_rec_text,
    ResultItem,
    FuzzyText,
)


class TestUniqueStackList:
    def test_dedup(self):
        stacks = [
            {"hash": "abc", "data": 1},
            {"hash": "def", "data": 2},
            {"hash": "abc", "data": 3},
        ]
        result = unique_stack_list(stacks)
        assert len(result) == 2
        assert result[0]["data"] == 1
        assert result[1]["data"] == 2

    def test_empty(self):
        assert unique_stack_list([]) == []

    def test_all_unique(self):
        stacks = [{"hash": str(i), "data": i} for i in range(5)]
        assert len(unique_stack_list(stacks)) == 5


class TestUniqueHashable:
    def test_dedup_preserves_order(self):
        assert unique_hashable([3, 1, 2, 1, 3]) == [3, 1, 2]

    def test_empty(self):
        assert unique_hashable([]) == []

    def test_strings(self):
        assert unique_hashable(["a", "b", "a", "c"]) == ["a", "b", "c"]


class TestNormalize:
    def test_strips_whitespace(self):
        assert normalize("  hello  ") == "hello"

    def test_nfkd(self):
        # ﬁ (fi ligature) -> fi
        assert normalize("\ufb01le") == "file"

    def test_non_string(self):
        assert normalize(123) == 123


class TestTextMatch:
    def test_exact_match(self):
        assert text_match("hello", "hello", 1.0)
        assert not text_match("hello", "world", 1.0)

    def test_fuzzy_match(self):
        assert text_match("hello", "helo", 0.7)
        assert not text_match("hello", "xyz", 0.7)

    def test_regex_match(self):
        import re
        pattern = re.compile(r"hello\d+")
        assert text_match(pattern, "hello123", 1.0)
        assert not text_match(pattern, "hello", 1.0)


class TestResultItem:
    def test_str(self):
        item = ResultItem("hello", 0)
        assert str(item) == "hello"
        assert item.text == "hello"
        assert item.index == 0


class TestFuzzyText:
    def test_search_match(self):
        ft = FuzzyText("hello world", 0.8)
        assert ft.search("hello world")

    def test_search_no_match(self):
        ft = FuzzyText("hello", 0.9)
        assert not ft.search("xyz")

    def test_search_partial(self):
        ft = FuzzyText("config.json", 0.7)
        assert ft.search("config.json")
        assert not ft.search("totally_different")


class TestGetNonRecText:
    def test_basic(self):
        from bs4 import BeautifulSoup
        html = "<div>outer<span>inner</span>text</div>"
        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div")
        # 只取直接文本，不含子元素文本
        result = get_non_rec_text(div)
        assert "inner" not in result
        assert "outer" in result
