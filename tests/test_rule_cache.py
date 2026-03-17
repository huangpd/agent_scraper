"""测试 agent_scraper.pipeline.rule_cache — 规则学习缓存"""

import json

import pytest
from agent_scraper.core.models import PageRules
from agent_scraper.pipeline.rule_cache import CacheHit, RuleCache
from agent_scraper.rule_learner import AutoScraper


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / ".scraper_cache"


@pytest.fixture
def rule_cache(cache_dir):
    return RuleCache(cache_dir=cache_dir)


@pytest.fixture
def sample_scraper():
    """创建一个有 stack_list 的 AutoScraper 实例。"""
    scraper = AutoScraper()
    scraper.stack_list = [
        {
            "content": [("div", {"class": ["repo"]}), ("a", {})],
            "wanted_attr": "href",
            "hash": "abc123",
            "stack_id": "abc1",
            "alias": "url",
        },
        {
            "content": [("div", {"class": ["repo"]}), ("span", {})],
            "wanted_attr": None,
            "hash": "def456",
            "stack_id": "def4",
            "alias": "name",
        },
    ]
    return scraper


@pytest.fixture
def sample_fields():
    return {"url": "仓库链接", "name": "仓库名称", "star": "星标数"}


@pytest.fixture
def sample_data():
    return [
        {"url": "/repo/a", "name": "RepoA", "star": "100"},
        {"url": "/repo/b", "name": "RepoB", "star": "200"},
        {"url": "/repo/c", "name": "RepoC", "star": "300"},
    ]


class TestCacheKey:
    def test_basic_key(self, rule_cache):
        key = rule_cache.cache_key(
            "https://github.com/trending",
            {"star": "星标", "url": "链接", "name": "名称"},
        )
        assert key == "github.com|name,star,url"

    def test_same_domain_different_path(self, rule_cache):
        k1 = rule_cache.cache_key("https://github.com/trending", {"a": "x"})
        k2 = rule_cache.cache_key("https://github.com/explore", {"a": "x"})
        assert k1 == k2  # 同域名同字段 → 同一缓存

    def test_different_fields(self, rule_cache):
        k1 = rule_cache.cache_key("https://example.com", {"a": "x"})
        k2 = rule_cache.cache_key("https://example.com", {"a": "x", "b": "y"})
        assert k1 != k2

    def test_field_order_irrelevant(self, rule_cache):
        k1 = rule_cache.cache_key("https://example.com", {"b": "2", "a": "1"})
        k2 = rule_cache.cache_key("https://example.com", {"a": "1", "b": "2"})
        assert k1 == k2


class TestSaveAndLookup:
    def test_save_creates_file(self, rule_cache, cache_dir, sample_scraper, sample_fields, sample_data):
        path = rule_cache.save(
            url="https://github.com/trending",
            fields=sample_fields,
            scraper=sample_scraper,
            page_rules=None,
            sample_data=sample_data,
        )
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["source_url"] == "https://github.com/trending"
        assert sorted(data["fields"]) == ["name", "star", "url"]
        assert len(data["autoscraper_stacks"]) == 2
        assert len(data["sample_data"]) == 3

    def test_lookup_hit(self, rule_cache, sample_scraper, sample_fields, sample_data):
        rule_cache.save(
            url="https://github.com/trending",
            fields=sample_fields,
            scraper=sample_scraper,
            page_rules=PageRules(next_button_selector="a.next"),
            sample_data=sample_data,
        )
        hit = rule_cache.lookup("https://github.com/trending", sample_fields)
        assert hit is not None
        assert isinstance(hit, CacheHit)
        assert len(hit.scraper.stack_list) == 2
        assert hit.page_rules is not None
        assert hit.page_rules.next_button_selector == "a.next"
        assert "url" in hit.samples
        assert hit.samples["url"] == ["/repo/a", "/repo/b", "/repo/c"]

    def test_lookup_miss(self, rule_cache, sample_fields):
        hit = rule_cache.lookup("https://nonexist.com", sample_fields)
        assert hit is None

    def test_lookup_different_fields(self, rule_cache, sample_scraper, sample_fields, sample_data):
        rule_cache.save(
            url="https://github.com/trending",
            fields=sample_fields,
            scraper=sample_scraper,
            page_rules=None,
            sample_data=sample_data,
        )
        hit = rule_cache.lookup("https://github.com/trending", {"other_field": "desc"})
        assert hit is None

    def test_save_without_page_rules(self, rule_cache, sample_scraper, sample_fields, sample_data):
        rule_cache.save(
            url="https://example.com",
            fields=sample_fields,
            scraper=sample_scraper,
            page_rules=None,
            sample_data=sample_data,
        )
        hit = rule_cache.lookup("https://example.com", sample_fields)
        assert hit is not None
        assert hit.page_rules is None

    def test_overwrite_existing_cache(self, rule_cache, cache_dir, sample_scraper, sample_fields):
        rule_cache.save(
            url="https://example.com",
            fields=sample_fields,
            scraper=sample_scraper,
            page_rules=None,
            sample_data=[{"url": "old"}],
        )

        new_scraper = AutoScraper()
        new_scraper.stack_list = [{"content": [], "wanted_attr": None, "hash": "new", "stack_id": "new1", "alias": "x"}]
        rule_cache.save(
            url="https://example.com",
            fields=sample_fields,
            scraper=new_scraper,
            page_rules=None,
            sample_data=[{"url": "new"}],
        )

        hit = rule_cache.lookup("https://example.com", sample_fields)
        assert hit is not None
        assert len(hit.scraper.stack_list) == 1
        assert hit.scraper.stack_list[0]["hash"] == "new"


class TestInvalidate:
    def test_invalidate_existing(self, rule_cache, sample_scraper, sample_fields, sample_data):
        rule_cache.save(
            url="https://example.com",
            fields=sample_fields,
            scraper=sample_scraper,
            page_rules=None,
            sample_data=sample_data,
        )
        assert rule_cache.invalidate("https://example.com", sample_fields) is True
        assert rule_cache.lookup("https://example.com", sample_fields) is None

    def test_invalidate_nonexistent(self, rule_cache, sample_fields):
        assert rule_cache.invalidate("https://nonexist.com", sample_fields) is False


class TestCorruptCache:
    def test_corrupted_json(self, rule_cache, cache_dir, sample_fields):
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = rule_cache.cache_key("https://example.com", sample_fields)
        filepath = cache_dir / rule_cache._key_to_filename(key)
        filepath.write_text("not valid json", encoding="utf-8")
        hit = rule_cache.lookup("https://example.com", sample_fields)
        assert hit is None

    def test_empty_stack_list(self, rule_cache, cache_dir, sample_fields):
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = rule_cache.cache_key("https://example.com", sample_fields)
        filepath = cache_dir / rule_cache._key_to_filename(key)
        filepath.write_text(json.dumps({
            "autoscraper_stacks": [],
            "page_rules": None,
            "sample_data": [],
        }), encoding="utf-8")
        hit = rule_cache.lookup("https://example.com", sample_fields)
        assert hit is None


class TestSampleDataToWantedDict:
    def test_conversion(self):
        result = RuleCache._sample_data_to_wanted_dict([
            {"name": "a", "url": "/x"},
            {"name": "b", "url": "/y"},
        ])
        assert result == {"name": ["a", "b"], "url": ["/x", "/y"]}

    def test_empty_input(self):
        assert RuleCache._sample_data_to_wanted_dict([]) == {}
