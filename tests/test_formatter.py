"""测试 agent_scraper.formatter — 输出格式化"""

import pytest
from agent_scraper.extraction.formatter import Formatter
from agent_scraper.core.models import ExtractionGoal, ScrapedResult


@pytest.fixture
def formatter():
    return Formatter()


class TestAlignFields:
    def test_align_to_shortest(self):
        raw = {"name": ["a", "b", "c"], "size": ["1", "2"]}
        result = Formatter._align_fields(raw)
        assert len(result["name"]) == 2
        assert len(result["size"]) == 2

    def test_equal_lengths(self):
        raw = {"name": ["a", "b"], "size": ["1", "2"]}
        result = Formatter._align_fields(raw)
        assert len(result["name"]) == 2

    def test_empty(self):
        assert Formatter._align_fields({}) == {}


class TestResolveUrls:
    def test_relative_to_absolute(self):
        records = [{"url": "/repo/file.txt", "name": "file.txt"}]
        result = Formatter._resolve_urls(records, "https://example.com/page")
        assert result[0]["url"] == "https://example.com/repo/file.txt"

    def test_absolute_unchanged(self):
        records = [{"url": "https://cdn.example.com/file.txt"}]
        result = Formatter._resolve_urls(records, "https://example.com")
        assert result[0]["url"] == "https://cdn.example.com/file.txt"

    def test_multiple_url_fields(self):
        records = [{"url": "/a", "href": "/b", "link": "/c", "download_url": "/d"}]
        result = Formatter._resolve_urls(records, "https://example.com")
        for field in ["url", "href", "link", "download_url"]:
            assert result[0][field].startswith("https://")

    def test_non_url_fields_untouched(self):
        records = [{"name": "file.txt", "size": "10KB"}]
        result = Formatter._resolve_urls(records, "https://example.com")
        assert result[0]["name"] == "file.txt"


class TestApplyUrlPattern:
    def test_basic_pattern(self):
        records = [{"name": "file.txt"}, {"name": "data.csv"}]
        result = Formatter._apply_url_pattern(records, "https://cdn.com/{name}", "")
        assert result[0]["url"] == "https://cdn.com/file.txt"
        assert result[1]["url"] == "https://cdn.com/data.csv"

    def test_no_placeholders(self):
        records = [{"name": "file.txt"}]
        result = Formatter._apply_url_pattern(records, "https://static.com/all", "")
        assert "url" not in result[0]

    def test_missing_field_skipped(self):
        records = [{"name": "file.txt"}]
        result = Formatter._apply_url_pattern(records, "https://cdn.com/{missing}", "")
        assert "url" not in result[0]


class TestFillMissingUrlFields:
    def test_infer_url_from_samples(self):
        records = [
            {"file_name": ".gitattributes"},
            {"file_name": "config.json"},
        ]
        samples = {
            "file_name": [".gitattributes", "config.json"],
            "download_url": [
                "https://hf.co/resolve/main/.gitattributes",
                "https://hf.co/resolve/main/config.json",
            ],
        }
        result = Formatter._fill_missing_url_fields(records, samples)
        assert result[0]["download_url"] == "https://hf.co/resolve/main/.gitattributes"
        assert result[1]["download_url"] == "https://hf.co/resolve/main/config.json"

    def test_no_inference_when_already_exists(self):
        records = [{"file_name": "a.txt", "download_url": "https://existing.com/a.txt"}]
        samples = {
            "file_name": ["a.txt"],
            "download_url": ["https://other.com/a.txt"],
        }
        result = Formatter._fill_missing_url_fields(records, samples)
        assert result[0]["download_url"] == "https://existing.com/a.txt"

    def test_no_inference_for_non_url(self):
        records = [{"file_name": "a.txt"}]
        samples = {
            "file_name": ["a.txt"],
            "size": ["10KB"],  # not URL
        }
        result = Formatter._fill_missing_url_fields(records, samples)
        assert "size" not in result[0]

    def test_empty_records(self):
        assert Formatter._fill_missing_url_fields([], {"a": ["b"]}) == []


class TestDedupRecords:
    def test_removes_duplicates(self):
        records = [
            {"name": "a.txt", "url": "/a"},
            {"name": "b.txt", "url": "/b"},
            {"name": "a.txt", "url": "/a"},  # dup
            {"name": "b.txt", "url": "/b"},  # dup
        ]
        result = Formatter._dedup_records(records)
        assert len(result) == 2

    def test_preserves_order(self):
        records = [
            {"name": "b.txt"},
            {"name": "a.txt"},
            {"name": "b.txt"},
        ]
        result = Formatter._dedup_records(records)
        assert result[0]["name"] == "b.txt"
        assert result[1]["name"] == "a.txt"

    def test_no_duplicates(self):
        records = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        assert len(Formatter._dedup_records(records)) == 3

    def test_empty(self):
        assert Formatter._dedup_records([]) == []


class TestFormat:
    @pytest.mark.asyncio
    async def test_basic_format(self, formatter):
        raw = {"name": ["a.txt", "b.txt"], "size": ["1KB", "2KB"]}
        goal = ExtractionGoal(fields={"name": "文件名", "size": "大小"})
        result = await formatter.format(raw, goal, "https://example.com")
        assert isinstance(result, ScrapedResult)
        assert result.total_count == 2
        assert result.data[0]["name"] == "a.txt"

    @pytest.mark.asyncio
    async def test_empty_data(self, formatter):
        goal = ExtractionGoal(fields={"name": "文件名"})
        result = await formatter.format({}, goal, "")
        assert result.total_count == 0
        assert result.data == []

    @pytest.mark.asyncio
    async def test_url_resolved(self, formatter):
        raw = {"name": ["a.txt"], "url": ["/repo/a.txt"]}
        goal = ExtractionGoal(fields={"name": "名称", "url": "链接"})
        result = await formatter.format(raw, goal, "https://example.com/page")
        assert result.data[0]["url"].startswith("https://")


class TestOutputFormats:
    def test_to_json(self):
        result = ScrapedResult(
            data=[{"name": "a.txt"}], total_count=1, source_url=""
        )
        json_str = Formatter.to_json(result)
        assert '"name"' in json_str
        assert "a.txt" in json_str

    def test_to_csv(self):
        result = ScrapedResult(
            data=[{"name": "a.txt", "size": "1KB"}], total_count=1, source_url=""
        )
        csv_str = Formatter.to_csv(result)
        assert "name" in csv_str
        assert "a.txt" in csv_str

    def test_to_csv_empty(self):
        result = ScrapedResult(data=[], total_count=0, source_url="")
        assert Formatter.to_csv(result) == ""
