"""输出格式化：将原始提取结果按 ExtractionGoal 格式化输出"""

import csv
import io
import json
import re
from urllib.parse import urljoin

from agent_scraper.models import ExtractionGoal, ScrapedResult


class Formatter:
    async def format(
        self,
        raw_data: dict[str, list],
        goal: ExtractionGoal,
        source_url: str = "",
    ) -> ScrapedResult:
        """将 AutoScraper 原始提取结果格式化为 ScrapedResult"""
        if not raw_data:
            return ScrapedResult(data=[], total_count=0, source_url=source_url)

        # 对齐各字段长度
        aligned = self._align_fields(raw_data)

        # 转为 list[dict] 格式
        field_names = list(aligned.keys())
        count = len(next(iter(aligned.values()))) if aligned else 0
        records = []
        for i in range(count):
            record = {name: aligned[name][i] for name in field_names}
            records.append(record)

        # 自动从样本推断缺失的 URL 字段（如 download_url = prefix + file_name）
        if goal.samples and records:
            records = self._fill_missing_url_fields(records, goal.samples)

        # 处理 URL 模式替换
        if goal.url_pattern:
            records = self._apply_url_pattern(records, goal.url_pattern, source_url)

        # 补全相对 URL
        if source_url:
            records = self._resolve_urls(records, source_url)

        return ScrapedResult(
            data=records,
            total_count=len(records),
            source_url=source_url,
        )

    @staticmethod
    def _fill_missing_url_fields(
        records: list[dict], samples: dict[str, list[str]]
    ) -> list[dict]:
        """从用户样本中推断缺失的 URL 字段构造规则。
        例: samples 有 file_name=[".gitattributes"] 和 download_url=["https://.../resolve/main/.gitattributes"]
        → 推断出 download_url = "https://.../resolve/main/" + file_name
        → 自动为每条记录构造 download_url
        """
        if not records:
            return records

        existing = set(records[0].keys())

        for field_name, sample_values in samples.items():
            if field_name in existing:
                continue  # 已提取到，不需要推断
            if not sample_values or not sample_values[0].startswith("http"):
                continue  # 不是 URL 字段

            # 尝试找到: sample_url = prefix + sample_of_other_field + suffix
            for other_field, other_samples in samples.items():
                if other_field == field_name or other_field not in existing:
                    continue

                pairs = list(zip(sample_values, other_samples))
                patterns = []
                for url_val, text_val in pairs:
                    if text_val in url_val:
                        idx = url_val.index(text_val)
                        prefix = url_val[:idx]
                        suffix = url_val[idx + len(text_val) :]
                        patterns.append((prefix, suffix))

                if len(patterns) == len(pairs) and patterns and all(
                    p == patterns[0] for p in patterns
                ):
                    prefix, suffix = patterns[0]
                    print(
                        f"[Formatter] 自动推断: {field_name} = '{prefix}' + {other_field} + '{suffix}'"
                    )
                    for record in records:
                        if other_field in record:
                            record[field_name] = prefix + record[other_field] + suffix
                    break

        return records

    @staticmethod
    def _align_fields(raw_data: dict[str, list]) -> dict[str, list]:
        """对齐各字段长度（截取到最短字段的长度）"""
        if not raw_data:
            return {}
        min_len = min(len(v) for v in raw_data.values())
        return {k: v[:min_len] for k, v in raw_data.items()}

    @staticmethod
    def _apply_url_pattern(
        records: list[dict], pattern: str, source_url: str
    ) -> list[dict]:
        """用 URL 模式模板构造完整 URL"""
        placeholders = re.findall(r"\{(\w+)\}", pattern)
        if not placeholders:
            return records

        for record in records:
            try:
                url = pattern.format(**record)
                record["url"] = url
            except KeyError:
                pass

        return records

    @staticmethod
    def _resolve_urls(records: list[dict], base_url: str) -> list[dict]:
        """将相对 URL 补全为绝对 URL"""
        url_fields = ["url", "href", "link", "download_url"]
        for record in records:
            for field in url_fields:
                if field in record and record[field]:
                    val = record[field]
                    if val.startswith("/") or (not val.startswith("http")):
                        record[field] = urljoin(base_url, val)
        return records

    @staticmethod
    def to_json(result: ScrapedResult, indent: int = 2) -> str:
        return json.dumps(result.data, ensure_ascii=False, indent=indent)

    @staticmethod
    def to_csv(result: ScrapedResult) -> str:
        if not result.data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=result.data[0].keys())
        writer.writeheader()
        writer.writerows(result.data)
        return output.getvalue()
