"""输出格式化：将原始提取结果按 ExtractionGoal 格式化输出"""

import csv
import io
import json
import logging
import re
from urllib.parse import urljoin

from agent_scraper.core.models import ExtractionGoal, ScrapedResult

logger = logging.getLogger(__name__)


class Formatter:
    async def format(
        self,
        raw_data: list[dict],
        goal: ExtractionGoal,
        source_url: str = "",
    ) -> ScrapedResult:
        """将行式 list[dict] 提取结果格式化为 ScrapedResult"""
        if not raw_data:
            return ScrapedResult(data=[], total_count=0, source_url=source_url)

        records = list(raw_data)

        # 用 _source_url 元数据填充空的 URL 类字段
        self._fill_url_from_source(records, goal.fields)

        # 清除内部元数据字段
        for rec in records:
            rec.pop("_source_url", None)

        # 跨页面去重（多个页面可能提取到相同记录）
        before = len(records)
        records = self._dedup_records(records)
        if len(records) < before:
            logger.info("去重: %d → %d 条", before, len(records))

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
    def _fill_url_from_source(records: list[dict], fields: dict[str, str]):
        """用 _source_url 元数据自动填充空的 URL 类字段。

        当记录来自详情页时，_source_url 就是该详情页的地址，
        可以直接填入名为 url/href/link 的空字段。
        """
        _URL_KEYWORDS = {"url", "href", "link", "链接", "地址"}
        url_fields = [
            f for f in fields
            if any(kw in f.lower() for kw in _URL_KEYWORDS)
        ]
        if not url_fields:
            return
        for rec in records:
            src = rec.get("_source_url", "")
            if not src:
                continue
            for f in url_fields:
                if not rec.get(f):
                    rec[f] = src

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
                    logger.info(
                        "自动推断: %s = '%s' + %s + '%s'", field_name, prefix, other_field, suffix
                    )
                    for record in records:
                        if other_field in record:
                            record[field_name] = prefix + record[other_field] + suffix
                    break

        return records

    @staticmethod
    def _dedup_records(records: list[dict]) -> list[dict]:
        """跨页面去重，保持顺序"""
        seen = set()
        unique = []
        for record in records:
            key = tuple(sorted(record.items()))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique

    @staticmethod
    def _align_fields(raw_data: dict[str, list]) -> dict[str, list]:
        """智能对齐各字段：优先按内容匹配配对，退化为索引截断。

        典型场景：file_name=["a.txt","b.csv"] download_url=[".../b.csv",".../a.txt"]
        按索引 zip 会错位，但按内容匹配（a.txt 出现在 .../a.txt 中）可以正确配对。
        """
        if not raw_data:
            return {}

        fields = list(raw_data.keys())
        if len(fields) < 2:
            return raw_data

        # 尝试找到一对可以通过「文本包含」关系配对的字段
        # 常见：file_name 的值是 download_url 值的子串
        for i, f1 in enumerate(fields):
            for f2 in fields[i + 1:]:
                vals1 = raw_data[f1]
                vals2 = raw_data[f2]
                # 尝试 f1 值 ⊂ f2 值（或反向）
                matched = _match_by_containment(vals1, vals2)
                if matched is not None:
                    aligned1, aligned2 = matched
                    result = dict(raw_data)
                    result[f1] = aligned1
                    result[f2] = aligned2
                    # 其他字段截断到配对长度
                    n = len(aligned1)
                    for f in fields:
                        if f not in (f1, f2):
                            result[f] = raw_data[f][:n]
                    logger.info("智能对齐: %s ↔ %s (%d 对)", f1, f2, n)
                    return result

        # 退化：按最短字段截断
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
        """将相对 URL 补全为绝对 URL（动态检测 URL 字段，不依赖硬编码列表）"""
        _URL_KEYWORDS = {"url", "href", "link", "src", "链接", "地址"}
        for record in records:
            for field, val in record.items():
                if not val or not isinstance(val, str):
                    continue
                # 已经是绝对 URL 则跳过
                if val.startswith("http://") or val.startswith("https://"):
                    continue
                # 判断是否需要补全：字段名含 URL 关键词，或值像相对路径
                field_lower = field.lower()
                is_url_field = any(kw in field_lower for kw in _URL_KEYWORDS)
                is_relative_path = val.startswith("/")
                if is_url_field or is_relative_path:
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


def _match_by_containment(
    vals1: list[str], vals2: list[str],
) -> tuple[list[str], list[str]] | None:
    """尝试通过「文本包含」关系配对两个字段列表。

    如果 vals1 中大部分值都能在某个 vals2 值中找到（或反向），
    返回配对后的 (aligned_vals1, aligned_vals2)。

    例: vals1=["a.txt","b.csv"], vals2=[".../b.csv",".../a.txt"]
    → (["a.txt","b.csv"], [".../a.txt",".../b.csv"])
    """
    # 尝试两个方向：vals1 ⊂ vals2 和 vals2 ⊂ vals1
    for short, long, reversed_order in [(vals1, vals2, False), (vals2, vals1, True)]:
        pairs = []
        used = set()
        for sv in short:
            sv_s = str(sv).strip()
            if not sv_s:
                continue
            for j, lv in enumerate(long):
                if j in used:
                    continue
                if sv_s in str(lv):
                    pairs.append((sv, lv))
                    used.add(j)
                    break

        # 至少 60% 的短列表能配对上才算成功
        if len(pairs) >= max(2, len(short) * 0.6):
            if reversed_order:
                aligned_long = [p[0] for p in pairs]
                aligned_short = [p[1] for p in pairs]
            else:
                aligned_short = [p[0] for p in pairs]
                aligned_long = [p[1] for p in pairs]
            return (aligned_short, aligned_long)

    return None
