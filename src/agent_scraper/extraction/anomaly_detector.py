"""Lightweight anomaly detection for URL and file-name lists.

Targets three practical scenarios:
- URLs from a different domain mixed into a batch
- Non-target pages (login, error, ad redirects)
- Abnormal file formats mixed into a file list

Uses batch-relative frequency features + IsolationForest & LOF ensemble.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# URL 中常见的非目标页面关键词
# 匹配路径段级别的可疑关键词（用 \b 词边界，避免匹配数字 ID 中的子串）
_SUSPICIOUS_RE = re.compile(
    r"(?:^|/)("
    r"login|signin|sign[_-]?in|signup|sign[_-]?up|logout|auth|oauth"
    r"|register|captcha|verify|confirm"
    r"|error|not[_-]?found"
    r"|redirect|jump|click|track"
    r"|advert|banner|popup|promo"
    r")(?=[/?#.]|$)",
    re.IGNORECASE,
)
# 状态码只在整个路径段匹配时才算（如 /404.html，不匹配 /496_1563403）
_STATUS_CODE_RE = re.compile(r"(?:^|/)([345]\d{2})(?:\.html?|/|$)")

_FEAT_NAMES = [
    "域名/目录占比",
    "扩展名占比",
    "路径前缀占比",
    "可疑关键词",
    "查询参数数量",
    "长度偏离中位数",
    "路径深度偏离中位数",
]


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "ftp://"))


def _parse_entry(entry: str) -> dict:
    """将 URL 或文件路径解析为统一的结构化字段。"""
    if _is_url(entry):
        p = urlparse(entry)
        parts = [x for x in p.path.split("/") if x]
        return {
            "domain": p.netloc,
            "ext": os.path.splitext(p.path)[1].lower(),
            "prefix": parts[0] if parts else "",
            "depth": len(parts),
            "query_count": len(parse_qs(p.query)),
            "path": p.path,
            "length": len(entry),
        }
    else:
        norm = entry.replace("\\", "/")
        parts = [x for x in norm.split("/") if x]
        return {
            "domain": parts[0] if parts else "",
            "ext": os.path.splitext(entry)[1].lower(),
            "prefix": parts[1] if len(parts) > 1 else "",
            "depth": len(parts),
            "query_count": 0,
            "path": norm,
            "length": len(entry),
        }


def _build_features(entries: Sequence[str]) -> tuple[np.ndarray, list[dict]]:
    """构建基于批次频率的特征矩阵。"""
    parsed = [_parse_entry(e) for e in entries]
    n = len(parsed)

    # 统计各维度的批次频率
    domain_counts = Counter(p["domain"] for p in parsed)
    ext_counts = Counter(p["ext"] for p in parsed)
    prefix_counts = Counter(p["prefix"] for p in parsed)

    lengths = np.array([p["length"] for p in parsed], float)
    depths = np.array([p["depth"] for p in parsed], float)
    median_len = float(np.median(lengths))
    median_depth = float(np.median(depths))

    features = []
    for p in parsed:
        domain_freq = domain_counts[p["domain"]] / n
        ext_freq = ext_counts[p["ext"]] / n
        prefix_freq = prefix_counts[p["prefix"]] / n
        has_suspicious = float(
            bool(_SUSPICIOUS_RE.search(p["path"]))
            or bool(_STATUS_CODE_RE.search(p["path"]))
        )
        query_count = float(p["query_count"])
        # 取对数比值，避免绝对值差异被淹没
        len_ratio = abs(p["length"] - median_len) / (median_len + 1e-8)
        depth_ratio = abs(p["depth"] - median_depth) / (median_depth + 1e-8)

        features.append([
            domain_freq,
            ext_freq,
            prefix_freq,
            has_suspicious,
            query_count,
            len_ratio,
            depth_ratio,
        ])

    return np.array(features, float), parsed


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def _explain(feats: np.ndarray, idx: int, parsed: list[dict]) -> str:
    """生成可读的异常原因。"""
    f = feats[idx]
    reasons = []

    if f[0] < 0.05:
        reasons.append(f"域名 '{parsed[idx]['domain']}' 在批次中极少出现")
    elif f[0] < 0.1:
        reasons.append(f"域名 '{parsed[idx]['domain']}' 在批次中较少出现")

    if f[1] < 0.05:
        reasons.append(f"扩展名 '{parsed[idx]['ext'] or '(无)'}' 在批次中极少出现")
    elif f[1] < 0.1:
        reasons.append(f"扩展名 '{parsed[idx]['ext'] or '(无)'}' 在批次中较少出现")

    if f[2] < 0.05:
        reasons.append(f"路径前缀 '/{parsed[idx]['prefix']}/' 在批次中极少出现")

    if f[3] > 0:
        match = _SUSPICIOUS_RE.search(parsed[idx]["path"])
        status_match = _STATUS_CODE_RE.search(parsed[idx]["path"])
        kw = (match.group(1) if match else None) or (
            status_match.group(1) if status_match else "?"
        )
        reasons.append(f"路径含可疑关键词 '{kw}'")

    if f[4] > 3:
        reasons.append(f"查询参数数量异常 ({int(f[4])} 个)")

    if f[5] > 1.0:
        reasons.append(f"长度偏离中位数 {f[5]:.0%}")

    if f[6] > 1.0:
        reasons.append(f"路径深度偏离中位数 {f[6]:.0%}")

    return "; ".join(reasons)


def _classify(feats: np.ndarray, idx: int, parsed: dict) -> str:
    """将异常归入场景分类。"""
    f = feats[idx]
    if f[0] < 0.05:
        return "foreign_domain"
    if f[3] > 0:
        return "suspicious_page"
    if f[1] < 0.05:
        return "abnormal_format"
    return "structural_outlier"


_CATEGORY_LABELS = {
    "foreign_domain": "外部域名混入",
    "suspicious_page": "疑似非目标页面",
    "abnormal_format": "异常文件格式",
    "structural_outlier": "结构偏离",
}


def detect_anomalies(entries: Sequence[str], top_k: int = 5) -> dict:
    """检测批次中的异常条目。

    适用场景:
    - 混入了不同域名的 URL
    - 爬到了登录页/错误页/广告跳转等非目标页面
    - 文件列表中混入了异常格式的文件

    Args:
        entries: URL 字符串或文件路径列表。
        top_k: 每个检测器取 top-k 异常进行交集。

    Returns:
        {
            summary: 面向用户的结论文本,
            total: 总条目数,
            anomaly_count: 异常数,
            categories: {category: {label, count, items}},
            items: [{index, entry, category, category_label, reason}],
        }
    """
    empty = {
        "summary": f"共 {len(entries)} 条数据，未发现异常。",
        "total": len(entries),
        "anomaly_count": 0,
        "categories": {},
        "items": [],
    }

    if len(entries) <= 3:
        return empty

    X_raw, parsed = _build_features(entries)
    X = _standardize(X_raw)
    top_k = max(1, min(top_k, len(entries)))

    detector_sets: dict[str, set[int]] = {}

    try:
        iso = IsolationForest(
            n_estimators=200, contamination="auto", random_state=0
        ).fit(X)
        iso_scores = -iso.score_samples(X)
        detector_sets["iforest"] = set(np.argsort(iso_scores)[::-1][:top_k])
    except Exception:
        pass

    try:
        neigh = max(5, min(30, len(entries) - 1))
        lof = LocalOutlierFactor(n_neighbors=neigh, contamination="auto")
        lof.fit(X)
        lof_scores = -lof.negative_outlier_factor_
        detector_sets["lof"] = set(np.argsort(lof_scores)[::-1][:top_k])
    except Exception:
        pass

    available = list(detector_sets.values())
    if not available:
        return empty
    intersect = set.intersection(*available)
    if not intersect:
        intersect = set.union(*available)

    # 构建分类结果（无具体原因的跳过——解释不了就不算真异常）
    items = []
    cat_groups: dict[str, list[dict]] = {}
    for idx in sorted(intersect):
        reason = _explain(X_raw, idx, parsed)
        if not reason:
            continue
        cat = _classify(X_raw, idx, parsed[idx])
        item = {
            "index": int(idx),
            "entry": entries[idx],
            "category": cat,
            "category_label": _CATEGORY_LABELS[cat],
            "reason": reason,
        }
        items.append(item)
        cat_groups.setdefault(cat, []).append(item)

    if not items:
        return empty

    # 构建分类摘要
    categories = {}
    for cat, group_items in cat_groups.items():
        categories[cat] = {
            "label": _CATEGORY_LABELS[cat],
            "count": len(group_items),
            "items": group_items,
        }

    # 生成面向用户的总结文本
    parts = []
    for cat in ["foreign_domain", "suspicious_page", "abnormal_format", "structural_outlier"]:
        if cat in categories:
            c = categories[cat]
            parts.append(f"{c['count']} 条{c['label']}")
    summary_detail = "、".join(parts)
    summary = f"共 {len(entries)} 条数据，发现 {len(items)} 条可疑：{summary_detail}。"

    return {
        "summary": summary,
        "total": len(entries),
        "anomaly_count": len(items),
        "categories": categories,
        "items": items,
    }


# backward compat
detect_url_anomalies = detect_anomalies
