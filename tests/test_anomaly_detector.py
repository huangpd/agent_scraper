import importlib

import pytest

from agent_scraper.extraction import anomaly_detector as ad


def _sample_urls():
    base = "https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix"
    files = [
        f"{base}/blob/main/part_{i}.tokenized.jsonl" for i in range(5)
    ]
    return [base] + files


def test_knn_fallback_without_sklearn(monkeypatch):
    monkeypatch.setattr(ad, "_HAS_SKLEARN", False)
    urls = _sample_urls()
    res = ad.detect_url_anomalies(urls, top_k=3)
    assert res, "should return anomalies even without sklearn"
    assert res[0]["url"] == urls[0]
    assert "knn" in res[0]["detectors"]


@pytest.mark.skipif(
    not importlib.util.find_spec("sklearn"),
    reason="sklearn not installed",
)
def test_intersection_includes_root_with_sklearn():
    urls = _sample_urls()
    res = ad.detect_url_anomalies(urls, top_k=3)
    assert res, "anomaly list should not be empty with sklearn available"
    # all returned items should carry knn plus at least one sklearn detector
    for item in res:
        detectors = set(item["detectors"])
        assert "knn" in detectors
        assert detectors & {"iforest", "lof"}, "should include a sklearn-based detector"
