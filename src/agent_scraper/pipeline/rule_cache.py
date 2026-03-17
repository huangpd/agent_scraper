"""规则学习缓存：自由模式提取成功后自动沉淀 XPath 规则到磁盘。

首次无样本提取成功 → 训练 AutoScraper → 保存缓存
下次同域名同字段 → 加载缓存 → 直接走 AutoScraper 快速路径（秒级，零 LLM token）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from agent_scraper.core.models import PageRules
from agent_scraper.rule_learner import AutoScraper

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".scraper_cache"


@dataclass
class CacheHit:
    """缓存命中结果。"""
    scraper: AutoScraper
    page_rules: PageRules | None
    samples: dict[str, list[str]]


class RuleCache:
    """管理 AutoScraper XPath 规则的磁盘缓存。"""

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            self.cache_dir = Path(CACHE_DIR_NAME)
        else:
            self.cache_dir = Path(cache_dir)

    @staticmethod
    def cache_key(url: str, fields: dict[str, str]) -> str:
        """生成缓存键: 'domain|sorted_field_names'"""
        domain = urlparse(url).netloc or urlparse(url).hostname or "unknown"
        sorted_names = ",".join(sorted(fields.keys()))
        return f"{domain}|{sorted_names}"

    @staticmethod
    def _key_to_filename(key: str) -> str:
        """缓存键 → 安全文件名"""
        return key.replace("|", "__").replace(",", "_").replace(":", "_") + ".json"

    def save(
        self,
        url: str,
        fields: dict[str, str],
        scraper: AutoScraper,
        page_rules: PageRules | None,
        sample_data: list[dict],
    ) -> Path:
        """保存缓存到磁盘，返回文件路径。"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        key = self.cache_key(url, fields)
        filepath = self.cache_dir / self._key_to_filename(key)

        # 取前 3 条作为调试样本
        preview = sample_data[:3] if sample_data else []

        payload = {
            "cache_key": key,
            "source_url": url,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fields": sorted(fields.keys()),
            "autoscraper_stacks": scraper.stack_list,
            "xpath_rules": scraper.get_result_xpath_rule(),
            "page_rules": page_rules.model_dump() if page_rules else None,
            "sample_data": preview,
        }

        filepath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[RuleCache] 已保存缓存: %s → %s", key, filepath)
        return filepath

    def lookup(self, url: str, fields: dict[str, str]) -> CacheHit | None:
        """查找缓存，命中返回 CacheHit，未命中返回 None。"""
        key = self.cache_key(url, fields)
        filepath = self.cache_dir / self._key_to_filename(key)

        if not filepath.exists():
            return None

        try:
            payload = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[RuleCache] 缓存文件损坏: %s — %s", filepath, e)
            return None

        # 恢复 AutoScraper
        scraper = AutoScraper()
        scraper.stack_list = payload.get("autoscraper_stacks", [])
        if not scraper.stack_list:
            logger.warning("[RuleCache] 缓存中无 stack_list: %s", filepath)
            return None

        # 恢复 PageRules
        page_rules_data = payload.get("page_rules")
        page_rules = PageRules(**page_rules_data) if page_rules_data else None

        # 从 sample_data 构造 wanted_dict 格式
        samples = self._sample_data_to_wanted_dict(payload.get("sample_data", []))

        logger.info("[RuleCache] 缓存命中: %s", key)
        return CacheHit(scraper=scraper, page_rules=page_rules, samples=samples)

    @staticmethod
    def _sample_data_to_wanted_dict(sample_data: list[dict]) -> dict[str, list[str]]:
        """将 list[dict] 行式数据转为 wanted_dict 列式格式。

        例: [{"name": "a", "url": "/x"}, {"name": "b", "url": "/y"}]
          → {"name": ["a", "b"], "url": ["/x", "/y"]}
        """
        if not sample_data:
            return {}
        result: dict[str, list[str]] = {}
        for row in sample_data:
            for k, v in row.items():
                if k not in result:
                    result[k] = []
                result[k].append(str(v))
        return result

    def invalidate(self, url: str, fields: dict[str, str]) -> bool:
        """删除指定缓存，返回是否成功删除。"""
        key = self.cache_key(url, fields)
        filepath = self.cache_dir / self._key_to_filename(key)
        if filepath.exists():
            filepath.unlink()
            logger.info("[RuleCache] 已删除缓存: %s", key)
            return True
        return False
