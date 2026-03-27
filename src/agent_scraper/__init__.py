import logging
import os
from pathlib import Path

from agent_scraper.core.models import ScrapedResult, PageRules

__all__ = ["AgentScraper", "ScrapedResult", "PageRules"]

# ── 日志配置：agent_scraper.* 写入本地文件 ──────────────────
_log_dir = Path(os.getenv("AGENT_SCRAPER_LOG_DIR", "logs"))
_log_dir.mkdir(exist_ok=True)

_pkg_logger = logging.getLogger("agent_scraper")
_pkg_logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler(_log_dir / "agent_scraper.log", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s\n%(message)s\n"))

_pkg_logger.addHandler(_fh)


def __getattr__(name):
    if name == "AgentScraper":
        from agent_scraper.pipeline.orchestrator import AgentScraper
        return AgentScraper
    raise AttributeError(f"module 'agent_scraper' has no attribute {name}")
