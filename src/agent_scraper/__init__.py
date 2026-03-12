from agent_scraper.core.models import ScrapedResult, PageRules

__all__ = ["AgentScraper", "ScrapedResult", "PageRules"]


def __getattr__(name):
    if name == "AgentScraper":
        from agent_scraper.pipeline.orchestrator import AgentScraper
        return AgentScraper
    raise AttributeError(f"module 'agent_scraper' has no attribute {name}")
