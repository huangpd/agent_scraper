"""主编排器: Navigator → RuleDiscoverer → PageIterator → Extractor → Formatter"""

import os

from openai import AsyncOpenAI

from agent_scraper.extractor import Extractor
from agent_scraper.formatter import Formatter
from agent_scraper.models import ScrapedResult
from agent_scraper.navigator import Navigator
from agent_scraper.page_iterator import PageIterator
from agent_scraper.rule_discoverer import RuleDiscoverer
from agent_scraper.task_parser import TaskParser


class AgentScraper:
    def __init__(self, headless: bool = False):
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.task_parser = TaskParser(client=client)
        self.navigator = Navigator(headless=headless)
        self.rule_discoverer = RuleDiscoverer(client=client)
        self.extractor = Extractor(client=client)
        self.formatter = Formatter()

    async def run(self, instruction: str) -> ScrapedResult:
        # 1. 解析自然语言 → 结构化任务
        print("=" * 60)
        print("[AgentScraper] 步骤1: 解析指令...")
        task = await self.task_parser.parse(instruction)
        print(f"  导航步骤: {len(task.navigation_steps)} 步")
        print(f"  提取字段: {list(task.extraction_goal.fields.keys())}")
        if task.extraction_goal.samples:
            print(f"  用户样本: {len(list(task.extraction_goal.samples.values())[0])} 条")
        if task.extraction_goal.traversal_hints:
            print(f"  遍历模式: {task.extraction_goal.traversal_hints}")
        else:
            print(f"  遍历模式: 单页（无遍历）")

        # 推断 source_url
        source_url = ""
        for step in task.navigation_steps:
            if step.action == "goto":
                source_url = step.target
                break

        # 2. Agent 首次导航 → browser + page + HTML
        print("=" * 60)
        print("[AgentScraper] 步骤2: Agent 首次导航...")
        nav = await self.navigator.navigate(task.navigation_steps)

        try:
            # 3. AI 发现页面遍历规则（仅用户要求遍历时才调用 LLM）
            print("=" * 60)
            print("[AgentScraper] 步骤3: 分析页面规则...")
            rules = await self.rule_discoverer.discover(
                nav.html, source_url, task.extraction_goal.traversal_hints
            )
            # 打印发现的规则详情
            rules_info = []
            if rules.load_more_selector:
                rules_info.append(f"load_more='{rules.load_more_selector}'")
            if rules.sub_page_selector:
                rules_info.append(f"sub_page='{rules.sub_page_selector}' (attr={rules.sub_page_url_attr}, recursive={rules.sub_page_recursive})")
            if rules.next_button_selector:
                rules_info.append(f"next_button='{rules.next_button_selector}'")
            if rules.pagination_url:
                rules_info.append(f"pagination='{rules.pagination_url}' (max={rules.pagination_max})")
            print(f"  >>> 规则: {', '.join(rules_info) if rules_info else '无(单页模式)'}")

            # 4. 代码执行规则，遍历所有页面
            print("=" * 60)
            print("[AgentScraper] 步骤4: 代码执行遍历规则...")
            iterator = PageIterator(nav.browser)
            all_htmls = await iterator.iterate(nav.html, rules, source_url)

            # 5. 对每个 HTML 提取数据
            print("=" * 60)
            print(f"[AgentScraper] 步骤5: 提取数据 ({len(all_htmls)} 个页面)...")
            all_data: dict[str, list] = {}
            for i, html in enumerate(all_htmls):
                print(f"  提取页面 [{i + 1}/{len(all_htmls)}] ({len(html)/1024:.0f}KB)...")
                page_data = await self.extractor.extract(html, task.extraction_goal)
                for key, values in page_data.items():
                    all_data.setdefault(key, []).extend(values)
                print(f"    → 字段: { {k: len(v) for k, v in page_data.items()} }")

            # 汇总
            print(f"  >>> 总计: { {k: len(v) for k, v in all_data.items()} }")

            # 6. 格式化输出
            print("=" * 60)
            print("[AgentScraper] 步骤6: 格式化输出...")
            result = await self.formatter.format(all_data, task.extraction_goal, source_url)

            print("=" * 60)
            print(f"[AgentScraper] 完成! 提取 {result.total_count} 条数据")
            return result

        finally:
            # 确保浏览器关闭
            try:
                await nav.browser.stop()
            except Exception:
                pass
