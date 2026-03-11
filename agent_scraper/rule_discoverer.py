"""RuleDiscoverer: LLM 分析页面结构，**只发现用户要求的遍历模式**。
用户没要求遍历 → 不调用 LLM，直接返回空规则（单页提取）。
"""

import json
import os
import re

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from agent_scraper.models import PageRules

DISCOVER_PROMPT = """\
你是一个网页结构分析专家。分析下面的 HTML 片段，**只**找出用户要求的遍历规则。

页面当前URL: {current_url}
用户要求的遍历模式: {requested_modes}

HTML 片段:
```html
{html_snippet}
```

根据用户要求的模式，输出对应的 CSS 选择器或 URL 模式（严格JSON，不要多余文字）：

{{
  "load_more_selector": "（仅当用户要求 load_more 时）'加载更多'按钮的CSS选择器，否则null",
  "next_button_selector": "（仅当用户要求 next_button 时）'下一页'按钮的CSS选择器，否则null",
  "pagination_url": "（仅当用户要求 pagination 时）URL模板用{{n}}表示页码，否则null",
  "pagination_max": "（仅当用户要求 pagination 时）总页数，否则null",
  "sub_page_selector": "（仅当用户要求 sub_pages 时）子页面/文件夹链接的CSS选择器，否则null",
  "sub_page_url_attr": "子页面链接的URL属性，通常是href",
  "sub_page_recursive": false
}}

CSS选择器要求:
1. 尽量精确，能唯一定位到目标元素
2. 用户没要求的模式，对应字段必须返回null
3. 对于 load_more，优先用精确选择器；如果按钮没有 class/id，可以用文本匹配描述

只输出JSON。
"""

MAX_HTML_SIZE = 50 * 1024


class RuleDiscoverer:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("MODEL_NAME", "gpt-4o")

    async def discover(
        self, html: str, current_url: str = "", traversal_hints: list[str] | None = None
    ) -> PageRules:
        """
        根据用户指定的 traversal_hints 分析页面。
        如果 hints 为空 → 直接返回空规则（单页模式，不调用 LLM）。
        """
        if not traversal_hints:
            print("[RuleDiscoverer] 用户未要求遍历，单页模式")
            return PageRules()

        print(f"[RuleDiscoverer] 用户要求: {traversal_hints}")
        snippet = self._get_clean_snippet(html)

        prompt = DISCOVER_PROMPT.format(
            current_url=current_url,
            requested_modes=", ".join(traversal_hints),
            html_snippet=snippet,
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content.strip()

            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # 只保留用户要求的模式
            filtered = {}
            if "load_more" in traversal_hints:
                filtered["load_more_selector"] = data.get("load_more_selector")
            if "next_button" in traversal_hints:
                filtered["next_button_selector"] = data.get("next_button_selector")
            if "pagination" in traversal_hints:
                filtered["pagination_url"] = data.get("pagination_url")
                filtered["pagination_max"] = data.get("pagination_max")
            if "sub_pages" in traversal_hints:
                filtered["sub_page_selector"] = data.get("sub_page_selector")
                filtered["sub_page_url_attr"] = data.get("sub_page_url_attr", "href")
                filtered["sub_page_recursive"] = data.get("sub_page_recursive", False)

            rules = PageRules(**{k: v for k, v in filtered.items() if v is not None})
            self._log_rules(rules)
            return rules

        except Exception as e:
            print(f"[RuleDiscoverer] 分析失败: {e}，返回空规则")
            return PageRules()

    @staticmethod
    def _get_clean_snippet(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all(["script", "style", "noscript"]):
            tag.decompose()

        main = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|container", re.I))
            or soup.body
            or soup
        )

        content = str(main)
        if len(content) > MAX_HTML_SIZE:
            content = content[:MAX_HTML_SIZE]
        return content

    @staticmethod
    def _log_rules(rules: PageRules):
        found = []
        if rules.load_more_selector:
            found.append(f"load_more: {rules.load_more_selector}")
        if rules.next_button_selector:
            found.append(f"next_button: {rules.next_button_selector}")
        if rules.pagination_url:
            found.append(f"pagination: {rules.pagination_url} (max={rules.pagination_max})")
        if rules.sub_page_selector:
            found.append(f"sub_pages: {rules.sub_page_selector} (recursive={rules.sub_page_recursive})")

        if found:
            print(f"[RuleDiscoverer] 发现 {len(found)} 条规则:")
            for r in found:
                print(f"  - {r}")
        else:
            print("[RuleDiscoverer] 未找到匹配的遍历规则")
