"""RuleDiscoverer: LLM 分析页面结构，**只发现用户要求的遍历模式**。
用户没要求遍历 → 不调用 LLM，直接返回空规则（单页提取）。
"""

import json
import logging
import re

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from agent_scraper.core.llm import create_openai_client, get_model_name
from agent_scraper.core.models import PageRules

logger = logging.getLogger(__name__)

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
  "sub_page_url_filter": "URL中必须包含的关键词（如'/tree/'），用于过滤掉文件链接，没有则null",
  "sub_page_recursive": false
}}

CSS选择器要求:
1. 尽量精确，能唯一定位到目标元素
2. 用户没要求的模式，对应字段必须返回null
3. 对于 load_more，优先用精确选择器；如果按钮没有 class/id，可以用文本匹配描述
4. 对于 sub_pages，选择器必须**只匹配文件夹/目录链接**，不要匹配单个文件链接。
   文件夹通常有文件夹图标(svg)、特殊class、或URL中包含 /tree/ 等标志。
   如果无法区分文件夹和文件，在 sub_page_url_pattern 中说明过滤规则。

只输出JSON。
"""

MAX_HTML_SIZE = 50 * 1024  # 50KB — RuleDiscoverer 只需导航骨架，不需要完整内容


class RuleDiscoverer:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or create_openai_client()
        self.model = get_model_name()

    async def discover(
        self, html: str, current_url: str = "", traversal_hints: list[str] | None = None
    ) -> PageRules:
        """
        根据用户指定的 traversal_hints 分析页面。
        如果 hints 为空 → 直接返回空规则（单页模式，不调用 LLM）。
        """
        if not traversal_hints:
            logger.info("用户未要求遍历，单页模式")
            return PageRules()

        logger.info("用户要求: %s", traversal_hints)
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
                filtered["sub_page_url_filter"] = data.get("sub_page_url_filter")
                filtered["sub_page_recursive"] = data.get("sub_page_recursive", False)

            rules = PageRules(**{k: v for k, v in filtered.items() if v is not None})
            self._log_rules(rules)
            return rules

        except Exception as e:
            logger.error("分析失败: %s，返回空规则", e)
            return PageRules()

    @staticmethod
    def _get_clean_snippet(html: str) -> str:
        """精简 HTML：去噪 + 压缩重复列表项，只保留导航骨架。"""
        soup = BeautifulSoup(html, "lxml")

        # 1. 去掉无用标签
        for tag in soup.find_all(["script", "style", "noscript", "svg", "img", "picture", "video"]):
            tag.decompose()

        # 2. 去掉内联样式和 data-* 属性（减少噪声）
        for tag in soup.find_all(True):
            remove_attrs = [a for a in tag.attrs if a.startswith("data-") or a == "style"]
            for a in remove_attrs:
                del tag[a]

        main = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|container", re.I))
            or soup.body
            or soup
        )

        # 3. 压缩重复列表项：如果 <ul>/<ol>/<tbody> 有 >5 个同类子项，只保留前3+后1
        for container in main.find_all(["ul", "ol", "tbody", "div"]):
            children = [c for c in container.children if hasattr(c, "name") and c.name]
            if len(children) > 5:
                # 检查子元素是否同质（同标签名）
                tag_names = [c.name for c in children]
                most_common = max(set(tag_names), key=tag_names.count)
                same_tag = [c for c in children if c.name == most_common]
                if len(same_tag) > 5:
                    # 保留前3个 + 最后1个，中间替换为占位提示
                    keep_head = same_tag[:3]
                    keep_tail = [same_tag[-1]]
                    removed_count = len(same_tag) - 4
                    for item in same_tag:
                        if item not in keep_head and item not in keep_tail:
                            item.decompose()
                    # 插入占位注释
                    placeholder = soup.new_string(f"\n<!-- ... 省略 {removed_count} 个同类元素 ... -->\n")
                    keep_head[-1].insert_after(placeholder)

        content = str(main)
        if len(content) > MAX_HTML_SIZE:
            content = content[:MAX_HTML_SIZE]

        logger.info("HTML 精简: %.0fKB → %.0fKB", len(html)/1024, len(content)/1024)
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
            logger.info("发现 %d 条规则:", len(found))
            for r in found:
                logger.info("  - %s", r)
        else:
            logger.info("未找到匹配的遍历规则")
