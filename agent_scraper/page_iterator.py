"""PageIterator: 纯代码页面遍历器，零AI
根据 PageRules 机械执行所有翻页/子页面遍历。
每次导航后重新获取 page 引用，确保不会因为页面切换而失效。
"""

import asyncio
import json as json_mod
from urllib.parse import urljoin

from agent_scraper.models import PageRules


class PageIterator:
    def __init__(self, browser):
        """browser: browser_use BrowserSession"""
        self.browser = browser

    async def _get_page(self):
        """每次操作前重新获取当前 page，防止引用失效"""
        page = await self.browser.get_current_page()
        if not page:
            raise RuntimeError("浏览器页面丢失")
        return page

    async def _eval(self, js: str) -> str:
        """安全执行 JS evaluate"""
        page = await self._get_page()
        return await page.evaluate(js)

    async def _goto(self, url: str):
        """导航到 URL"""
        page = await self._get_page()
        await page.goto(url)
        await asyncio.sleep(2)

    async def _get_html(self) -> str:
        return await self._eval("() => document.documentElement.outerHTML")

    async def iterate(self, first_html: str, rules: PageRules, base_url: str = "") -> list[str]:
        """根据规则遍历所有页面，返回 HTML 列表"""
        htmls = []

        # 1. load_more: 在当前页循环点击
        if rules.load_more_selector:
            await self._try_load_more(rules.load_more_selector)
            first_html = await self._get_html()
        else:
            # 即使没有特定选择器，也尝试通用 Load more
            await self._try_load_more(None)
            first_html = await self._get_html()

        # 2. sub_pages: 递归遍历子页面
        if rules.sub_page_selector:
            htmls.append(first_html)
            sub_htmls = await self._do_sub_pages(
                selector=rules.sub_page_selector,
                url_attr=rules.sub_page_url_attr,
                load_more_selector=rules.load_more_selector,
                base_url=base_url,
            )
            htmls.extend(sub_htmls)

        # 3. pagination URL 模式
        elif rules.pagination_url:
            htmls.append(first_html)
            htmls.extend(await self._do_pagination_url(rules.pagination_url, rules.pagination_max or 20))

        # 4. next_button 翻页
        elif rules.next_button_selector:
            htmls.append(first_html)
            htmls.extend(await self._do_next_button(rules.next_button_selector))

        # 5. 无规则: 单页
        else:
            htmls.append(first_html)

        print(f"[PageIterator] 共收集 {len(htmls)} 个页面的 HTML")
        return htmls

    # ── load_more ────────────────────────────────────────

    async def _try_load_more(self, selector: str | None):
        """尝试点击 Load more 按钮。有选择器用选择器，没有用通用文本匹配"""
        click_count = 0
        while True:
            js = self._build_load_more_js(selector)
            result = await self._eval(js)
            if result != "clicked":
                break
            click_count += 1
            if click_count % 5 == 0:
                print(f"  [PageIterator] load_more 已点击 {click_count} 次...")
            await asyncio.sleep(1.5)
        if click_count > 0:
            print(f"  [PageIterator] load_more 完成，共点击 {click_count} 次")

    @staticmethod
    def _build_load_more_js(selector: str | None) -> str:
        """构建 Load more 点击的 JS"""
        if selector:
            safe_sel = selector.replace("'", "\\'")
            return (
                f"() => {{"
                f"  let btn = document.querySelector('{safe_sel}');"
                f"  if (!btn || btn.offsetParent === null) {{"
                f"    const all = [...document.querySelectorAll('button, a')];"
                f"    btn = all.find(e => /load more|加载更多|show more/i.test(e.textContent.trim()));"
                f"  }}"
                f"  if (btn && btn.offsetParent !== null) {{"
                f"    btn.scrollIntoView(); btn.click(); return 'clicked';"
                f"  }}"
                f"  return 'not_found';"
                f"}}"
            )
        else:
            return (
                "() => {"
                "  const all = [...document.querySelectorAll('button, a')];"
                "  const btn = all.find(e => /load more|加载更多|show more|load more files/i.test(e.textContent.trim()));"
                "  if (btn && btn.offsetParent !== null) {"
                "    btn.scrollIntoView(); btn.click(); return 'clicked';"
                "  }"
                "  return 'not_found';"
                "}"
            )

    # ── sub_pages (真正递归) ─────────────────────────────

    async def _do_sub_pages(
        self,
        selector: str,
        url_attr: str,
        load_more_selector: str | None,
        base_url: str,
        visited: set | None = None,
        depth: int = 0,
        max_depth: int = 5,
    ) -> list[str]:
        """递归提取子页面链接，逐个进入，收集 HTML。自动检测更深层子页面。"""
        if visited is None:
            visited = set()

        if depth >= max_depth:
            print(f"  [PageIterator] 达到最大递归深度 {max_depth}，停止")
            return []

        indent = "  " * (depth + 1)

        # 提取当前页的子页面链接
        urls = await self._extract_links(selector, url_attr, base_url)
        urls = [u for u in urls if u not in visited]

        if not urls:
            return []

        print(f"{indent}[PageIterator] 发现 {len(urls)} 个子页面 (depth={depth})")

        htmls = []
        for i, url in enumerate(urls):
            visited.add(url)
            print(f"{indent}[PageIterator] 进入子页面 [{i + 1}/{len(urls)}]: {url}")
            try:
                await self._goto(url)

                # 每个子页面都尝试 Load more
                await self._try_load_more(load_more_selector)

                html = await self._get_html()
                htmls.append(html)

                # 递归：检查这个子页面里是否还有更深层子页面
                deeper = await self._do_sub_pages(
                    selector=selector,
                    url_attr=url_attr,
                    load_more_selector=load_more_selector,
                    base_url=url,
                    visited=visited,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                htmls.extend(deeper)

            except Exception as e:
                print(f"{indent}[PageIterator] 子页面 [{i + 1}] 失败: {e}")

        return htmls

    async def _extract_links(self, selector: str, url_attr: str, base_url: str) -> list[str]:
        """从当前页面提取子页面链接"""
        safe_sel = selector.replace("'", "\\'")
        safe_attr = url_attr.replace("'", "\\'")

        raw = await self._eval(
            f"() => {{"
            f"  const els = document.querySelectorAll('{safe_sel}');"
            f"  return JSON.stringify([...els].map(el => el.getAttribute('{safe_attr}')).filter(Boolean));"
            f"}}"
        )

        try:
            raw_urls = json_mod.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            raw_urls = []

        if not isinstance(raw_urls, list):
            raw_urls = []

        # 补全相对 URL + 去重
        seen = set()
        urls = []
        for u in raw_urls:
            if not u:
                continue
            full = urljoin(base_url, u) if not u.startswith("http") else u
            if full not in seen:
                seen.add(full)
                urls.append(full)
        return urls

    # ── pagination URL ───────────────────────────────────

    async def _do_pagination_url(self, url_pattern: str, max_pages: int) -> list[str]:
        print(f"  [PageIterator] URL 分页: max={max_pages}")
        htmls = []
        for n in range(2, max_pages + 1):
            url = url_pattern.replace("{n}", str(n))
            try:
                await self._goto(url)
                html = await self._get_html()
                if len(html) < 1000:
                    print(f"  [PageIterator] 第 {n} 页内容过少，停止")
                    break
                htmls.append(html)
                if n % 5 == 0:
                    print(f"  [PageIterator] 已完成 {n} 页...")
            except Exception as e:
                print(f"  [PageIterator] 分页 {n} 失败: {e}，停止")
                break
        return htmls

    # ── next_button ──────────────────────────────────────

    async def _do_next_button(self, selector: str) -> list[str]:
        print(f"  [PageIterator] 翻页按钮: {selector}")
        safe_sel = selector.replace("'", "\\'")
        htmls = []
        for i in range(100):
            result = await self._eval(
                f"() => {{"
                f"  const btn = document.querySelector('{safe_sel}');"
                f"  if (btn && btn.offsetParent !== null) {{ btn.click(); return 'clicked'; }}"
                f"  return 'not_found';"
                f"}}"
            )
            if result != "clicked":
                break
            await asyncio.sleep(2)
            htmls.append(await self._get_html())
            if (i + 1) % 5 == 0:
                print(f"  [PageIterator] 已翻 {i + 1} 页...")
        print(f"  [PageIterator] 翻页完成，共 {len(htmls)} 个额外页面")
        return htmls
