"""PageIterator: 纯代码页面遍历器，零AI
根据 PageRules 机械执行所有翻页/子页面遍历。
每次导航后重新获取 page 引用，确保不会因为页面切换而失效。
"""

import asyncio
import json as json_mod
import logging
import re
from collections.abc import AsyncGenerator
from urllib.parse import urljoin, urlparse

from agent_scraper.core.models import PageRules

logger = logging.getLogger(__name__)

# 未指定 max_pages 时的默认翻页上限（pagination_url 和 next_button 共用）
DEFAULT_MAX_PAGES = 20


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

    async def iterate(self, first_html: str, rules: PageRules, base_url: str = "") -> AsyncGenerator[str, None]:
        """根据规则遍历所有页面，逐页 yield HTML（流式，不缓存）"""
        max_pages = rules.pagination_max

        # 1. load_more: 仅在有明确选择器时执行（避免误点击无关按钮）
        if rules.load_more_selector:
            await self._try_load_more(rules.load_more_selector)
            first_html = await self._get_html()

        # 2. sub_pages: 递归遍历子页面
        if rules.sub_page_selector:
            yield first_html
            async for html in self._do_sub_pages(
                selector=rules.sub_page_selector,
                url_attr=rules.sub_page_url_attr,
                url_filter=rules.sub_page_url_filter,
                load_more_selector=rules.load_more_selector,
                base_url=base_url,
            ):
                yield html

        # 3. pagination URL 模式
        elif rules.pagination_url:
            yield first_html
            async for html in self._do_pagination_url(rules.pagination_url, max_pages or DEFAULT_MAX_PAGES):
                yield html

        # 4. next_button 翻页
        elif rules.next_button_selector:
            yield first_html
            async for html in self._do_next_button(rules.next_button_selector, max_pages):
                yield html

        # 5. 无规则: 单页
        else:
            yield first_html

    # ── load_more ────────────────────────────────────────

    async def _try_load_more(self, selector: str | None, max_clicks: int = 50):
        """尝试点击 Load more 按钮。有选择器用选择器，没有用通用文本匹配"""
        click_count = 0
        prev_height = 0
        while click_count < max_clicks:
            js = self._build_load_more_js(selector)
            result = await self._eval(js)
            if result != "clicked":
                break
            click_count += 1
            if click_count % 5 == 0:
                logger.info("load_more 已点击 %d 次...", click_count)
            await asyncio.sleep(1.5)
            # 检测页面是否有变化（防止按钮始终可见的死循环）
            cur_height = await self._eval("() => document.body.scrollHeight")
            if cur_height == prev_height:
                logger.info("load_more 页面无变化，停止")
                break
            prev_height = cur_height
        if click_count >= max_clicks:
            logger.warning("load_more 达到上限 %d 次，停止", max_clicks)
        elif click_count > 0:
            logger.info("load_more 完成，共点击 %d 次", click_count)

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
        url_filter: str | None = None,
        load_more_selector: str | None = None,
        base_url: str = "",
        visited: set | None = None,
        depth: int = 0,
        max_depth: int = 5,
    ) -> AsyncGenerator[str, None]:
        """递归提取子页面链接，逐个进入，yield HTML。自动检测更深层子页面。"""
        if visited is None:
            visited = set()

        if depth >= max_depth:
            logger.warning("达到最大递归深度 %d，停止", max_depth)
            return

        # 提取当前页的子页面链接
        urls = await self._extract_links(selector, url_attr, base_url)
        urls = [u for u in urls if u not in visited]

        # 应用 URL 过滤
        if url_filter:
            before = len(urls)
            urls = [u for u in urls if url_filter in u]
            filtered_out = before - len(urls)
            if filtered_out:
                logger.info("URL 过滤 '%s': %d → %d (排除 %d)", url_filter, before, len(urls), filtered_out)
        else:
            # 通用兜底：排除明显的文件链接（含 /blob/、/raw/ 或常见文件扩展名）
            before = len(urls)
            urls = [u for u in urls if not self._is_file_url(u)]
            filtered_out = before - len(urls)
            if filtered_out:
                logger.info("自动排除 %d 个文件链接", filtered_out)

        if not urls:
            return

        logger.info("发现 %d 个子页面 (depth=%d)", len(urls), depth)

        for i, url in enumerate(urls):
            if url in visited:
                continue
            visited.add(url)
            logger.info("进入子页面 [%d/%d]: %s", i + 1, len(urls), url)
            try:
                await self._goto(url)

                # 每个子页面都尝试 Load more
                await self._try_load_more(load_more_selector)

                html = await self._get_html()
                yield html

                # 递归：检查这个子页面里是否还有更深层子页面
                async for deeper_html in self._do_sub_pages(
                    selector=selector,
                    url_attr=url_attr,
                    url_filter=url_filter,
                    load_more_selector=load_more_selector,
                    base_url=url,
                    visited=visited,
                    depth=depth + 1,
                    max_depth=max_depth,
                ):
                    yield deeper_html

            except Exception as e:
                logger.error("子页面 [%d] 失败: %s", i + 1, e)

    # 常见文件扩展名（用于过滤非目录链接）
    # 注意：html/css/scss/less/svg 是网页/样式/图标，不算"文件"，不应过滤
    _FILE_EXTENSIONS = re.compile(
        r'\.'
        r'(?:'
        # 文本/数据
        r'md|txt|json|jsonl|csv|tsv|xml|yaml|yml|toml|cfg|ini|conf|log'
        # 源代码
        r'|py|js|ts|java|c|cpp|h|go|rs|rb|php|sh|bat|ps1'
        # 图片
        r'|png|jpg|jpeg|gif|ico|webp|bmp|tiff'
        # 文档
        r'|pdf|doc|docx|xls|xlsx|ppt|pptx'
        # 压缩包
        r'|zip|tar|gz|bz2|7z|rar'
        # 二进制/模型
        r'|bin|exe|dll|so|dylib|whl|safetensors|gguf|pt|onnx'
        # dotfiles
        r'|gitattributes|gitignore|gitmodules|dockerignore|editorconfig'
        r')$',
        re.I,
    )

    @classmethod
    def _is_file_url(cls, url: str) -> bool:
        """判断 URL 是否指向单个文件而非目录/页面（基于扩展名）"""
        path = urlparse(url).path
        return bool(cls._FILE_EXTENSIONS.search(path))

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

    async def _do_pagination_url(self, url_pattern: str, max_pages: int) -> AsyncGenerator[str, None]:
        """URL 模板分页。max_pages 为总页数（含第1页）。"""
        logger.info("URL 分页: max=%d 页", max_pages)
        for n in range(2, max_pages + 1):  # 第1页已有，从第2页开始
            url = url_pattern.replace("{n}", str(n))
            try:
                await self._goto(url)
                html = await self._get_html()
                if len(html) < 1000:
                    logger.info("第 %d 页内容过少，停止", n)
                    break
                yield html
                if n % 5 == 0:
                    logger.info("已完成 %d 页...", n)
            except Exception as e:
                logger.error("分页 %d 失败: %s，停止", n, e)
                break

    # ── next_button ──────────────────────────────────────

    async def _do_next_button(self, selector: str, max_pages: int | None = None) -> AsyncGenerator[str, None]:
        """点击"下一页"翻页。max_pages 为总页数限制（含第1页），None 则不限制。"""
        # max_extra = 需要额外翻的页数（第1页已有，所以减1）
        max_extra = (max_pages - 1) if max_pages else DEFAULT_MAX_PAGES
        logger.info("翻页按钮: %s (最多翻 %d 页)", selector, max_extra)
        safe_sel = selector.replace("'", "\\'")
        page_count = 0
        for i in range(max_extra):
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
            yield await self._get_html()
            page_count += 1
            if (i + 1) % 5 == 0:
                logger.info("已翻 %d 页...", i + 1)
        logger.info("翻页完成，共 %d 个额外页面", page_count)
