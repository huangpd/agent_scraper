"""PageIterator: 纯代码页面遍历器，零AI

设计: 两个正交维度自动组合，无需为每种模式组合写专用代码
  维度 1 — 列表页翻页: 单页 / pagination_url / next_button  → _iter_list_pages
  维度 2 — 每页处理:   直接提取 / 进详情页提取              → iterate 中的 has_sub 分支

组合方式:
  Phase 1: _iter_list_pages yield 列表页 → 在每个列表页上提取详情页链接（不离开列表页）
  Phase 2: 逐一访问详情页，yield HTML
  因为 Phase 1 不做页面跳转，所以 next_button/pagination_url 的翻页状态不会被破坏。
"""

import asyncio
import json as json_mod
import logging
import re
from collections.abc import AsyncGenerator
from urllib.parse import urljoin, urlparse

from agent_scraper.core.models import PageRules

logger = logging.getLogger(__name__)

# 未指定 max_pages 时的默认翻页上限
DEFAULT_MAX_PAGES = 20


class PageIterator:
    def __init__(self, browser):
        """browser: browser_use BrowserSession"""
        self.browser = browser

    # ── 基础浏览器操作 ────────────────────────────────────

    async def _get_page(self):
        """每次操作前重新获取当前 page，防止引用失效"""
        page = await self.browser.get_current_page()
        if not page:
            raise RuntimeError("浏览器页面丢失")
        return page

    async def _eval(self, js: str) -> str:
        page = await self._get_page()
        return await page.evaluate(js)

    async def _goto(self, url: str):
        page = await self._get_page()
        await page.goto(url)
        await asyncio.sleep(1)

    async def _get_html(self) -> str:
        return await self._eval("() => document.documentElement.outerHTML")

    async def _current_url(self) -> str:
        try:
            return await self._eval("() => window.location.href")
        except Exception:
            return ""

    # ── 核心入口 ──────────────────────────────────────────

    async def iterate(
        self, first_html: str, rules: PageRules, base_url: str = "",
        load_more_hint: bool = False, load_more_text: str | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """根据规则遍历所有页面，yield (page_url, html) 元组。

        两个维度自动组合:
          列表页序列 (单页/pagination/next_button) × 每页处理 (直接/进详情页)
        page_url 标识数据来源，下游可用它填充 URL 字段或做调试追踪。

        Args:
            load_more_hint: 用户是否请求了 load_more。为 True 时即使无精确选择器
                也会尝试正则兜底点击 "Load more" 按钮。
            load_more_text: 用户描述的自定义按钮文本，如"更多>>"，
                会被加入正则匹配模式。
        """
        # 预处理: load_more — 有精确选择器或用户请求了 load_more 都尝试
        if rules.load_more_selector or load_more_hint:
            await self._try_load_more(rules.load_more_selector, custom_text=load_more_text)
            first_html = await self._get_html()

        has_sub = bool(rules.sub_page_selector)
        has_paging = bool(rules.pagination_url) or bool(rules.next_button_selector)

        # ① 子页面递归模式 (无翻页，如目录树遍历)
        if has_sub and not has_paging:
            yield base_url, first_html
            async for sub_url, html in self._do_sub_pages_recursive(
                selector=rules.sub_page_selector,
                url_attr=rules.sub_page_url_attr,
                url_filter=rules.sub_page_url_filter,
                load_more_selector=rules.load_more_selector,
                load_more_text=load_more_text,
                base_url=base_url,
            ):
                yield sub_url, html
            return

        # ② 通用组合流程: 列表页翻页 × 每页处理
        if has_sub:
            # 有详情页 → 两阶段
            #   Phase 1: 遍历列表页，在每页上提取详情页链接（不离开列表页，保持翻页状态）
            #   Phase 2: 逐一访问详情页
            visited: set[str] = set()
            all_detail_urls: list[str] = []

            async for page_num, page_url, _ in self._iter_list_pages(
                first_html, rules, base_url,
            ):
                urls = await self._extract_detail_urls(rules, page_url, visited)
                logger.info("列表页 [%d]: 发现 %d 个详情页", page_num, len(urls))
                visited.update(urls)
                all_detail_urls.extend(urls)

            logger.info("共收集 %d 个详情页链接，开始逐一访问", len(all_detail_urls))
            for i, url in enumerate(all_detail_urls):
                logger.info("访问详情页 [%d/%d]: %s", i + 1, len(all_detail_urls), url)
                try:
                    await self._goto(url)
                    yield url, await self._get_html()
                except Exception as e:
                    logger.error("详情页 [%d] 失败: %s", i + 1, e)
            logger.info("遍历完成，共访问 %d 个详情页", len(all_detail_urls))
        else:
            # 无详情页 → 直接 yield 每个列表页
            async for _, page_url, page_html in self._iter_list_pages(
                first_html, rules, base_url,
            ):
                yield page_url, page_html

    # ── 维度 1: 列表页序列 ────────────────────────────────

    async def _iter_list_pages(
        self, first_html: str, rules: PageRules, base_url: str,
    ) -> AsyncGenerator[tuple[int, str, str], None]:
        """yield (page_num, page_url, html) — 统一的列表页序列。

        自动处理: 单页 / pagination_url / next_button，上层无需关心翻页方式。
        """
        max_pages = rules.pagination_max or DEFAULT_MAX_PAGES

        # 确保浏览器在列表页（重试场景下可能在其他页面）
        if base_url:
            await self._goto(base_url)

        # 第 1 页
        yield 1, base_url, first_html

        # 后续页
        if rules.pagination_url:
            async for item in self._paginate_by_url(rules.pagination_url, max_pages):
                yield item
        elif rules.next_button_selector:
            async for item in self._paginate_by_button(rules.next_button_selector, max_pages):
                yield item

    async def _paginate_by_url(
        self, url_pattern: str, max_pages: int,
    ) -> AsyncGenerator[tuple[int, str, str], None]:
        """URL 模板分页"""
        for n in range(2, max_pages + 1):
            url = url_pattern.replace("{n}", str(n))
            try:
                await self._goto(url)
                html = await self._get_html()
                if len(html) < 1000:
                    logger.info("列表页 [%d] 内容过少，停止", n)
                    break
                yield n, url, html
                if n % 5 == 0:
                    logger.info("已翻 %d 页...", n)
            except Exception as e:
                logger.error("列表页 [%d] 失败: %s", n, e)
                break

    async def _paginate_by_button(
        self, selector: str, max_pages: int,
    ) -> AsyncGenerator[tuple[int, str, str], None]:
        """点击翻页按钮"""
        max_extra = max_pages - 1
        safe_sel = selector.replace("'", "\\'")
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
            url = await self._current_url()
            html = await self._get_html()
            yield i + 2, url, html
            if (i + 1) % 5 == 0:
                logger.info("已翻 %d 页...", i + 1)

    # ── 维度 2: 详情页链接提取 ────────────────────────────

    async def _extract_detail_urls(
        self, rules: PageRules, base_url: str, visited: set[str],
    ) -> list[str]:
        """从当前浏览器页面提取详情页链接（已去重、已过滤）"""
        urls = await self._extract_links(
            rules.sub_page_selector, rules.sub_page_url_attr, base_url,
        )
        urls = [u for u in urls if u not in visited]
        if rules.sub_page_url_filter:
            urls = [u for u in urls if rules.sub_page_url_filter in u]
        else:
            urls = [u for u in urls if not self._is_file_url(u)]
        return urls

    async def _extract_links(self, selector: str, url_attr: str, base_url: str) -> list[str]:
        """从当前页面提取链接"""
        # 用 JSON.stringify 传参，避免选择器中的引号转义问题
        import json as _json
        sel_json = _json.dumps(selector)
        attr_json = _json.dumps(url_attr)

        raw = await self._eval(
            f"() => {{"
            f"  const sel = {sel_json};"
            f"  const attr = {attr_json};"
            f"  const els = document.querySelectorAll(sel);"
            f"  return JSON.stringify([...els].map(el => el.getAttribute(attr)).filter(Boolean));"
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
        logger.debug("_extract_links: selector='%s' matched=%d, dedup=%d", selector, len(raw_urls), len(urls))
        return urls

    # ── load_more ─────────────────────────────────────────

    async def _try_load_more(self, selector: str | None, max_clicks: int = 20,
                             custom_text: str | None = None):
        """尝试点击 Load more 按钮。

        停止条件（任一触发）:
        1. 按钮消失（JS 返回 not_found）
        2. 连续 2 次点击后页面高度无变化
        3. 点击后页面被整页刷新（DOM 标记消失，即使 URL 不变）
        4. 达到 max_clicks 上限
        """
        click_count = 0
        prev_height = 0
        stale_count = 0

        while click_count < max_clicks:
            # 每次点击前在 DOM 中插入标记，点击后检查是否消失（整页刷新检测）
            try:
                await self._eval(
                    "() => document.body.setAttribute('data-lm-marker', '1')"
                )
            except Exception:
                pass

            js = self._build_load_more_js(selector, custom_text=custom_text)
            try:
                result = await self._eval(js)
            except Exception as e:
                logger.warning("load_more JS 执行异常: %s，停止", e)
                break
            if result != "clicked":
                break
            click_count += 1
            if click_count % 5 == 0:
                logger.info("load_more 已点击 %d 次...", click_count)
            # 等待页面加载
            wait = min(2.0 + click_count * 0.1, 4.0)
            await asyncio.sleep(wait)

            # 滚动到底部，让新加载的内容和按钮可见
            try:
                await self._eval("() => window.scrollTo(0, document.body.scrollHeight)")
            except Exception:
                pass
            await asyncio.sleep(0.3)

            # 检测整页刷新：DOM 标记消失 → 页面被重新加载（即使 URL 不变）
            try:
                marker = await self._eval(
                    "() => document.body.getAttribute('data-lm-marker')"
                )
            except Exception:
                marker = None
            if not marker:
                logger.info("load_more 触发了整页刷新（DOM 标记消失），停止")
                break

            try:
                cur_height = await self._eval("() => document.body.scrollHeight")
            except Exception:
                logger.warning("load_more 无法获取页面高度，停止")
                break
            if cur_height == prev_height:
                stale_count += 1
                if stale_count >= 2:
                    logger.info("load_more 连续 %d 次页面无变化，停止", stale_count)
                    break
            else:
                stale_count = 0
            prev_height = cur_height
        if click_count >= max_clicks:
            logger.warning("load_more 达到上限 %d 次，停止", max_clicks)
        elif click_count > 0:
            logger.info("load_more 完成，共点击 %d 次", click_count)

    @staticmethod
    def _build_load_more_js(selector: str | None, custom_text: str | None = None) -> str:
        # 构建正则模式：内置 + 用户自定义文本
        builtin = "load more|加载更多|show more|load more files|更多"
        if custom_text:
            # 转义正则特殊字符
            escaped = re.escape(custom_text)
            pattern = f"{escaped}|{builtin}"
        else:
            pattern = builtin

        if selector:
            # 检测 :contains() 伪选择器（jQuery 语法，原生不支持）→ 转为文本匹配
            contains_match = re.match(r'^(.+?):contains\(["\']?(.+?)["\']?\)$', selector, re.I)
            if contains_match:
                tag_sel = contains_match.group(1).strip()
                contains_text = contains_match.group(2).strip()
                escaped_text = re.escape(contains_text)
                # 合并 :contains 文本到正则模式
                pattern = f"{escaped_text}|{pattern}"
                logger.info("检测到 :contains() 伪选择器，转为文本匹配: tag='%s', text='%s'", tag_sel, contains_text)
                return (
                    f"() => {{"
                    f"  const all = [...document.querySelectorAll('{tag_sel}, button, a')];"
                    f"  const btn = all.find(e => /{pattern}/i.test(e.textContent.trim()));"
                    f"  if (btn && btn.offsetParent !== null) {{"
                    f"    btn.scrollIntoView(); btn.click(); return 'clicked';"
                    f"  }}"
                    f"  return 'not_found';"
                    f"}}"
                )

            safe_sel = selector.replace("'", "\\'")
            return (
                f"() => {{"
                f"  let btn = document.querySelector('{safe_sel}');"
                f"  if (!btn || btn.offsetParent === null) {{"
                f"    const all = [...document.querySelectorAll('button, a')];"
                f"    btn = all.find(e => /{pattern}/i.test(e.textContent.trim()));"
                f"  }}"
                f"  if (btn && btn.offsetParent !== null) {{"
                f"    btn.scrollIntoView(); btn.click(); return 'clicked';"
                f"  }}"
                f"  return 'not_found';"
                f"}}"
            )
        else:
            return (
                f"() => {{"
                f"  const all = [...document.querySelectorAll('button, a')];"
                f"  const btn = all.find(e => /{pattern}/i.test(e.textContent.trim()));"
                f"  if (btn && btn.offsetParent !== null) {{"
                f"    btn.scrollIntoView(); btn.click(); return 'clicked';"
                f"  }}"
                f"  return 'not_found';"
                f"}}"
            )

    # ── 子页面递归遍历 (目录树等场景) ─────────────────────

    async def _do_sub_pages_recursive(
        self,
        selector: str,
        url_attr: str,
        url_filter: str | None = None,
        load_more_selector: str | None = None,
        load_more_text: str | None = None,
        base_url: str = "",
        visited: set | None = None,
        depth: int = 0,
        max_depth: int = 5,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """递归提取子页面链接，逐个进入，yield (url, html)。用于目录树等递归结构。"""
        if visited is None:
            visited = set()
        # 把当前页面加入 visited，防止子页面链接指回自己
        if base_url:
            visited.add(base_url)
            # 也排除末尾带/不带斜杠的变体
            visited.add(base_url.rstrip("/"))
            visited.add(base_url.rstrip("/") + "/")

        if depth >= max_depth:
            logger.warning("达到最大递归深度 %d，停止", max_depth)
            return

        raw_urls = await self._extract_links(selector, url_attr, base_url)
        urls = [u for u in raw_urls if u not in visited]

        if url_filter:
            before = len(urls)
            urls = [u for u in urls if url_filter in u]
            filtered_out = before - len(urls)
            if filtered_out:
                logger.info("URL 过滤 '%s': %d → %d (排除 %d)", url_filter, before, len(urls), filtered_out)
        else:
            before = len(urls)
            urls = [u for u in urls if not self._is_file_url(u)]
            filtered_out = before - len(urls)
            if filtered_out:
                logger.info("自动排除 %d 个文件链接", filtered_out)

        if not urls:
            logger.info(
                "子页面链接为空 (depth=%d, selector='%s', raw=%d, after_visited=%d, after_filter=%d)",
                depth, selector, len(raw_urls), len([u for u in raw_urls if u not in visited]), 0,
            )
            return

        logger.info("发现 %d 个子页面 (depth=%d)", len(urls), depth)

        for i, url in enumerate(urls):
            if url in visited:
                continue
            visited.add(url)
            logger.info("进入子页面 [%d/%d]: %s", i + 1, len(urls), url)
            try:
                await self._goto(url)
                await self._try_load_more(load_more_selector, custom_text=load_more_text)
                html = await self._get_html()
                yield url, html

                async for deeper_url, deeper_html in self._do_sub_pages_recursive(
                    selector=selector,
                    url_attr=url_attr,
                    url_filter=url_filter,
                    load_more_selector=load_more_selector,
                    load_more_text=load_more_text,
                    base_url=url,
                    visited=visited,
                    depth=depth + 1,
                    max_depth=max_depth,
                ):
                    yield deeper_url, deeper_html

            except Exception as e:
                logger.error("子页面 [%d] 失败: %s", i + 1, e)

    # ── 工具方法 ──────────────────────────────────────────

    _FILE_EXTENSIONS = re.compile(
        r'\.(md|txt|json|jsonl|csv|tsv|xml|yaml|yml|toml|cfg|ini|conf|log'
        r'|py|js|ts|java|c|cpp|h|go|rs|rb|php|sh|bat|ps1'
        r'|html|css|scss|less'
        r'|png|jpg|jpeg|gif|svg|ico|webp|bmp'
        r'|pdf|doc|docx|xls|xlsx|ppt|pptx'
        r'|zip|tar|gz|bz2|7z|rar'
        r'|bin|exe|dll|so|dylib|whl|safetensors|gguf|pt|onnx'
        r'|gitattributes|gitignore|gitmodules|dockerignore|editorconfig)$',
        re.I,
    )

    _FILE_PATH_PATTERNS = re.compile(r'/blob/|/raw/')

    @classmethod
    def _is_file_url(cls, url: str) -> bool:
        """判断 URL 是否指向单个文件而非目录/页面"""
        path = urlparse(url).path
        if cls._FILE_PATH_PATTERNS.search(path):
            return True
        if cls._FILE_EXTENSIONS.search(path):
            return True
        return False
