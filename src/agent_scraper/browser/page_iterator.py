"""PageIterator: 纯代码页面遍历器，零AI
根据 PageRules 机械执行所有翻页/子页面遍历。
每次导航后重新获取 page 引用，确保不会因为页面切换而失效。
"""

import asyncio
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

    async def _get_page(self, retries: int = 3):
        """每次操作前重新获取当前 page，防止引用失效。

        恢复策略（按优先级）：
        1. 快速路径：直接获取
        2. 协同 SessionManager：等待其自动恢复 detached target（事件驱动，非轮询）
        3. 传统重试：兼容无 SessionManager 的场景
        4. 已有标签页兜底
        5. 创建新标签页
        """
        # 快速路径
        page = await self.browser.get_current_page()
        if page:
            return page

        # 等待 SessionManager 自动恢复（协同而非赛跑）
        sm = getattr(self.browser, 'session_manager', None)
        if sm:
            logger.info("页面丢失，等待 SessionManager 恢复...")
            try:
                recovered = await sm.ensure_valid_focus(timeout=5.0)
                if recovered:
                    page = await self.browser.get_current_page()
                    if page:
                        logger.info("SessionManager 恢复成功")
                        return page
            except Exception as e:
                logger.warning("SessionManager 恢复异常: %s", e)

        # 恢复失败，传统重试（兼容无 SessionManager 的场景）
        for attempt in range(retries):
            await asyncio.sleep(2)
            page = await self.browser.get_current_page()
            if page:
                return page
            logger.warning("页面丢失，重试 %d/%d", attempt + 1, retries)

        # 尝试从已有标签页恢复
        pages = await self.browser.get_pages()
        if pages:
            logger.info("从已有标签页恢复 (共 %d 个)", len(pages))
            return pages[-1]

        # 创建新标签页
        try:
            page = await self.browser.new_page("about:blank")
            logger.info("手动创建新标签页恢复成功")
            return page
        except Exception as e:
            raise RuntimeError(f"浏览器页面丢失且无法恢复: {e}")

    async def _goto(self, url: str):
        """导航到 URL，带 target detach 恢复"""
        try:
            page = await self._get_page()
            await page.goto(url)
        except Exception as e:
            err = str(e)
            if "-32000" in err or "detach" in err.lower() or "session" in err.lower():
                logger.warning("导航时 target 丢失，尝试用新标签页恢复: %s", err)
                await asyncio.sleep(2)
                page = await self._get_page()
                await page.goto(url)
            else:
                raise
        await asyncio.sleep(2)

    async def _get_html(self) -> str:
        page = await self._get_page()
        return await page.evaluate("() => document.documentElement.outerHTML")

    async def iterate(
        self, first_html: str, rules: PageRules, base_url: str = "",
        load_more_text: str | None = None,
        next_button_text: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """根据规则遍历所有页面，逐页 yield HTML（流式，不缓存）"""
        max_pages = rules.pagination_max

        # 1. load_more: XPath（AutoScraper ML 动态发现）→ CSS selector → 文本兜底
        load_more_selector = rules.load_more_selector
        load_more_xpath = None

        # 有按钮文本 → 在当前页用 AutoScraper 发现 XPath
        if load_more_text:
            load_more_xpath = self._find_load_more_xpath(first_html, load_more_text)

        if load_more_xpath or load_more_selector:
            await self._try_load_more(selector=load_more_selector, xpath=load_more_xpath,
                                      button_text=load_more_text or "")
            try:
                first_html = await self._get_html()
            except Exception as e:
                logger.warning("load_more 后获取 HTML 失败，使用原始页面: %s", e)

        # 2. sub_pages: 递归遍历子页面
        if rules.sub_page_selector:
            yield first_html
            async for html in self._do_sub_pages(
                selector=rules.sub_page_selector,
                url_attr=rules.sub_page_url_attr,
                url_filter=rules.sub_page_url_filter,
                load_more_selector=load_more_selector,
                load_more_text=load_more_text,
                base_url=base_url,
            ):
                yield html

        # 3. pagination URL 模式
        elif rules.pagination_url:
            yield first_html
            async for html in self._do_pagination_url(rules.pagination_url, max_pages or DEFAULT_MAX_PAGES):
                yield html

        # 4. next_button 翻页: selector 优先，文本兜底
        elif rules.next_button_selector or next_button_text:
            yield first_html
            async for html in self._do_next_button(
                selector=rules.next_button_selector,
                button_text=next_button_text,
                max_pages=max_pages,
            ):
                yield html

        # 5. 无规则: 单页
        else:
            yield first_html

    # ── load_more ────────────────────────────────────────

    async def _try_load_more(self, selector: str | None = None, xpath: str | None = None,
                             button_text: str = "", max_clicks: int = 50):
        """尝试点击 Load more 按钮。通过 JS evaluate 定位并点击。
        优先级: XPath(含button_text过滤) → CSS selector → 文本匹配

        浏览器崩溃时内部 catch，不向上抛异常，保证 iterate() 流程继续。
        """
        click_count = 0
        prev_height = 0
        use_xpath = bool(xpath)
        try:
            while click_count < max_clicks:
                page = await self._get_page()
                clicked = False

                # 1. XPath 定位（AutoScraper 发现）
                if use_xpath:
                    clicked = await self._click_by_xpath(page, xpath, button_text)
                    if not clicked:
                        if click_count > 0:
                            # XPath 之前成功过，按钮消失说明加载完成，直接结束
                            break
                        logger.info("load_more XPath 未命中，降级到 CSS/文本匹配")
                        use_xpath = False

                # 2. CSS selector 定位
                if not clicked and selector:
                    clicked = await self._click_by_selector(page, selector)

                # 3. 用户指定的按钮文本
                if not clicked and button_text:
                    clicked = await self._click_by_text(page, button_text)

                if not clicked:
                    break

                click_count += 1
                if click_count % 5 == 0:
                    logger.info("load_more 已点击 %d 次...", click_count)
                await asyncio.sleep(1.5)

                # 检测页面是否有变化（防止按钮始终可见的死循环）
                cur_height = int(await page.evaluate("() => document.body.scrollHeight"))
                if cur_height == prev_height:
                    logger.info("load_more 页面无变化，停止")
                    break
                prev_height = cur_height
        except Exception as e:
            if click_count > 0:
                logger.warning("load_more 浏览器异常，已点击 %d 次后停止: %s", click_count, e)
            else:
                logger.info("load_more 按钮未找到，跳过")
            return

        if click_count >= max_clicks:
            logger.warning("load_more 达到上限 %d 次，停止", max_clicks)
        elif click_count > 0:
            logger.info("load_more 完成，共点击 %d 次", click_count)

    # ── JS evaluate 点击辅助方法 ──────────────────────────

    @staticmethod
    async def _click_by_xpath(page, xpath: str, text_filter: str = "") -> bool:
        """通过 XPath 定位并点击元素（JS evaluate）。
        AutoScraper 可能匹配到容器元素（li/div/span），而非实际可点击的 button/a，
        因此匹配到非可点击标签时，优先在子元素中查找 button/a/input 点击。
        """
        result = await page.evaluate("""(xpath, textFilter) => {
            const CLICKABLE = new Set(['BUTTON','A','INPUT']);
            const iter = document.evaluate(xpath, document, null,
                XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
            for (let i = 0; i < iter.snapshotLength; i++) {
                const el = iter.snapshotItem(i);
                if (textFilter && !el.textContent.includes(textFilter)) continue;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                // 如果元素本身不是可点击标签，优先找子元素中的 button/a/input
                let target = el;
                if (!CLICKABLE.has(el.tagName)) {
                    const child = el.querySelector('button, a, input[type="button"], input[type="submit"]');
                    if (child) target = child;
                }
                target.scrollIntoView({block: 'center'});
                target.click();
                return 'clicked';
            }
            return '';
        }""", xpath, text_filter)
        return result == "clicked"

    @staticmethod
    async def _click_by_selector(page, selector: str) -> bool:
        """通过 CSS selector 定位并点击第一个可见元素（JS evaluate）"""
        result = await page.evaluate("""(selector) => {
            try {
                const els = document.querySelectorAll(selector);
                for (const el of els) {
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    el.scrollIntoView({block: 'center'});
                    el.click();
                    return 'clicked';
                }
            } catch(e) {}
            return '';
        }""", selector)
        return result == "clicked"

    @staticmethod
    async def _click_by_text(page, text: str) -> bool:
        """通过文本匹配点击按钮或链接（JS evaluate）"""
        result = await page.evaluate("""(text) => {
            const els = document.querySelectorAll(
                'button, a, [role="button"], [role="link"], '
                + 'input[type="button"], input[type="submit"]');
            const lower = text.toLowerCase();
            for (const el of els) {
                if (!el.textContent.trim().toLowerCase().includes(lower)) continue;
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                el.scrollIntoView({block: 'center'});
                el.click();
                return 'clicked';
            }
            return '';
        }""", text)
        return result == "clicked"

    # ── sub_pages (真正递归) ─────────────────────────────

    async def _do_sub_pages(
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

                # 每个子页面动态发现 load_more XPath（按钮在不同页面可能结构不同）
                page_xpath = None
                if load_more_text:
                    html = await self._get_html()
                    page_xpath = self._find_load_more_xpath(html, load_more_text)

                # 每个子页面都尝试 Load more
                await self._try_load_more(selector=load_more_selector, xpath=page_xpath,
                                          button_text=load_more_text or "")

                html = await self._get_html()
                yield html

                # 递归：检查这个子页面里是否还有更深层子页面
                async for deeper_html in self._do_sub_pages(
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
                    yield deeper_html

            except Exception as e:
                err = str(e)
                # 浏览器连接断开是致命错误，不要继续尝试后续子页面
                if "无法恢复" in err or "Failed to open" in err or "-32000" in err:
                    logger.error("子页面 [%d] 浏览器连接丢失，停止遍历: %s", i + 1, e)
                    return
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
        """从当前页面提取子页面链接（JS evaluate）"""
        import json as _json
        page = await self._get_page()
        raw = await page.evaluate("""(selector, urlAttr) => {
            const els = [...document.querySelectorAll(selector)];
            return els.map(e => e.getAttribute(urlAttr)).filter(Boolean);
        }""", selector, url_attr)
        vals = _json.loads(raw) if isinstance(raw, str) and raw.startswith("[") else []

        # 补全相对 URL + 去重
        seen: set[str] = set()
        urls: list[str] = []
        for val in vals:
            full = urljoin(base_url, val) if not val.startswith("http") else val
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

    async def _do_next_button(
        self, selector: str | None = None, button_text: str | None = None,
        max_pages: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """点击"下一页"翻页。优先 CSS selector，失败则按文本匹配兜底。
        max_pages 为总页数限制（含第1页），None 则不限制。
        """
        max_extra = (max_pages - 1) if max_pages else DEFAULT_MAX_PAGES
        logger.info("翻页按钮: selector=%s, text=%s (最多翻 %d 页)", selector, button_text, max_extra)
        page_count = 0
        for i in range(max_extra):
            page = await self._get_page()
            clicked = False

            # 1. CSS selector（RuleDiscoverer 发现）
            if selector:
                clicked = await self._click_by_selector(page, selector)

            # 2. 用户指定的按钮文本
            if not clicked and button_text:
                clicked = await self._click_by_text(page, button_text)

            if not clicked:
                break

            await asyncio.sleep(2)
            yield await self._get_html()
            page_count += 1
            if (i + 1) % 5 == 0:
                logger.info("已翻 %d 页...", i + 1)
        logger.info("翻页完成，共 %d 个额外页面", page_count)

    # ── AutoScraper 发现 load_more XPath ──────────────────

    @staticmethod
    def _find_load_more_xpath(html: str, button_text: str) -> str | None:
        """在当前页面 HTML 上用 AutoScraper 找 load_more 按钮的 XPath"""
        from autoscraper.auto_scraper import AutoScraper

        try:
            scraper = AutoScraper()
            scraper.build(html=html, wanted_dict={"load_more": [button_text]})
            rules = scraper.get_result_xpath_rule()
            if rules:
                xpath = next(iter(rules.values()))
                logger.info("[PageIterator] AutoScraper 发现 load_more XPath: %s", xpath)
                return xpath
            logger.warning("[PageIterator] AutoScraper 未找到 '%s' 的 XPath", button_text)
        except Exception as e:
            logger.warning("[PageIterator] AutoScraper 查找 load_more 失败: %s", e)
        return None
