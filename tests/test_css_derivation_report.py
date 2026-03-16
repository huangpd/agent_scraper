"""CSS 反推效果验证报告

模拟真实场景：43 个页面，3 种不同结构，验证：
1. 反推的 CSS 规则能否在同结构子页面复用
2. 多规则缓存能否覆盖多种结构
3. LLM 调用次数是否显著减少
"""

from unittest.mock import AsyncMock, patch

import pytest
from agent_scraper.core.models import ExtractionGoal
from agent_scraper.extraction.extractor import Extractor


# ── 3 种页面结构模板 ───────────────────────────────────

def make_type_a_page(files: list[tuple[str, str]]) -> str:
    """结构 A：主列表页（类似 OpenNeuro 文件列表）"""
    items = "\n".join(
        f'''  <li class="grid-cols-24">
    <a class="group" href="{href}" download>
      <span class="truncate">{name}</span>
    </a>
  </li>'''
        for name, href in files
    )
    return f"<html><body><ul class='file-list'>{items}</ul></body></html>"


def make_type_b_page(files: list[tuple[str, str]]) -> str:
    """结构 B：子文件夹页（不同的 CSS 类和层级）"""
    items = "\n".join(
        f'''  <div class="file-row">
    <a class="file-link" href="{href}">
      <span class="file-name">{name}</span>
    </a>
  </div>'''
        for name, href in files
    )
    return f"<html><body><div class='folder-content'>{items}</div></body></html>"


def make_type_c_page(files: list[tuple[str, str]]) -> str:
    """结构 C：表格式页面"""
    rows = "\n".join(
        f'''  <tr class="data-row">
    <td class="col-name">{name}</td>
    <td class="col-link"><a href="{href}">Download</a></td>
  </tr>'''
        for name, href in files
    )
    return f"<html><body><table><tbody>{rows}</tbody></table></body></html>"


# ── 生成 43 页测试数据 ─────────────────────────────────

def generate_pages() -> list[tuple[str, str]]:
    """生成 43 页：20 页结构A + 15 页结构B + 8 页结构C"""
    pages = []
    for i in range(20):
        html = make_type_a_page([
            (f"file_a{i}_{j}.txt", f"/data/a{i}/file_{j}.txt")
            for j in range(5)
        ])
        pages.append((f"A-{i+1}", html))

    for i in range(15):
        html = make_type_b_page([
            (f"file_b{i}_{j}.csv", f"/sub/b{i}/file_{j}.csv")
            for j in range(4)
        ])
        pages.append((f"B-{i+1}", html))

    for i in range(8):
        html = make_type_c_page([
            (f"file_c{i}_{j}.json", f"/table/c{i}/file_{j}.json")
            for j in range(6)
        ])
        pages.append((f"C-{i+1}", html))

    return pages


GOAL = ExtractionGoal(
    fields={"file_name": "文件名", "download_url": "下载链接"},
    samples={
        "file_name": ["file_a0_0.txt", "file_a0_1.txt", "file_a0_2.txt"],
        "download_url": ["/data/a0/file_0.txt", "/data/a0/file_1.txt"],
    },
)


class TestCssDerivationReport:

    @pytest.mark.asyncio
    async def test_full_43_page_report(self):
        """模拟 43 页提取，生成详细报告"""
        llm_call_count = 0
        original_llm_call = None

        async def mock_llm_call(self_inner, prompt):
            nonlocal llm_call_count
            llm_call_count += 1
            # 根据 HTML 内容返回不同的 CSS 选择器
            if "grid-cols-24" in prompt:
                return '{"file_name": {"selector": "span.truncate", "attr": "text"}, "download_url": {"selector": "a.group", "attr": "href"}}'
            elif "file-row" in prompt:
                return '{"file_name": {"selector": "span.file-name", "attr": "text"}, "download_url": {"selector": "a.file-link", "attr": "href"}}'
            elif "data-row" in prompt:
                return '{"file_name": {"selector": "td.col-name", "attr": "text"}, "download_url": {"selector": "td.col-link a", "attr": "href"}}'
            return '{}'

        with patch("agent_scraper.extraction.extractor.create_openai_client"), \
             patch("agent_scraper.extraction.extractor.get_model_name", return_value="test"):
            extractor = Extractor()
            extractor._llm_call = lambda prompt: mock_llm_call(extractor, prompt)

        pages = generate_pages()

        report_lines = []
        report_lines.append("=" * 72)
        report_lines.append("CSS 反推效果验证报告")
        report_lines.append("=" * 72)
        report_lines.append(f"总页面数: {len(pages)} (结构A×20 + 结构B×15 + 结构C×8)")
        report_lines.append(f"提取字段: file_name, download_url")
        report_lines.append("-" * 72)

        results = []
        for page_name, html in pages:
            before = llm_call_count
            data = await extractor.extract(html, GOAL)
            after = llm_call_count
            llm_used = after - before

            file_count = len(data.get("file_name", []))
            url_count = len(data.get("download_url", []))
            cache_count = len(extractor._css_rule_cache)

            method = "LLM" if llm_used > 0 else "Cache"
            if extractor._trained_scraper and method == "Cache" and cache_count == 0:
                method = "AutoScraper"

            results.append({
                "page": page_name,
                "method": method,
                "llm_calls": llm_used,
                "files": file_count,
                "urls": url_count,
                "rules_cached": cache_count,
            })

            if llm_used > 0 or page_name.endswith("-1"):
                report_lines.append(
                    f"  {page_name:8s} | {method:12s} | LLM={llm_used} | "
                    f"files={file_count} urls={url_count} | 缓存规则={cache_count}套"
                )

        # ── 汇总 ──
        total_llm = sum(r["llm_calls"] for r in results)
        cache_hits = sum(1 for r in results if r["llm_calls"] == 0)
        total_files = sum(r["files"] for r in results)
        total_urls = sum(r["urls"] for r in results)

        report_lines.append("-" * 72)
        report_lines.append("汇总:")
        report_lines.append(f"  总 LLM 调用次数:  {total_llm}")
        report_lines.append(f"  缓存命中页面数:   {cache_hits} / {len(pages)}")
        report_lines.append(f"  缓存命中率:       {cache_hits/len(pages)*100:.1f}%")
        report_lines.append(f"  累计规则数:       {len(extractor._css_rule_cache)} 套")
        report_lines.append(f"  总提取 file_name: {total_files} 条")
        report_lines.append(f"  总提取 URL:       {total_urls} 条")
        report_lines.append(f"  空提取页面:       {sum(1 for r in results if r['files']==0)}")

        # 对比：没有反推+多规则缓存时的理论 LLM 次数
        # 最差情况：每种新结构都调 LLM（旧单规则缓存会覆盖）
        # 结构A第1页 + 结构B每页都可能调（因为A的缓存被B覆盖）
        naive_llm = 3 + 14 + 7  # 首次3种 + B覆盖A后A再来 + C覆盖B后B再来
        report_lines.append("-" * 72)
        report_lines.append(f"  优化前预估 LLM:   ~{naive_llm}+ 次（单规则缓存互相覆盖）")
        report_lines.append(f"  优化后实际 LLM:   {total_llm} 次")
        report_lines.append(f"  节省:             {max(0, naive_llm - total_llm)} 次 LLM 调用")
        report_lines.append("=" * 72)

        report = "\n".join(report_lines)
        print("\n" + report)

        # ── 断言 ──
        # 核心：首页可能需要 1 次 LLM（AutoScraper 训练不完整时），子页面不应调 LLM
        assert total_llm <= 1, f"LLM 调用过多: {total_llm}，期望 ≤ 1（仅首页）"
        assert total_files > 0, "没有提取到任何 file_name"
        assert total_urls > 0, "没有提取到任何 download_url"
        # 结构 A 页面（20页）应全部提取到数据
        a_results = [r for r in results if r["page"].startswith("A")]
        assert all(r["files"] > 0 for r in a_results), "结构 A 页面存在空提取"

    @pytest.mark.asyncio
    async def test_new_vs_old_architecture(self):
        """对比：新架构(反推+多规则) vs 旧架构(单规则覆盖)

        旧架构问题：_cached_css_selectors 是单个 dict，每次新生成就覆盖旧的。
        A→B→A 的页面序列中，A 的缓存被 B 覆盖，回到 A 时又要调 LLM。
        """
        # 交替页面序列：A B A B A C A B C A ...（模拟真实子页面混合出现）
        pages = []
        for i in range(15):
            pages.append(("A", make_type_a_page([
                (f"a{i}_{j}.txt", f"/a/{j}") for j in range(3)
            ])))
            pages.append(("B", make_type_b_page([
                (f"b{i}_{j}.csv", f"/b/{j}") for j in range(3)
            ])))
        for i in range(5):
            pages.append(("C", make_type_c_page([
                (f"c{i}_{j}.json", f"/c/{j}") for j in range(3)
            ])))
            pages.append(("A", make_type_a_page([
                (f"a2{i}_{j}.txt", f"/a2/{j}") for j in range(3)
            ])))

        async def mock_llm(prompt):
            if "grid-cols-24" in prompt:
                return '{"file_name": {"selector": "span.truncate", "attr": "text"}, "download_url": {"selector": "a.group", "attr": "href"}}'
            elif "file-row" in prompt:
                return '{"file_name": {"selector": "span.file-name", "attr": "text"}, "download_url": {"selector": "a.file-link", "attr": "href"}}'
            elif "data-row" in prompt:
                return '{"file_name": {"selector": "td.col-name", "attr": "text"}, "download_url": {"selector": "td.col-link a", "attr": "href"}}'
            return '{}'

        # ── 新架构：反推 + 多规则缓存 ──
        new_llm_calls = 0

        async def new_mock(prompt):
            nonlocal new_llm_calls
            new_llm_calls += 1
            return await mock_llm(prompt)

        with patch("agent_scraper.extraction.extractor.create_openai_client"), \
             patch("agent_scraper.extraction.extractor.get_model_name", return_value="test"):
            ext_new = Extractor()
            ext_new._llm_call = new_mock
            for _, html in pages:
                await ext_new.extract(html, GOAL)

        # ── 旧架构模拟：单规则覆盖，无反推 ──
        old_llm_calls = 0

        async def old_mock(prompt):
            nonlocal old_llm_calls
            old_llm_calls += 1
            return await mock_llm(prompt)

        with patch("agent_scraper.extraction.extractor.create_openai_client"), \
             patch("agent_scraper.extraction.extractor.get_model_name", return_value="test"):
            ext_old = Extractor()
            ext_old._llm_call = old_mock
            # 禁用反推
            ext_old._derive_css_from_extraction = lambda *a, **kw: None

            # 模拟旧的单规则覆盖行为
            original_css_extract = ext_old._css_selector_extract

            async def old_css_extract(html, goal, expected_fields):
                result = await original_css_extract(html, goal, expected_fields)
                # 旧行为：只保留最后一个规则（覆盖）
                if len(ext_old._css_rule_cache) > 1:
                    ext_old._css_rule_cache[:] = ext_old._css_rule_cache[-1:]
                return result

            ext_old._css_selector_extract = old_css_extract

            for _, html in pages:
                await ext_old.extract(html, GOAL)

        print(f"\n{'='*56}")
        print(f"架构对比报告 ({len(pages)} 页, A/B/C 交替)")
        print(f"{'='*56}")
        print(f"  新架构 (反推+多规则):  {new_llm_calls:3d} 次 LLM 调用")
        print(f"  旧架构 (单规则覆盖):   {old_llm_calls:3d} 次 LLM 调用")
        saved = old_llm_calls - new_llm_calls
        pct = saved / max(old_llm_calls, 1) * 100
        print(f"  节省:                  {saved:3d} 次 ({pct:.0f}%)")
        print(f"{'='*56}")

        assert new_llm_calls < old_llm_calls, (
            f"新架构应减少 LLM 调用: new={new_llm_calls}, old={old_llm_calls}"
        )
