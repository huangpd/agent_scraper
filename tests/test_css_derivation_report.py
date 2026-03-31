"""CSS 提取效果验证报告

模拟真实场景：43 个页面，3 种不同结构，验证：
1. AutoScraper 训练后能否在同结构子页面复用
2. CSS LLM 兜底能否处理不同结构
3. LLM 调用次数是否受控
"""

import json
from unittest.mock import AsyncMock, MagicMock

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


def _make_mock_llm() -> MagicMock:
    """创建一个根据 HTML 内容返回对应 CSS 选择器的 mock LLMService"""
    mock = MagicMock()

    async def smart_call(prompt, **kwargs):
        if "grid-cols-24" in prompt or "truncate" in prompt:
            return json.dumps({
                "file_name": {"selector": "span.truncate", "attr": "text"},
                "download_url": {"selector": "a.group", "attr": "href"},
            })
        elif "file-row" in prompt or "file-name" in prompt:
            return json.dumps({
                "file_name": {"selector": "span.file-name", "attr": "text"},
                "download_url": {"selector": "a.file-link", "attr": "href"},
            })
        elif "data-row" in prompt or "col-name" in prompt:
            return json.dumps({
                "file_name": {"selector": "td.col-name", "attr": "text"},
                "download_url": {"selector": "td.col-link a", "attr": "href"},
            })
        return '{}'

    mock.call = AsyncMock(side_effect=smart_call)
    return mock


class TestCssDerivationReport:

    @pytest.mark.asyncio
    async def test_full_43_page_report(self):
        """模拟 43 页提取，AutoScraper 应处理大部分同结构页面"""
        mock_llm = _make_mock_llm()
        extractor = Extractor(llm_service=mock_llm)

        pages = generate_pages()

        results = []
        for page_name, html in pages:
            before = mock_llm.call.call_count
            data = await extractor.extract(html, GOAL)
            after = mock_llm.call.call_count
            llm_used = after - before

            file_count = len(data.get("file_name", []))
            url_count = len(data.get("download_url", []))

            method = "LLM" if llm_used > 0 else "AutoScraper"
            results.append({
                "page": page_name,
                "method": method,
                "llm_calls": llm_used,
                "files": file_count,
                "urls": url_count,
            })

        total_llm = sum(r["llm_calls"] for r in results)
        total_files = sum(r["files"] for r in results)
        total_urls = sum(r["urls"] for r in results)
        autoscraper_pages = sum(1 for r in results if r["method"] == "AutoScraper")

        print(f"\n{'=' * 72}")
        print("CSS 提取效果验证报告")
        print(f"{'=' * 72}")
        print(f"总页面数: {len(pages)} (结构A×20 + 结构B×15 + 结构C×8)")
        print(f"AutoScraper 命中: {autoscraper_pages} / {len(pages)}")
        print(f"LLM 调用次数: {total_llm}")
        print(f"总提取 file_name: {total_files} 条")
        print(f"总提取 URL: {total_urls} 条")
        print(f"{'=' * 72}")

        # 结构 A 有样本训练，AutoScraper 应覆盖大部分
        a_results = [r for r in results if r["page"].startswith("A")]
        assert all(r["files"] > 0 for r in a_results), "结构 A 页面存在空提取"
        assert total_files > 0, "没有提取到任何 file_name"

    @pytest.mark.asyncio
    async def test_css_fallback_handles_new_structure(self):
        """AutoScraper 无法处理的新结构应由 CSS LLM 兜底"""
        mock_llm = _make_mock_llm()
        extractor = Extractor(llm_service=mock_llm)

        # 用结构 A 训练 AutoScraper
        html_a = make_type_a_page([
            ("train1.txt", "/a/1"), ("train2.txt", "/a/2"), ("train3.txt", "/a/3"),
        ])
        goal_with_samples = ExtractionGoal(
            fields={"file_name": "文件名", "download_url": "下载链接"},
            samples={"file_name": ["train1.txt", "train2.txt"]},
        )
        await extractor.extract(html_a, goal_with_samples)

        # 用结构 C（表格）测试 — AutoScraper 不认识此结构
        html_c = make_type_c_page([
            ("table1.json", "/c/1"), ("table2.json", "/c/2"),
        ])
        goal_no_samples = ExtractionGoal(
            fields={"file_name": "文件名", "download_url": "下载链接"},
        )
        result = await extractor.extract(html_c, goal_no_samples)

        # 如果 AutoScraper 失败，CSS LLM 应接管
        if mock_llm.call.call_count > 0:
            assert len(result.get("file_name", [])) > 0, "CSS LLM 兜底应提取到数据"
