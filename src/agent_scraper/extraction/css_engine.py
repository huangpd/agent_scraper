"""共享 CSS 选择器引擎：HTML 预处理 + 提示词 + 验证

供 RuleDiscoverer（翻页规则）和 Extractor（数据字段）共用。
"""

import json
import logging
import re

from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

# ── HTML 预处理 ──────────────────────────────────────────────

_STRIP_TAGS = {
    "script", "style", "noscript", "svg", "path",
    "meta", "link", "head", "iframe", "canvas",
}

_NOISE_CLASS_RE = re.compile(
    r'\b(?:'
    r'(?:sm|md|lg|xl|2xl|dark|hover|focus|active|group|peer):'
    r'|(?:flex|grid|block|hidden|relative|absolute|fixed|sticky|contents)\b'
    r'|(?:items|justify|content|self|place)[-\w]+'
    r'|(?:w|h|min|max|size)[-\w]+'
    r'|(?:p|m|px|py|pt|pb|pl|pr|mx|my|mt|mb|ml|mr)[-\w]+'
    r'|(?:gap|space)[-\w]+'
    r'|(?:text|font|leading|tracking)[-\w]+'
    r'|(?:bg|border|ring|shadow|outline|divide)[-\w]+'
    r'|(?:rounded|overflow|truncate|transition|duration|ease|animate)[-\w]*'
    r'|(?:col|row)-(?:span|start|end)[-\w]+'
    r'|(?:z|opacity|cursor|transform|scale|rotate|translate)[-\w]+'
    r'|(?:clearfix|cf|wrapper|inner|outer|pull-(?:left|right))\b'
    r')',
    re.VERBOSE,
)

_KEEP_ATTRS = (
    "id", "class", "href", "src", "role",
    "data-testid", "data-cy", "data-test",
    "data-id", "data-key", "data-type",
    "data-name", "data-target", "data-component",
    "aria-label", "aria-labelledby", "type", "name",
)


def preprocess_html(html: str, max_chars: int = 15000) -> str:
    """HTML 预处理：去噪 + 裁剪 Tailwind + 压缩重复列表 + 截断。

    max_chars > 0 (默认): LLM 场景 — 裁剪属性/Tailwind/文本/压缩列表/截断
    max_chars = 0:        AutoScraper 场景 — 只做去噪标签+删注释+取主内容，保留完整数据
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. 删除无用标签（始终执行）
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # 2. 删除注释（始终执行）
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 3. LLM 专用优化：裁剪属性/Tailwind class/文本截断
    if max_chars:
        for tag in soup.find_all(True):
            new_attrs = {}
            for attr in _KEEP_ATTRS:
                val = tag.attrs.get(attr)
                if val is None:
                    continue
                if attr == "class":
                    if isinstance(val, list):
                        val = " ".join(val)
                    cleaned = _NOISE_CLASS_RE.sub("", val).split()
                    cleaned = list(dict.fromkeys(c for c in cleaned if len(c) >= 3))
                    if cleaned:
                        new_attrs["class"] = " ".join(cleaned)
                else:
                    if isinstance(val, list):
                        val = val[0]
                    if val:
                        new_attrs[attr] = val.strip()
            tag.attrs = new_attrs

            # 截断过长文本节点
            for child in tag.children:
                if isinstance(child, str) and len(child.strip()) > 80:
                    child.replace_with(child.strip()[:80] + "…")

    # 4. 优先取 main 区域
    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main|container", re.I))
        or soup.body
        or soup
    )

    # 5. 压缩重复列表项 + 截断（仅 LLM 场景）
    if max_chars:
        _compress_repeated_children(main, soup)

    result = str(main)
    if max_chars and len(result) > max_chars:
        result = result[:max_chars] + "\n<!-- [HTML 已截断] -->"

    logger.info("HTML 预处理: %.0fKB → %.0fKB", len(html) / 1024, len(result) / 1024)
    return result


def _compress_repeated_children(root, soup):
    """压缩重复列表项：>5 个同类子元素只保留前3+末1，中间用注释占位。"""
    for container in root.find_all(["ul", "ol", "tbody", "div"]):
        children = [c for c in container.children if hasattr(c, "name") and c.name]
        if len(children) > 5:
            tag_names = [c.name for c in children]
            most_common = max(set(tag_names), key=tag_names.count)
            same_tag = [c for c in children if c.name == most_common]
            if len(same_tag) > 5:
                keep_head = same_tag[:3]
                keep_tail = [same_tag[-1]]
                removed = len(same_tag) - 4
                for item in same_tag:
                    if item not in keep_head and item not in keep_tail:
                        item.decompose()
                placeholder = soup.new_string(f"\n<!-- ... 省略 {removed} 个同类元素 ... -->\n")
                keep_head[-1].insert_after(placeholder)


# ── 选择器验证 ───────────────────────────────────────────────

def validate_css(html: str, selector: str, attr: str = "text") -> dict:
    """验证单个 CSS 选择器，返回 {valid, count, samples}。"""
    try:
        soup = BeautifulSoup(html, "lxml")
        nodes = soup.select(selector)
        samples = []
        for node in nodes[:3]:
            if attr == "text":
                val = node.get_text(strip=True)
            elif attr == "innerHTML":
                val = node.decode_contents()
            else:
                val = node.get(attr, "")
            if val:
                samples.append(str(val)[:100])
        return {"valid": len(nodes) > 0, "count": len(nodes), "samples": samples}
    except Exception:
        return {"valid": False, "count": 0, "samples": []}


def validate_selectors_batch(html: str, selectors: dict) -> list[str]:
    """批量验证，返回无效的字段名列表。

    selectors 格式: {field_name: {"css": "...", "attr": "..."}}
    """
    invalid = []
    for field, info in selectors.items():
        css = info if isinstance(info, str) else info.get("css", info.get("selector", ""))
        attr = "text" if isinstance(info, str) else info.get("attr", "text")
        result = validate_css(html, css, attr)
        if not result["valid"]:
            invalid.append(field)
    return invalid


# ── JSON 解析 ────────────────────────────────────────────────

def parse_json_response(text: str) -> dict:
    """从 LLM 响应中提取 JSON（容错：去 markdown 包裹）。"""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"LLM 未返回合法 JSON：{text[:200]}")
    return json.loads(match.group())


# ── System Prompt（集中管理在 prompts.py）──────────────────────
# 向后兼容：从 prompts 重新导出，已有 import 不需要改动
from agent_scraper.pipeline.prompts import CSS_SYSTEM_PROMPT  # noqa: F401
