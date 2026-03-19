"""
AutoScraper - 基于样本数据自动推导 XPath 规则的网页抓取器

流程：
  1. build()  — 在训练页面上根据样本数据学习 DOM 路径（stack）
     · 规则模式：精确文本/属性匹配
     · ML 模式（fallback）：文本匹配失败时，用随机森林定位目标节点
  2. _stack_to_xpath()  — 将学到的 stack 反推为精准 XPath 表达式
  3. get_result_similar() — 在任意页面上应用 XPath 规则提取数据
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from html import unescape
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

from autoscraper.utils import (
    FuzzyText,
    ResultItem,
    get_non_rec_text,
    normalize,
    text_match,
    unique_hashable,
    unique_stack_list,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# 哈希 class 检测
# ─────────────────────────────────────────────────────────
_HASHED_PATTERNS = [
    re.compile(r'-[A-Za-z0-9]{4,8}$'),
    re.compile(r'__[A-Za-z0-9]{4,}$'),
    re.compile(r'^[a-z]-[a-f0-9]{6,}'),
    re.compile(r'[A-Z]{2,}[0-9][A-Za-z0-9]{2,}'),
]

def _is_hashed_class(cls: str) -> bool:
    for pattern in _HASHED_PATTERNS:
        if pattern.search(cls):
            suffix = cls.split('-')[-1]
            if any(c.isdigit() for c in suffix) and any(c.isalpha() for c in suffix):
                return True
    return False

def _stable_classes(classes) -> list:
    if not classes: return []
    if isinstance(classes, str): classes = classes.split()
    return [c for c in classes if not _is_hashed_class(c) and not _is_tailwind_class(c)]

# ─────────────────────────────────────────────────────────
# Tailwind CSS 原子 class 检测
# ─────────────────────────────────────────────────────────
# 这些单词即使是 Tailwind 内置类也保留，因为它们有足够的语义锚定价值
_TAILWIND_BARE_EXCEPTIONS = frozenset({
    "container", "group", "peer", "contents", "border",
    "prose", "sr-only", "not-sr-only",
})

_TAILWIND_PATTERNS = [
    re.compile(r'^(hover|focus|focus-within|focus-visible|active|visited|disabled'
               r'|checked|indeterminate|placeholder|before|after|first-line|first-letter'
               r'|marker|selection|dark|rtl|ltr'
               r'|sm|md|lg|xl|2xl|3xl):'),           # 响应式 / 伪类 / 暗色前缀
    re.compile(r'^(flex|grid|block|inline-block|inline-flex|inline-grid'
               r'|inline|hidden|table|flow-root|contents)$'),  # 纯显示原语
    re.compile(r'^(p|m|px|py|mx|my|pt|pb|pl|pr|mt|mb|ms|me|gap|space)-'),  # 间距
    re.compile(r'^(w|h|min-w|min-h|max-w|max-h|size)-'),                    # 尺寸
    re.compile(r'^(text|font|leading|tracking|whitespace|break|truncate'
               r'|overflow-ellipsis|line-clamp)-'),             # 排版
    re.compile(r'^(bg|from|to|via|gradient)-'),                 # 背景 / 渐变
    re.compile(r'^(border|outline|ring|divide)-'),              # 边框（带修饰符）
    re.compile(r'^(rounded|shadow|opacity|mix-blend|bg-blend)-'),  # 视觉
    re.compile(r'^(flex|grid|col|row|order|place|items|justify|self|content)-'),  # 布局
    re.compile(r'^(overflow|overscroll|scroll|snap)-'),         # 滚动
    re.compile(r'^(cursor|pointer|select|resize|appearance)-'), # 交互
    re.compile(r'^(transition|duration|ease|delay|animate)-'),  # 动画
    re.compile(r'^(z|top|right|bottom|left|inset)-'),           # 定位
    re.compile(r'^(static|fixed|absolute|relative|sticky)$'),  # position 原语
    re.compile(r'^(capitalize|uppercase|lowercase|normal-case)$'),
    re.compile(r'^(italic|not-italic|underline|line-through|no-underline)$'),
    re.compile(r'^(visible|invisible|collapse)$'),
    re.compile(r'^(aspect|columns|basis|grow|shrink)-'),
    re.compile(r'^(object|origin|accent|caret|fill|stroke)-'),
    re.compile(r'^(list|table|caption|border-collapse|border-separate)$'),
    re.compile(r'^(float|clear)-'),
    re.compile(r'-(xs|sm|md|lg|xl|2xl|3xl|4xl|full|screen|auto|none|px)$'),  # 常见尺寸后缀
    re.compile(r'-\d+$'),          # 以数字结尾: p-4, mt-2, w-64 等
    re.compile(r'-\[.+\]$'),       # 任意值: w-[200px], text-[#fff]
]

def _is_tailwind_class(cls: str) -> bool:
    """判断一个 class 是否是 Tailwind 原子功能类（不适合作为 XPath 锚点）。"""
    if cls in _TAILWIND_BARE_EXCEPTIONS:
        return False
    return any(p.search(cls) for p in _TAILWIND_PATTERNS)

# ─────────────────────────────────────────────────────────
# Variant class 检测（位置/状态类，不应出现在 XPath 谓词中）
# ─────────────────────────────────────────────────────────
_VARIANT_EXACT = frozenset({
    "first", "last", "odd", "even",
    "active", "current", "selected",
    "open", "closed", "disabled", "hidden", "visible",
})
_VARIANT_PATTERNS = [
    re.compile(r'-(first|last|odd|even|active|current|selected)$', re.I),
    re.compile(r'^(first|last|odd|even|active)-', re.I),
    re.compile(r'^is-'),   # is-active, is-open, etc.
]

def _is_variant_class(cls: str) -> bool:
    if cls.lower() in _VARIANT_EXACT:
        return True
    return any(p.search(cls) for p in _VARIANT_PATTERNS)

# ─────────────────────────────────────────────────────────
# 动态 ID 检测
# ─────────────────────────────────────────────────────────
_DYNAMIC_ID_PATTERNS = [
    re.compile(r'\d{5,}'),                 # 长数字序列: id="el12345"
    re.compile(r'^[a-z]-[a-f0-9]{6,}'),    # CSS-module 风格: id="a-3f2b1c"
    re.compile(r'^[a-f0-9]{8,}$'),          # 纯十六进制哈希
    re.compile(r'[-_][a-f0-9]{6,}$'),       # 尾部哈希: id="item-a3b2c1"
    re.compile(r'^:'),                       # React/框架生成: id=":r1:"
    re.compile(r'^(ember|react|vue|ng)-'),   # 框架前缀
]

def _is_stable_id(id_str: str) -> bool:
    """判断 HTML id 属性是否稳定（非动态生成）。"""
    if not id_str:
        return False
    for pattern in _DYNAMIC_ID_PATTERNS:
        if pattern.search(id_str):
            return False
    return True

# ─────────────────────────────────────────────────────────
# ML 特征提取
# ─────────────────────────────────────────────────────────
_SEMANTIC_TAGS = {'h1','h2','h3','h4','h5','h6','p','a','span','li','td','button'}
_SEMANTIC_CLASSES = ['title','description','name','price','date','link','item']
_NUMERIC_KEYS = ['depth','sibling_count','sibling_index','child_count']
_CAT_KEYS = ['tag', 'id_prefix', 'ancestor_0_tag']

def _extract_node_features(node, soup) -> dict:
    f = {'tag': node.name or '', 'depth': len(list(node.parents))}
    parent = node.parent
    if parent:
        siblings = [s for s in parent.children if hasattr(s, 'name') and s.name == node.name]
        f['sibling_count'] = len(siblings)
        f['sibling_index'] = siblings.index(node) if node in siblings else 0
    else:
        f['sibling_count'] = f['sibling_index'] = 0
    f['id_prefix'] = re.sub(r'-?\d+$', '', node.attrs.get('id', ''))
    f['child_count'] = len(list(node.children))
    ancestors = list(node.parents)
    f['ancestor_0_tag'] = ancestors[0].name if ancestors else ''
    return f

def _features_to_vector(features: dict, vocab: dict = None):
    numeric_vec = [float(features.get(k, 0)) for k in _NUMERIC_KEYS]
    building = vocab is None
    if building: vocab = {k: {} for k in _CAT_KEYS}
    cat_vec = []
    for k in _CAT_KEYS:
        val = str(features.get(k, ''))
        if building and val not in vocab[k]: vocab[k][val] = len(vocab[k])
        cat_vec.append(float(vocab[k].get(val, len(vocab[k]))))
    return np.array(numeric_vec + cat_vec, dtype=np.float32), vocab

# ─────────────────────────────────────────────────────────
# AutoScraper 类
# ─────────────────────────────────────────────────────────

class AutoScraper(object):
    request_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    def __init__(self, stack_list=None):
        self.stack_list = stack_list or []

    # ── 持久化 ──────────────────────────────────────────

    def save(self, file_path):
        """保存学习结果到 JSON 文件（stacks + XPath 规则）。"""
        data = {
            "stack_list": self.stack_list,
            "xpath_rules": self.get_result_xpath_rule(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path):
        """从 JSON 文件加载学习结果。"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            self.stack_list = data
        else:
            self.stack_list = data.get("stack_list", [])

    @classmethod
    def _fetch_html(cls, url, request_args=None):
        res = requests.get(url, headers=cls.request_headers, **(request_args or {}))
        res.encoding = res.apparent_encoding
        return res.text

    @classmethod
    def _get_soup(cls, url=None, html=None, request_args=None):
        content = html if html else cls._fetch_html(url, request_args)
        return BeautifulSoup(normalize(unescape(content)), "lxml")

    @staticmethod
    def _get_valid_attrs(item):
        key_attrs = {"class", "id"}
        attrs = {k: v for k, v in item.attrs.items() if k in key_attrs}
        if "class" in attrs and isinstance(attrs["class"], list):
            attrs["class"] = _stable_classes(attrs["class"])
        return attrs

    def build(self, url=None, wanted_list=None, wanted_dict=None, html=None,
              request_args=None, update=False, text_fuzz_ratio=1.0, use_ml=True):
        """在训练页面上根据样本数据学习 DOM 路径，反推 XPath 规则。

        流程：
        1. 规则模式：精确文本/属性匹配 → 构建 stack
        2. ML fallback（use_ml=True 且 sklearn 可用）：规则模式为空时，
           用随机森林分类器定位目标节点 → 构建 stack
        3. 所有 stack 统一经 _stack_to_xpath 转为 XPath 规则

        Returns:
            dict | None: {alias: xpath_string} 或 None（无匹配）
        """
        soup = self._get_soup(url=url, html=html, request_args=request_args)
        if not update:
            self.stack_list = []

        _wdict = wanted_dict or {"default": wanted_list or []}

        # Step 1: 规则模式 — 精确文本/属性匹配
        for alias, targets in _wdict.items():
            for target in targets:
                target = normalize(target)
                for child in reversed(soup.find_all(True)):
                    if self._child_matches(child, target, url, text_fuzz_ratio):
                        stack = self._build_stack(child, url)
                        stack["alias"] = alias
                        self.stack_list.append(stack)

        self.stack_list = unique_stack_list(self.stack_list)

        # Step 2: ML fallback — 规则模式未找到结果时启用
        if not self.stack_list and use_ml and _ML_AVAILABLE:
            logger.info("规则模式未找到结果，切换到 ML 模式...")
            self._ml_build_stacks(soup, url, _wdict, text_fuzz_ratio)
            self.stack_list = unique_stack_list(self.stack_list)

        # 统一返回 XPath 规则
        rules = self.get_result_xpath_rule()
        if rules:
            for alias, xpath in rules.items():
                logger.debug("最终 XPath  alias='%s': %s", alias, xpath)
        return rules

    def _child_matches(self, child, text, url, fuzz):
        """检查 child 是否匹配目标文本。

        匹配顺序：
        1. 完整文本（getText）
        2. 直接文本（不含子节点递归文本）
        3. 属性值精确匹配
        4. href/src 属性的完整 URL 匹配（urljoin 拼接后比对）
        """
        # 1) 完整文本
        child_text = normalize(child.get_text(strip=True))
        if text_match(text, child_text, fuzz):
            # 去重：如果父节点文本完全相同，跳过（保留最内层）
            parent_text = normalize(child.parent.get_text(strip=True)) if child.parent else ""
            if child_text == parent_text and child.parent and child.parent.parent:
                return False
            child.wanted_attr = None
            return True

        # 2) 直接文本（不含子节点）
        non_rec = normalize(get_non_rec_text(child))
        if non_rec and text_match(text, non_rec, fuzz):
            child.wanted_attr = None
            child.is_non_rec_text = True
            return True

        # 3) 属性值匹配 + URL 匹配
        for k, v in child.attrs.items():
            if not isinstance(v, str):
                continue
            v = normalize(v.strip())
            if text_match(text, v, fuzz):
                child.wanted_attr = k
                return True
            # URL 宽松匹配：完整 URL 样本 vs 相对路径属性值
            if k in ("href", "src") and url:
                full_url = urljoin(url, v)
                if text_match(text, full_url, fuzz):
                    child.wanted_attr = k
                    return True
        return False

    @classmethod
    def _build_stack(cls, child, url):
        content = [(child.name, cls._get_valid_attrs(child))]
        parent = child
        while True:
            gp = parent.find_parent()
            if not gp or gp.name == "[document]": break
            # 构造搜索属性，移除空 class 避免 BeautifulSoup 匹配异常
            # （空列表会匹配"没有 class 的元素"而非"所有同 tag 元素"）
            search_attrs = cls._get_valid_attrs(parent)
            if "class" in search_attrs and not search_attrs["class"]:
                search_attrs.pop("class")
            siblings = gp.find_all(parent.name, search_attrs, recursive=False)
            # tag-only 兄弟计数（更宽松，反映真实重复度）
            tag_only_siblings = gp.find_all(parent.name, recursive=False)
            tag_only_count = len(tag_only_siblings)
            for i, c in enumerate(siblings):
                if c == parent:
                    # 5-tuple: (tag, attrs, child_index, child_sibling_count, tag_only_count)
                    # child_index: content[k+1] 在同类兄弟中的位置
                    # child_sibling_count: class-filtered 同类兄弟总数
                    # tag_only_count: 仅按 tag 搜索的兄弟总数（用于判断是否为重复元素）
                    content.insert(0, (gp.name, cls._get_valid_attrs(gp), i, len(siblings), tag_only_count))
                    break
            else:
                # 未在过滤后的兄弟中找到自身，记录为无索引层级（避免丢层）
                content.insert(0, (gp.name, cls._get_valid_attrs(gp)))
            parent = gp

        wanted_attr = getattr(child, "wanted_attr", None)
        # hash 必须包含 wanted_attr，否则同路径不同提取目标的 stack 会被去重
        hash_input = str((content, wanted_attr)).encode()
        stack = {
            "content": content,
            "wanted_attr": wanted_attr,
            "hash": hashlib.sha256(hash_input).hexdigest(),
            "stack_id": hashlib.sha256(hash_input).hexdigest()[:8],
        }
        return stack

    # ── ML 辅助：节点签名 + 兄弟扩展 + ML → stack 桥接 ──

    @staticmethod
    def _node_signature(node) -> str:
        """节点结构签名：tag + 稳定 class + 父 tag，用于判断同类兄弟。"""
        stable = tuple(sorted(_stable_classes(node.attrs.get('class', []))))
        parent_tag = node.parent.name if node.parent else ''
        return f"{node.name}|{stable}|{parent_tag}"

    def _expand_to_siblings(self, seed_nodes, all_nodes) -> set:
        """从种子节点出发，找出所有结构相同的兄弟节点索引。

        两层策略：
        1. 严格模式：同一父节点下 tag+稳定class 相同（直接兄弟）
        2. 宽松模式：全页面 tag+稳定class+父tag 相同（列表项重复模式）
           额外要求：祖父节点签名也相同，防止误匹配导航栏等无关区域
        """
        expanded = set()
        for seed in seed_nodes:
            sig = self._node_signature(seed)
            parent = seed.parent
            if parent is None:
                continue

            # 策略1：同一父节点下的直接兄弟
            same_parent = {
                i for i, node in enumerate(all_nodes)
                if node.parent is parent and self._node_signature(node) == sig
            }

            if len(same_parent) > 1:
                expanded |= same_parent
            else:
                # 策略2：列表模式——父节点本身可重复（如 <li>）
                parent_sig = self._node_signature(parent)
                for i, node in enumerate(all_nodes):
                    if (self._node_signature(node) == sig
                            and node.parent is not None
                            and self._node_signature(node.parent) == parent_sig):
                        expanded.add(i)

        return expanded

    def _ml_build_stacks(self, soup, url, wanted_dict, fuzz_ratio):
        """ML 模式：用随机森林定位目标节点，然后构建 stack 反推 XPath。

        流程：
        1. 文本/属性匹配找到种子节点
        2. _expand_to_siblings 扩展到全部同结构节点
        3. 随机森林训练，找出概率最高的代表节点
        4. 从代表节点构建 stack（后续由 _stack_to_xpath 转为 XPath）
        """
        all_nodes = [n for n in soup.find_all(True) if n.name]
        all_features = [_extract_node_features(n, soup) for n in all_nodes]

        for alias, targets in wanted_dict.items():
            targets = [normalize(t) for t in targets]

            # ── 找种子节点 ──
            seed_nodes = []
            for target in targets:
                target_path = urlparse(target).path if target.startswith("http") else (
                    target if target.startswith("/") else None
                )
                for node in all_nodes:
                    hit = False
                    # 文本匹配（复用 text_match，与 _child_matches 一致）
                    text = normalize(node.get_text(strip=True))
                    hit = text_match(target, text, fuzz_ratio)
                    # 属性匹配
                    if not hit:
                        for val in node.attrs.values():
                            if not isinstance(val, str):
                                continue
                            val_s = normalize(val.strip())
                            if text_match(target, val_s, fuzz_ratio):
                                hit = True
                                break
                            if target_path and val_s.rstrip("/") == target_path.rstrip("/"):
                                hit = True
                                break
                    if hit:
                        # 只保留最内层节点（排除祖先）
                        is_ancestor = any(node in list(sn.parents) for sn in seed_nodes)
                        if not is_ancestor:
                            seed_nodes.append(node)

            if not seed_nodes:
                logger.warning("alias='%s' ML 模式未找到种子节点，跳过", alias)
                continue

            # ── 扩展到同结构兄弟 ──
            positive_indices = self._expand_to_siblings(seed_nodes, all_nodes)
            for sn in seed_nodes:
                if sn in all_nodes:
                    positive_indices.add(all_nodes.index(sn))

            logger.info(
                "alias='%s' ML 训练：%d 种子 → %d 正样本 / %d 总节点",
                alias, len(seed_nodes), len(positive_indices), len(all_nodes),
            )

            # ── 训练随机森林 ──
            vocab = None
            vecs = []
            for feat in all_features:
                vec, vocab = _features_to_vector(feat, vocab)
                vecs.append(vec)

            X = np.array(vecs)
            y = np.array([1 if i in positive_indices else 0 for i in range(len(all_nodes))])

            clf = RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1,
            )
            clf.fit(X, y)

            # ── 预测：找概率最高的代表节点 → 构建 stack ──
            proba = clf.predict_proba(X)
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            best_i = int(np.argmax(proba[:, pos_idx]))
            representative = all_nodes[best_i]

            # 判断字段类型：如果样本是 URL，wanted_attr 设为 href
            is_url_field = any(t.startswith("http") or t.startswith("/") for t in targets)
            if is_url_field and representative.get("href"):
                representative.wanted_attr = "href"
            elif not hasattr(representative, "wanted_attr"):
                representative.wanted_attr = None

            stack = self._build_stack(representative, url)
            stack["alias"] = alias
            self.stack_list.append(stack)

            xpath = self._stack_to_xpath(stack)
            logger.info(
                "alias='%s' ML → XPath: %s  (代表节点: <%s> prob=%.3f)",
                alias, xpath, representative.name, proba[best_i][pos_idx],
            )

    def build_xpath(self, url=None, wanted_dict=None, html=None, **kwargs):
        self.build(url=url, wanted_dict=wanted_dict, html=html, **kwargs)
        return self.get_result_xpath_rule()

    def get_result_xpath_rule(self, url=None):
        if not self.stack_list: return {}
        rules = {}
        for stack in self.stack_list:
            alias = stack.get("alias", "default")
            if alias not in rules:
                rules[alias] = self._stack_to_xpath(stack)
        return rules

    def _stack_to_xpath(self, stack):
        """将学习到的 stack 转换为精准 XPath 表达式。

        策略：
        - ID 锚点：遇到稳定 ID 时重置路径，且不再加 class（ID 已足够唯一）
        - class 匹配：过滤 variant class（first/odd/active 等位置/状态类）后使用词边界匹配
        - 位置谓词：用 tag-only 兄弟计数判断 is_repeating，避免 class-filtered count 偏低
        - 路径类型：起始用 //，后续用 / 表示直接父子关系

        数据结构说明：
        content[k] 可能是：
          2-tuple (tag, attrs) — 叶子节点或无索引的层级
          4-tuple (tag, attrs, child_index, child_sibling_count) — 旧格式中间层级
          5-tuple (tag, attrs, child_index, child_sibling_count, tag_only_count) — 新格式
        其中 child_index/child_sibling_count 描述 content[k+1] 在同类兄弟中的位置/总数
        """
        content = stack.get("content", [])
        if not content:
            return None

        xpath_parts = []
        last_idx = len(content) - 1

        for i, item in enumerate(content):
            tag = item[0]
            attrs = item[1] if len(item) > 1 else {}

            if tag in ("html", "body", "[document]"):
                continue

            # ── 读取兄弟索引数据（存储在上一个 content 条目）──
            sibling_idx = None
            sibling_count = None       # class-filtered count
            tag_only_count = None      # tag-only count（新增）
            if i > 0:
                prev = content[i - 1]
                if len(prev) >= 4:
                    sibling_idx = prev[2]
                    sibling_count = prev[3]
                if len(prev) >= 5:
                    tag_only_count = prev[4]
                else:
                    tag_only_count = sibling_count  # 向后兼容

            attr_predicates = []
            has_id = False

            # ── ID 谓词 ──
            node_id = attrs.get("id", "")
            if node_id and _is_stable_id(node_id):
                attr_predicates.append(f"@id='{node_id}'")
                has_id = True
                xpath_parts = []  # ID 全局唯一，重置路径
                # 有 ID 时不再加 class（ID 已足够唯一）

            # ── class 谓词（仅无 ID 时）──
            if not has_id:
                classes = attrs.get("class", [])
                if isinstance(classes, str):
                    classes = classes.split()
                # 过滤 variant class（first/odd/active 等位置/状态类）
                classes = [c for c in classes if not _is_variant_class(c)]

                # 分离稳定 class 和 Tailwind 原子 class
                stable = [c for c in classes if not _is_tailwind_class(c)]

                if stable:
                    # 有稳定 class → 用作锚点，重置路径（丢弃前面所有 Tailwind 父节点）
                    xpath_parts = []
                    for cls in stable:
                        attr_predicates.append(
                            f"contains(concat(' ',@class,' '),' {cls} ')"
                        )
                # 全是 Tailwind class → 不加 class 谓词，只保留 tag（靠位置谓词区分）

            # ── 构造节点段 ──
            part = tag
            if attr_predicates:
                part += "[" + " and ".join(attr_predicates) + "]"

            # ── 位置谓词 ──
            # 用 tag_only_count 判断是否为重复元素（比 class-filtered count 更准确）
            is_leaf = (i == last_idx)
            is_repeating = tag_only_count is not None and tag_only_count > 1
            if sibling_idx is not None and not has_id and not is_leaf and not is_repeating:
                part += f"[{sibling_idx + 1}]"

            xpath_parts.append(part)

        if not xpath_parts:
            return None

        xpath = "//" + "/".join(xpath_parts)

        wanted_attr = stack.get("wanted_attr")
        if wanted_attr:
            xpath += f"/@{wanted_attr}"

        return xpath

    def get_result_similar(self, url=None, html=None, soup=None,
                           group_by_alias=False, unique=True, **kwargs):
        """用已学习的 XPath 规则在页面上提取数据。

        Args:
            url: 目标页面 URL
            html: 目标页面 HTML 字符串
            soup: 已解析的 BeautifulSoup 对象（会转为 lxml tree）
            group_by_alias: True 返回 {alias: [values]}，False 返回扁平列表
            unique: 是否去重（默认 True）

        Returns:
            dict | list: 提取结果
        """
        from lxml import html as lxml_html

        rules = self.get_result_xpath_rule()
        if not rules:
            return {} if group_by_alias else []

        # 与 build()/_get_soup() 保持一致的解析链:
        # normalize + unescape → BS4(lxml) → str → lxml_html
        # 避免 BS4 与 lxml 直接解析同一 HTML 产出不同 DOM 树导致 XPath 匹配失败
        if html:
            soup_obj = self._get_soup(html=html)
            tree = lxml_html.fromstring(str(soup_obj))
        elif url:
            soup_obj = self._get_soup(url=url, request_args=kwargs.get("request_args"))
            tree = lxml_html.fromstring(str(soup_obj))
        elif soup:
            tree = lxml_html.fromstring(str(soup))
        else:
            return {} if group_by_alias else []

        results = defaultdict(list)
        for alias, xpath in rules.items():
            try:
                nodes = tree.xpath(xpath)
            except Exception as e:
                logger.warning("XPath 执行失败 alias='%s' xpath='%s': %s", alias, xpath, e)
                continue
            skipped_empty = 0
            skipped_dup = 0
            seen = set()
            for n in nodes:
                val = n if isinstance(n, str) else n.text_content().strip()
                if not val:
                    skipped_empty += 1
                    continue
                if unique and val in seen:
                    skipped_dup += 1
                    continue
                seen.add(val)
                results[alias].append(val)
            logger.debug(
                "XPath alias='%s': matched=%d, kept=%d, empty=%d, dup=%d | xpath=...%s",
                alias, len(nodes), len(results[alias]),
                skipped_empty, skipped_dup, xpath[-80:],
            )

        if group_by_alias:
            return dict(results)
        return [v for sub in results.values() for v in sub]
