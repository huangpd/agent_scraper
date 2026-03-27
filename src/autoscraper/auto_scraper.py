      
"""
AutoScraper - 改进版
主要改进：
  1. XPath 锚点策略：data-* > aria-* > role > 稳定ID > 语义class > 结构位置
  2. Tailwind / 原子CSS 工具类过滤
  3. 布局通用类过滤（clearfix/container/wrapper 等）
  4. class 谓词数量上限，防止过度约束
  5. _get_valid_attrs 扩展：采集 data-* / aria-* 属性
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
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
# 锚点属性优先级（从高到低）
# ─────────────────────────────────────────────────────────
# 遇到这些属性时重置 xpath_parts（从锚点重新出发），且不再叠加 class 谓词
ANCHOR_ATTRS_PRIORITY = [
    "data-testid",    # 测试专用，开发者有意稳定
    "data-cy",        # Cypress 测试锚点
    "data-test",      # 通用测试锚点
    "data-id",        # 业务 ID
    "data-key",       # 列表 key
    "data-type",      # 类型标记
    "data-name",      # 名称标记
    "data-target",    # Svelte/Stimulus 等框架的组件标记
    "data-component", # 组件标记
    "data-section",   # 区块标记
    "aria-label",     # 无障碍标签，语义明确
    "aria-labelledby",
    "role",           # ARIA role（navigation/main/list 等）
]

# role 值白名单：只有语义 role 才做锚点，忽略 presentation/none
_SEMANTIC_ROLES = frozenset({
    "main", "navigation", "complementary", "banner", "contentinfo",
    "search", "form", "region", "list", "listitem",
    "table", "row", "cell", "columnheader", "rowheader",
    "article", "dialog", "alert", "status", "tab", "tabpanel",
    "button", "link", "menuitem", "option", "treeitem",
})

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

# ─────────────────────────────────────────────────────────
# Tailwind / 原子CSS 工具类过滤
# 规则：纯布局/尺寸/颜色/间距类，对定位无意义
# ─────────────────────────────────────────────────────────
_TAILWIND_PATTERNS = [
    # 响应式 / 状态前缀
    re.compile(r'^(sm|md|lg|xl|2xl|dark|light|hover|focus|focus-within|'
               r'focus-visible|active|visited|disabled|checked|group|peer|'
               r'first|last|odd|even|not|aria|data|print|rtl|ltr):'),
    # 布局类型关键字（单独出现）
    re.compile(r'^(flex|grid|block|inline|inline-block|inline-flex|inline-grid|'
               r'hidden|invisible|visible|contents|flow-root|list-item|'
               r'table|table-row|table-cell|relative|absolute|fixed|sticky|static)$'),
    # flex / grid 对齐
    re.compile(r'^(items|justify|content|self|place|justify-items|justify-self|'
               r'place-items|place-content|place-self)[-]'),
    # 尺寸
    re.compile(r'^(w|h|min-w|min-h|max-w|max-h|size)[-]'),
    # 内外边距
    re.compile(r'^(p|m|px|py|pt|pb|pl|pr|mx|my|mt|mb|ml|mr|ps|pe|ms|me)[-]'),
    # 间距
    re.compile(r'^(gap|gap-x|gap-y|space-x|space-y)[-]'),
    # 排版
    re.compile(r'^(text|font|leading|tracking|indent|align|whitespace|'
               r'break|hyphens|list|decoration|underline|overline|'
               r'line-through|no-underline)[-]'),
    re.compile(r'^(text|font|leading|tracking|align)$'),
    # 颜色 / 背景 / 边框
    re.compile(r'^(bg|border|ring|shadow|outline|divide|accent|caret|'
               r'fill|stroke)[-]'),
    re.compile(r'^(border|outline|ring|shadow|divide)$'),
    # 圆角 / overflow / 截断
    re.compile(r'^(rounded|overflow|truncate|overscroll|scroll)'),
    # grid span / order
    re.compile(r'^(col|row)[-](span|start|end|auto)'),
    re.compile(r'^(col|row)[-]\d'),
    re.compile(r'^(order|grow|shrink|basis|flex)[-]'),
    re.compile(r'^(grow|shrink)$'),
    # z-index / opacity / cursor / transition / transform
    re.compile(r'^(z|opacity|cursor|pointer-events|select|resize|'
               r'transition|duration|ease|delay|animate|'
               r'scale|rotate|translate|skew|origin|'
               r'transform|will-change)[-]'),
    re.compile(r'^(transform|transition|animate|opacity|resize)$'),
    # 杂项布局
    re.compile(r'^(container|contents|aspect|object|float|clear|'
               r'inset|top|bottom|left|right|start|end)[-]?'),
    # 伪元素 / before / after
    re.compile(r'^(before|after):'),
    # 纯数字 class（bootstrap grid: col-6 等）
    re.compile(r'^[a-z]+-\d+$'),
]

def _is_tailwind_class(cls: str) -> bool:
    return any(p.search(cls) for p in _TAILWIND_PATTERNS)


# ─────────────────────────────────────────────────────────
# Variant class 过滤（位置/状态类）
# ─────────────────────────────────────────────────────────
_VARIANT_EXACT = frozenset({
    "first", "last", "odd", "even",
    "active", "current", "selected",
    "open", "closed", "disabled", "hidden", "visible",
})
_VARIANT_PATTERNS = [
    re.compile(r'-(first|last|odd|even|active|current|selected)$', re.I),
    re.compile(r'^(first|last|odd|even|active)-', re.I),
    re.compile(r'^is-'),
]

# 通用布局工具 class（非 Tailwind 框架也会出现）
_LAYOUT_UTILITY_CLASSES = frozenset({
    "clearfix", "clear", "cf",
    "container", "wrapper", "wrap", "inner", "outer",
    "row", "col", "column",
    "grid", "flex",
    "block", "inline",
    "left", "right", "center",
    "pull-left", "pull-right", "float-left", "float-right",
    "d-flex", "d-block", "d-none", "d-grid",   # Bootstrap 5
})

def _is_variant_class(cls: str) -> bool:
    if cls.lower() in _VARIANT_EXACT:
        return True
    if cls.lower() in _LAYOUT_UTILITY_CLASSES:
        return True
    return any(p.search(cls) for p in _VARIANT_PATTERNS)


def _is_noise_class(cls: str) -> bool:
    """综合判断一个 class 是否为噪声（对 XPath 定位无意义）。"""
    return _is_hashed_class(cls) or _is_tailwind_class(cls) or _is_variant_class(cls)


def _stable_classes(classes) -> list:
    if not classes:
        return []
    if isinstance(classes, str):
        classes = classes.split()
    return [c for c in classes if not _is_noise_class(c)]


# ─────────────────────────────────────────────────────────
# 动态 ID 检测
# ─────────────────────────────────────────────────────────
_DYNAMIC_ID_PATTERNS = [
    re.compile(r'\d{5,}'),
    re.compile(r'^[a-z]-[a-f0-9]{6,}'),
    re.compile(r'^[a-f0-9]{8,}$'),
    re.compile(r'[-_][a-f0-9]{6,}$'),
    re.compile(r'^:'),
    re.compile(r'^(ember|react|vue|ng|svelte)-'),
]

def _is_stable_id(id_str: str) -> bool:
    if not id_str:
        return False
    for pattern in _DYNAMIC_ID_PATTERNS:
        if pattern.search(id_str):
            return False
    return True


# ─────────────────────────────────────────────────────────
# ML 特征提取（保持不变）
# ─────────────────────────────────────────────────────────
_NUMERIC_KEYS = ['depth', 'sibling_count', 'sibling_index', 'child_count',
                 'text_length', 'has_href', 'has_img', 'repeated_siblings']
_CAT_KEYS = ['tag', 'id_prefix', 'ancestor_0_tag', 'ancestor_1_tag']

def _extract_node_features(node, soup) -> dict:
    f = {'tag': node.name or '', 'depth': len(list(node.parents))}
    parent = node.parent
    if parent:
        siblings = [s for s in parent.children if hasattr(s, 'name') and s.name == node.name]
        f['sibling_count'] = len(siblings)
        f['sibling_index'] = siblings.index(node) if node in siblings else 0
        # 同结构兄弟（tag + stable class 相同）
        stable = tuple(sorted(_stable_classes(node.attrs.get('class', []))))
        repeated = [s for s in siblings
                    if tuple(sorted(_stable_classes(s.attrs.get('class', [])))) == stable]
        f['repeated_siblings'] = len(repeated)
    else:
        f['sibling_count'] = f['sibling_index'] = f['repeated_siblings'] = 0

    f['id_prefix'] = re.sub(r'-?\d+$', '', node.attrs.get('id', ''))
    f['child_count'] = len(list(node.children))
    # 文本特征
    text = node.get_text(strip=True)
    f['text_length'] = len(text)
    # 链接/图片特征
    f['has_href'] = int(bool(node.attrs.get('href')))
    f['has_img'] = int(bool(node.find('img')))
    ancestors = list(node.parents)
    f['ancestor_0_tag'] = ancestors[0].name if len(ancestors) > 0 else ''
    f['ancestor_1_tag'] = ancestors[1].name if len(ancestors) > 1 else ''
    return f

def _features_to_vector(features: dict, vocab: dict = None):
    numeric_vec = [float(features.get(k, 0)) for k in _NUMERIC_KEYS]
    building = vocab is None
    if building:
        vocab = {k: {} for k in _CAT_KEYS}
    cat_vec = []
    for k in _CAT_KEYS:
        val = str(features.get(k, ''))
        if building and val not in vocab[k]:
            vocab[k][val] = len(vocab[k])
        cat_vec.append(float(vocab[k].get(val, len(vocab[k]))))
    return np.array(numeric_vec + cat_vec, dtype=np.float32), vocab


# ─────────────────────────────────────────────────────────
# AutoScraper 主类
# ─────────────────────────────────────────────────────────

class AutoScraper(object):
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # XPath 锚点策略参数
    MAX_CLASS_PREDICATES = 2    # 每个节点最多保留几个 class 谓词
    MIN_CLASS_LENGTH = 3        # 过短的 class 名通常无意义（如 "h1" "xs"），跳过

    def __init__(self, stack_list=None):
        self.stack_list = stack_list or []
        self._ml_active = False
        self._ml_classifiers: dict = {}
        self._xpath_overrides: dict[str, str] = {}  # 外部直接注入的 XPath 规则

    # ── 持久化 ──────────────────────────────────────────

    def save(self, file_path):
        data = {
            "stack_list": self.stack_list,
            "xpath_rules": self.get_result_xpath_rule(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, file_path):
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
        """
        采集节点属性，优先级：
          1. data-* 锚点属性（ANCHOR_ATTRS_PRIORITY 中的 data-/aria-/role）
          2. id
          3. class（只保留 stable classes）

        注意：只采集 ANCHOR_ATTRS_PRIORITY 中明确列出的 data-* 属性，
        避免采集动态注入的 data-v-xxxx / data-reactid 等框架内部属性。
        """
        attrs = {}

        # 采集锚点属性
        for anchor in ANCHOR_ATTRS_PRIORITY:
            val = item.attrs.get(anchor)
            if val:
                # role 只保留语义 role
                if anchor == "role":
                    if isinstance(val, list):
                        val = val[0]
                    if val not in _SEMANTIC_ROLES:
                        continue
                if isinstance(val, list):
                    val = val[0]
                attrs[anchor] = val.strip()

        # id
        id_val = item.attrs.get("id", "")
        if isinstance(id_val, list):
            id_val = id_val[0]
        if id_val:
            attrs["id"] = id_val.strip()

        # class（只保留 stable classes）
        raw_classes = item.attrs.get("class", [])
        stable = _stable_classes(raw_classes)
        if stable:
            attrs["class"] = stable

        return attrs

    def build(self, url=None, wanted_list=None, wanted_dict=None, html=None,
              request_args=None, update=False, text_fuzz_ratio=1.0, use_ml=True):
        soup = self._get_soup(url=url, html=html, request_args=request_args)
        if not update:
            self.stack_list = []

        _wdict = wanted_dict or {"default": wanted_list or []}

        for alias, targets in _wdict.items():
            for target in targets:
                target = normalize(target)
                for child in reversed(soup.find_all(True)):
                    if self._child_matches(child, target, url, text_fuzz_ratio):
                        stack = self._build_stack(child, url)
                        stack["alias"] = alias
                        self.stack_list.append(stack)

        self.stack_list = unique_stack_list(self.stack_list)

        if not self.stack_list and use_ml and _ML_AVAILABLE:
            logger.info("规则模式未找到结果，切换到 ML 模式...")
            self._ml_active = True
            self._ml_wanted_dict = _wdict
            self._ml_build_stacks(soup, url, _wdict, text_fuzz_ratio)
            self.stack_list = unique_stack_list(self.stack_list)

        rules = self.get_result_xpath_rule()
        if rules:
            for alias, xpath in rules.items():
                logger.debug("最终 XPath  alias='%s': %s", alias, xpath)
        return rules

    def _child_matches(self, child, text, url, fuzz):
        child_text = normalize(child.get_text(strip=True))
        if text_match(text, child_text, fuzz):
            parent_text = normalize(child.parent.get_text(strip=True)) if child.parent else ""
            if child_text == parent_text and child.parent and child.parent.parent:
                return False
            child.wanted_attr = None
            return True

        non_rec = normalize(get_non_rec_text(child))
        if non_rec and text_match(text, non_rec, fuzz):
            child.wanted_attr = None
            child.is_non_rec_text = True
            return True

        for k, v in child.attrs.items():
            if not isinstance(v, str):
                continue
            v = normalize(v.strip())
            if text_match(text, v, fuzz):
                child.wanted_attr = k
                return True
            if k in ("href", "src"):
                if url:
                    full_url = urljoin(url, v)
                    if text_match(text, full_url, fuzz):
                        child.wanted_attr = k
                        return True
                parsed = urlparse(v)
                if parsed.scheme and text.startswith("/") and parsed.path == text:
                    child.wanted_attr = k
                    return True
        return False

    @classmethod
    def _build_stack(cls, child, url):
        content = [(child.name, cls._get_valid_attrs(child))]
        parent = child
        while True:
            gp = parent.find_parent()
            if not gp or gp.name == "[document]":
                break
            search_attrs = cls._get_valid_attrs(parent)
            # 移除空 class 避免 BS4 匹配异常
            if "class" in search_attrs and not search_attrs["class"]:
                search_attrs.pop("class")
            # BS4 只能用 id/class 做 find_all 过滤，data-*/aria-* 需要手动处理
            bs4_attrs = {k: v for k, v in search_attrs.items() if k in ("id", "class")}
            siblings = gp.find_all(parent.name, bs4_attrs, recursive=False)
            tag_only_siblings = gp.find_all(parent.name, recursive=False)
            tag_only_count = len(tag_only_siblings)
            for i, c in enumerate(siblings):
                if c == parent:
                    content.insert(0, (gp.name, cls._get_valid_attrs(gp), i, len(siblings), tag_only_count))
                    break
            else:
                content.insert(0, (gp.name, cls._get_valid_attrs(gp)))
            parent = gp

        wanted_attr = getattr(child, "wanted_attr", None)
        hash_input = str((content, wanted_attr)).encode()
        stack = {
            "content": content,
            "wanted_attr": wanted_attr,
            "hash": hashlib.sha256(hash_input).hexdigest(),
            "stack_id": hashlib.sha256(hash_input).hexdigest()[:8],
        }
        return stack

    # ── ML 辅助 ──────────────────────────────────────────

    @staticmethod
    def _node_signature(node) -> str:
        stable = tuple(sorted(_stable_classes(node.attrs.get('class', []))))
        parent_tag = node.parent.name if node.parent else ''
        return f"{node.name}|{stable}|{parent_tag}"

    def _expand_to_siblings(self, seed_nodes, all_nodes) -> set:
        expanded = set()
        for seed in seed_nodes:
            sig = self._node_signature(seed)
            parent = seed.parent
            if parent is None:
                continue
            same_parent = {
                i for i, node in enumerate(all_nodes)
                if node.parent is parent and self._node_signature(node) == sig
            }
            if len(same_parent) > 1:
                expanded |= same_parent
            else:
                parent_sig = self._node_signature(parent)
                for i, node in enumerate(all_nodes):
                    if (self._node_signature(node) == sig
                            and node.parent is not None
                            and self._node_signature(node.parent) == parent_sig):
                        expanded.add(i)
        return expanded

    def _ml_build_stacks(self, soup, url, wanted_dict, fuzz_ratio):
        all_nodes = [n for n in soup.find_all(True) if n.name]
        all_features = [_extract_node_features(n, soup) for n in all_nodes]

        for alias, targets in wanted_dict.items():
            targets = [normalize(t) for t in targets]
            seed_nodes = []
            for target in targets:
                target_path = urlparse(target).path if target.startswith("http") else (
                    target if target.startswith("/") else None
                )
                for node in all_nodes:
                    hit = False
                    text = normalize(node.get_text(strip=True))
                    hit = text_match(target, text, fuzz_ratio)
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
                        is_ancestor = any(node in list(sn.parents) for sn in seed_nodes)
                        if not is_ancestor:
                            seed_nodes.append(node)

            if not seed_nodes:
                logger.warning("alias='%s' ML 模式未找到种子节点，跳过", alias)
                continue

            positive_indices = self._expand_to_siblings(seed_nodes, all_nodes)
            for sn in seed_nodes:
                if sn in all_nodes:
                    positive_indices.add(all_nodes.index(sn))

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

            proba = clf.predict_proba(X)
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            best_i = int(np.argmax(proba[:, pos_idx]))
            representative = all_nodes[best_i]

            is_url_field = any(t.startswith("http") or t.startswith("/") for t in targets)
            if is_url_field and representative.get("href"):
                representative.wanted_attr = "href"
            elif not hasattr(representative, "wanted_attr"):
                representative.wanted_attr = None

            stack = self._build_stack(representative, url)
            stack["alias"] = alias
            self.stack_list.append(stack)
            self._ml_classifiers[alias] = (clf, vocab)

    def build_xpath(self, url=None, wanted_dict=None, html=None, **kwargs):
        self.build(url=url, wanted_dict=wanted_dict, html=html, **kwargs)
        return self.get_result_xpath_rule()

    def get_result_xpath_rule(self, url=None):
        if not self.stack_list and not self._xpath_overrides:
            return {}
        rules = {}
        for stack in self.stack_list:
            alias = stack.get("alias", "default")
            if alias not in rules:
                rules[alias] = self._stack_to_xpath(stack)
        rules.update(self._xpath_overrides)
        return rules

    # ─────────────────────────────────────────────────────────
    # _stack_to_xpath：XPath 锚点策略核心
    # ─────────────────────────────────────────────────────────
    def _stack_to_xpath(self, stack):
        """
        将 stack 转换为 XPath，采用锚点优先策略：

        锚点优先级（遇到即重置路径，从 // 重新出发）：
          1. data-testid / data-cy / data-test / data-id / data-key /
             data-type / data-name / data-target / data-component / data-section
          2. aria-label / aria-labelledby
          3. role（仅语义 role）
          4. 稳定 ID（非动态生成）
          5. 语义 class（过滤 Tailwind/工具类后剩余的，最多 MAX_CLASS_PREDICATES 个）
          6. 结构位置（最后的兜底）

        各层谓词构造规则：
          - 有锚点/ID：直接用等值匹配（@attr='val'），不叠加 class
          - 仅有 class：uses contains(concat()) 词边界匹配，最多 MAX_CLASS_PREDICATES 个
          - 无任何有效属性：只用 tag，必要时加位置谓词 [n]
          - is_repeating（列表场景）：不加位置谓词，让 XPath 匹配所有兄弟
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

            # ── 读取兄弟索引（存在上一个 content 条目里）──
            sibling_idx = None
            sibling_count = None
            tag_only_count = None
            if i > 0:
                prev = content[i - 1]
                if len(prev) >= 4:
                    sibling_idx = prev[2]
                    sibling_count = prev[3]
                if len(prev) >= 5:
                    tag_only_count = prev[4]
                else:
                    tag_only_count = sibling_count

            is_leaf = (i == last_idx)
            is_repeating = tag_only_count is not None and tag_only_count > 1

            # ════════════════════════════════════════════
            # 锚点检测：按优先级找第一个有效锚点
            # ════════════════════════════════════════════
            anchor_pred = self._find_anchor_predicate(attrs)

            if anchor_pred:
                # 找到锚点 → 重置路径（全局锚点，用 // 出发）
                xpath_parts = []
                xpath_parts.append(f"{tag}[{anchor_pred}]")
                continue  # 锚点节点不加位置谓词，也不加 class

            # ════════════════════════════════════════════
            # 无锚点：按优先级构造谓词
            # ════════════════════════════════════════════
            attr_predicates = []
            has_id = False

            # ── 稳定 ID ──
            node_id = attrs.get("id", "")
            if node_id and _is_stable_id(node_id):
                attr_predicates.append(f"@id='{node_id}'")
                has_id = True
                xpath_parts = []  # ID 全局唯一，重置路径

            # ── 语义 class（仅无 ID 时，且过滤后有剩余）──
            if not has_id:
                classes = attrs.get("class", [])
                if isinstance(classes, str):
                    classes = classes.split()
                # 再次过滤（_get_valid_attrs 已经过滤一次，这里双重保险）
                classes = [c for c in classes if not _is_noise_class(c)]
                # 过滤过短的 class 名（通常是缩写工具类）
                classes = [c for c in classes if len(c) >= self.MIN_CLASS_LENGTH]
                # 按"越具体越好"排序：优先保留含连字符或大写的语义名
                classes = sorted(classes, key=lambda c: (
                    -int(bool(re.search(r'[A-Z]|-', c))),  # 有大写/连字符优先
                    len(c),                                  # 较短的排前面（通常更具体）
                ))
                # 最多保留 MAX_CLASS_PREDICATES 个
                classes = classes[:self.MAX_CLASS_PREDICATES]
                for cls in classes:
                    attr_predicates.append(
                        f"contains(concat(' ',@class,' '),' {cls} ')"
                    )

            # ── 构造节点段 ──
            part = tag
            if attr_predicates:
                part += "[" + " and ".join(attr_predicates) + "]"

            # ── 位置谓词（列表场景不加，避免只匹配第一行）──
            if (sibling_idx is not None
                    and not has_id
                    and not is_leaf
                    and not is_repeating):
                part += f"[{sibling_idx + 1}]"

            xpath_parts.append(part)

        if not xpath_parts:
            return None

        xpath = "//" + "/".join(xpath_parts)

        wanted_attr = stack.get("wanted_attr")
        if wanted_attr:
            xpath += f"/@{wanted_attr}"

        return xpath

    def _find_anchor_predicate(self, attrs: dict) -> str | None:
        """
        按 ANCHOR_ATTRS_PRIORITY 顺序扫描 attrs，返回第一个有效锚点的谓词字符串。

        返回格式示例：
          "@data-testid='file-row'"
          "@aria-label='file navigation'"
          "@role='navigation'"

        无有效锚点时返回 None。
        """
        for anchor in ANCHOR_ATTRS_PRIORITY:
            val = attrs.get(anchor)
            if not val:
                continue
            if isinstance(val, list):
                val = val[0]
            val = val.strip()
            if not val:
                continue
            # role 只接受语义 role
            if anchor == "role" and val not in _SEMANTIC_ROLES:
                continue
            # aria-label / data-* 值不能太短（避免 "a" "1" 这类噪声）
            if len(val) < 2:
                continue
            return f"@{anchor}='{val}'"
        return None

    def get_result_similar(self, url=None, html=None, soup=None,
                           group_by_alias=False, unique=True, **kwargs):
        from lxml import html as lxml_html

        rules = self.get_result_xpath_rule()
        if not rules:
            return {} if group_by_alias else []

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
