"""
AutoScraper - ML Enhanced Drop-in Replacement
原版 API 完全兼容，新增两项能力：
  1. _get_valid_attrs 自动过滤哈希 class（如 prc-Counter-Badge-wQ2rT）
  2. build() 失败时自动 fallback 到随机森林分类器
"""

import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from html import unescape
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from autoscraper.utils import (
    FuzzyText,
    ResultItem,
    get_non_rec_text,
    normalize,
    text_match,
    unique_hashable,
    unique_stack_list,
)

# ─────────────────────────────────────────────────────────
# ML 依赖（软依赖，未安装时自动降级为纯规则模式）
# ─────────────────────────────────────────────────────────
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False


# ─────────────────────────────────────────────────────────
# 哈希 class 检测
# ─────────────────────────────────────────────────────────
_HASHED_PATTERNS = [
    re.compile(r'-[A-Za-z0-9]{4,8}$'),             # 末尾 -xXxX，如 Badge-wQ2rT
    re.compile(r'__[A-Za-z0-9]{4,}$'),              # CSS Modules，如 content__IwGAp
    re.compile(r'^[a-z]-[a-f0-9]{6,}'),             # 单字母前缀 + hex
    re.compile(r'[A-Z]{2,}[0-9][A-Za-z0-9]{2,}'),  # 混合大写+数字，如 PageLayout3Xk
]

def _is_hashed_class(cls: str) -> bool:
    for pattern in _HASHED_PATTERNS:
        if pattern.search(cls):
            suffix = cls.split('-')[-1]
            if any(c.isdigit() for c in suffix) and any(c.isalpha() for c in suffix):
                return True
    return False

def _stable_classes(classes) -> list:
    if not classes:
        return []
    if isinstance(classes, str):
        classes = classes.split()
    return [c for c in classes if not _is_hashed_class(c)]


# ─────────────────────────────────────────────────────────
# ML 特征提取（仅在 sklearn 可用时使用）
# ─────────────────────────────────────────────────────────
_SEMANTIC_TAGS = {
    'h1','h2','h3','h4','h5','h6','p','a','span','strong',
    'em','li','td','th','code','pre','blockquote','button','label',
}
_SEMANTIC_CLASSES = [
    'f4','f3','f1','btn','link','title','readme','description',
    'counter','label','text','name','number','badge','header',
]
_NUMERIC_KEYS = [
    'depth','sibling_count','sibling_index','sibling_ratio',
    'stable_class_count','has_href','has_id','has_title_attr',
    'text_len','text_len_bucket','has_digits','is_short_number',
    'child_count','is_semantic_tag',
] + [f'has_cls_{c}' for c in _SEMANTIC_CLASSES]
_CAT_KEYS = ['tag', 'stable_classes', 'id_value', 'id_prefix',
             'ancestor_0_tag', 'ancestor_1_tag', 'ancestor_2_tag']

def _extract_node_features(node, soup) -> dict:
    f = {}
    f['tag'] = node.name or ''
    f['is_semantic_tag'] = int(node.name in _SEMANTIC_TAGS)

    ancestors = list(node.parents)
    f['depth'] = len(ancestors)

    parent = node.parent
    if parent:
        siblings = [s for s in parent.children if hasattr(s, 'name') and s.name == node.name]
        f['sibling_count'] = len(siblings)
        f['sibling_index'] = siblings.index(node) if node in siblings else 0
        f['sibling_ratio'] = f['sibling_index'] / max(f['sibling_count'] - 1, 1)
    else:
        f['sibling_count'] = f['sibling_index'] = 0
        f['sibling_ratio'] = 0.0

    classes = _stable_classes(node.attrs.get('class', []))
    f['stable_class_count'] = len(classes)
    f['stable_classes'] = ' '.join(sorted(classes))
    for c in _SEMANTIC_CLASSES:
        f[f'has_cls_{c}'] = int(c in classes)

    f['has_href'] = int('href' in node.attrs)
    f['has_id'] = int('id' in node.attrs)
    f['has_title_attr'] = int('title' in node.attrs)
    f['id_value'] = node.attrs.get('id', '')
    # id 前缀作为稳定特征（如 repo-stars vs repo-forks，去掉纯数字后缀）
    raw_id = node.attrs.get('id', '')
    f['id_prefix'] = re.sub(r'-?\d+$', '', raw_id)  # repo-stars-counter-star → repo-stars-counter-star

    text = node.get_text(strip=True)
    f['text_len'] = len(text)
    f['text_len_bucket'] = min(len(text) // 20, 10)
    f['has_digits'] = int(any(c.isdigit() for c in text))
    f['is_short_number'] = int(bool(re.match(r'^\d+\.?\d*[kKmMbB]?$', text)))
    f['child_count'] = len(list(node.children))

    for i, anc in enumerate(ancestors[:3]):
        f[f'ancestor_{i}_tag'] = anc.name or ''

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
# 主类：AutoScraper（兼容原版，自动 ML 增强）
# ─────────────────────────────────────────────────────────

class AutoScraper(object):
    """
    AutoScraper : A Smart, Automatic, Fast and Lightweight Web Scraper for Python.
    ML Enhanced: 自动过滤哈希 class，并在规则失效时 fallback 到随机森林分类器。

    新增参数：
      build(..., use_ml=True)            - 是否启用 ML fallback（默认开启）
      get_result_similar(..., ml_threshold=0.5) - ML 模式下的概率阈值

    原版所有 API 完全不变。
    """

    request_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/84.0.4147.135 Safari/537.36"
    }

    def __init__(self, stack_list=None):
        self.stack_list = stack_list or []
        # ML 状态（规则模式失败时启用）
        self._ml_classifiers = {}
        self._ml_vocabs = {}
        self._ml_n_positives = {}
        self._ml_active = False

    # ── 持久化 ──────────────────────────────────────────

    def save(self, file_path):
        data = dict(stack_list=self.stack_list)
        with open(file_path, "w") as f:
            json.dump(data, f)

    def load(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            self.stack_list = data
            return
        self.stack_list = data["stack_list"]

    # ── 网络 / 解析 ──────────────────────────────────────

    @classmethod
    def _fetch_html(cls, url, request_args=None):
        request_args = request_args or {}
        headers = dict(cls.request_headers)
        if url:
            headers["Host"] = urlparse(url).netloc
        user_headers = request_args.pop("headers", {})
        headers.update(user_headers)
        res = requests.get(url, headers=headers, **request_args)
        if res.encoding == "ISO-8859-1" and "ISO-8859-1" not in res.headers.get("Content-Type", ""):
            res.encoding = res.apparent_encoding
        return res.text

    @classmethod
    def _get_soup(cls, url=None, html=None, request_args=None):
        if html:
            return BeautifulSoup(normalize(unescape(html)), "lxml")
        html = cls._fetch_html(url, request_args)
        return BeautifulSoup(normalize(unescape(html)), "lxml")

    # ── 核心改动①：_get_valid_attrs 过滤哈希 class ──────

    @staticmethod
    def _get_valid_attrs(item):
        key_attrs = {"class", "style"}
        attrs = {
            k: v if v != [] else "" for k, v in item.attrs.items() if k in key_attrs
        }
        for attr in key_attrs:
            if attr not in attrs:
                attrs[attr] = ""

        # ★ 过滤哈希 class，只保留语义化稳定的 class
        if isinstance(attrs.get("class"), list):
            attrs["class"] = _stable_classes(attrs["class"])

        return attrs

    # ── 原版逻辑（基本不变）──────────────────────────────

    @staticmethod
    def _child_has_text(child, text, url, text_fuzz_ratio):
        child_text = child.getText().strip()
        if text_match(text, child_text, text_fuzz_ratio):
            parent_text = child.parent.getText().strip()
            if child_text == parent_text and child.parent.parent:
                return False
            child.wanted_attr = None
            return True

        if text_match(text, get_non_rec_text(child), text_fuzz_ratio):
            child.is_non_rec_text = True
            child.wanted_attr = None
            return True

        for key, value in child.attrs.items():
            if not isinstance(value, str):
                continue
            value = value.strip()
            if text_match(text, value, text_fuzz_ratio):
                child.wanted_attr = key
                return True
            if key in {"href", "src"}:
                full_url = urljoin(url, value)
                if text_match(text, full_url, text_fuzz_ratio):
                    child.wanted_attr = key
                    child.is_full_url = True
                    return True
        return False

    def _get_children(self, soup, text, url, text_fuzz_ratio):
        children = reversed(soup.findChildren())
        return [x for x in children if self._child_has_text(x, text, url, text_fuzz_ratio)]

    @classmethod
    def _build_stack(cls, child, url):
        content = [(child.name, cls._get_valid_attrs(child))]
        parent = child
        while True:
            grand_parent = parent.findParent()
            if not grand_parent:
                break
            children = grand_parent.findAll(parent.name, cls._get_valid_attrs(parent), recursive=False)
            for i, c in enumerate(children):
                if c == parent:
                    content.insert(0, (grand_parent.name, cls._get_valid_attrs(grand_parent), i))
                    break
            if not grand_parent.parent:
                break
            parent = grand_parent

        wanted_attr = getattr(child, "wanted_attr", None)
        is_full_url = getattr(child, "is_full_url", False)
        is_non_rec_text = getattr(child, "is_non_rec_text", False)
        stack = dict(
            content=content,
            wanted_attr=wanted_attr,
            is_full_url=is_full_url,
            is_non_rec_text=is_non_rec_text,
        )
        stack["url"] = url if is_full_url else ""
        stack["hash"] = hashlib.sha256(str(stack).encode("utf-8")).hexdigest()
        stack["stack_id"] = "rule_" + stack["hash"][:8]
        return stack

    def _get_result_for_child(self, child, soup, url):
        stack = self._build_stack(child, url)
        result = self._get_result_with_stack(stack, soup, url, 1.0)
        return result, stack

    @staticmethod
    def _fetch_result_from_child(child, wanted_attr, is_full_url, url, is_non_rec_text):
        if wanted_attr is None:
            if is_non_rec_text:
                return get_non_rec_text(child)
            return child.getText().strip()
        if wanted_attr not in child.attrs:
            return None
        if is_full_url:
            return urljoin(url, child.attrs[wanted_attr])
        return child.attrs[wanted_attr]

    @staticmethod
    def _get_fuzzy_attrs(attrs, attr_fuzz_ratio):
        attrs = dict(attrs)
        for key, val in attrs.items():
            if isinstance(val, str) and val:
                val = FuzzyText(val, attr_fuzz_ratio)
            elif isinstance(val, (list, tuple)):
                val = [FuzzyText(x, attr_fuzz_ratio) if x else x for x in val]
            attrs[key] = val
        return attrs

    def _get_result_with_stack(self, stack, soup, url, attr_fuzz_ratio, **kwargs):
        parents = [soup]
        stack_content = stack["content"]
        contain_sibling_leaves = kwargs.get("contain_sibling_leaves", False)

        for index, item in enumerate(stack_content):
            children = []
            if item[0] == "[document]":
                continue
            for parent in parents:
                attrs = item[1]
                if attr_fuzz_ratio < 1.0:
                    attrs = self._get_fuzzy_attrs(attrs, attr_fuzz_ratio)
                found = parent.findAll(item[0], attrs, recursive=False)
                if not found:
                    continue
                if not contain_sibling_leaves and index == len(stack_content) - 1:
                    idx = min(len(found) - 1, stack_content[index - 1][2])
                    found = [found[idx]]
                children += found
            parents = children

        wanted_attr = stack["wanted_attr"]
        is_full_url = stack["is_full_url"]
        is_non_rec_text = stack.get("is_non_rec_text", False)
        result = [
            ResultItem(
                self._fetch_result_from_child(i, wanted_attr, is_full_url, url, is_non_rec_text),
                getattr(i, "child_index", 0),
            )
            for i in parents
        ]
        if not kwargs.get("keep_blank", False):
            result = [x for x in result if x.text]
        return result

    def _get_result_with_stack_index_based(self, stack, soup, url, attr_fuzz_ratio, **kwargs):
        p = soup.findChildren(recursive=False)[0]
        stack_content = stack["content"]
        for index, item in enumerate(stack_content[:-1]):
            if item[0] == "[document]":
                continue
            content = stack_content[index + 1]
            attrs = content[1]
            if attr_fuzz_ratio < 1.0:
                attrs = self._get_fuzzy_attrs(attrs, attr_fuzz_ratio)
            p = p.findAll(content[0], attrs, recursive=False)
            if not p:
                return []
            idx = min(len(p) - 1, item[2])
            p = p[idx]

        result = [
            ResultItem(
                self._fetch_result_from_child(
                    p, stack["wanted_attr"], stack["is_full_url"], url, stack["is_non_rec_text"]
                ),
                getattr(p, "child_index", 0),
            )
        ]
        if not kwargs.get("keep_blank", False):
            result = [x for x in result if x.text]
        return result

    def _get_result_by_func(self, func, url, html, soup, request_args,
                            grouped, group_by_alias, unique, attr_fuzz_ratio, **kwargs):
        if not soup:
            soup = self._get_soup(url=url, html=html, request_args=request_args)

        keep_order = kwargs.get("keep_order", False)
        if group_by_alias or (keep_order and not grouped):
            for index, child in enumerate(soup.findChildren()):
                setattr(child, "child_index", index)

        result_list = []
        grouped_result = defaultdict(list)
        for stack in self.stack_list:
            if not url:
                url = stack.get("url", "")
            result = func(stack, soup, url, attr_fuzz_ratio, **kwargs)
            if not grouped and not group_by_alias:
                result_list += result
                continue
            group_id = stack.get("alias", "") if group_by_alias else stack["stack_id"]
            grouped_result[group_id] += result

        return self._clean_result(result_list, grouped_result, grouped, group_by_alias, unique, keep_order)

    @staticmethod
    def _clean_result(result_list, grouped_result, grouped, grouped_by_alias, unique, keep_order):
        if not grouped and not grouped_by_alias:
            if unique is None:
                unique = True
            if keep_order:
                result_list = sorted(result_list, key=lambda x: x.index)
            result = [x.text for x in result_list]
            if unique:
                result = unique_hashable(result)
            return result

        for k, val in grouped_result.items():
            if grouped_by_alias:
                val = sorted(val, key=lambda x: x.index)
            val = [x.text for x in val]
            if unique:
                val = unique_hashable(val)
            grouped_result[k] = val
        return dict(grouped_result)

    # ── 核心改动②：build() 失败时自动 fallback 到 ML ────

    def build(self, url=None, wanted_list=None, wanted_dict=None, html=None,
              request_args=None, update=False, text_fuzz_ratio=1.0, use_ml=True):
        """
        原版参数完全兼容。新增：
          use_ml: bool = True  - 规则模式返回空时自动 fallback 到随机森林
        """
        if not wanted_list and not (wanted_dict and any(wanted_dict.values())):
            raise ValueError("No targets were supplied")

        soup = self._get_soup(url=url, html=html, request_args=request_args)

        if update is False:
            self.stack_list = []

        result_list = []
        _wanted_dict = wanted_dict or {}
        if wanted_list:
            _wanted_dict = {"": wanted_list}

        _flat_wanted = []
        for alias, items in _wanted_dict.items():
            items = [normalize(w) for w in items]
            _flat_wanted += items
            for wanted in items:
                children = self._get_children(soup, wanted, url, text_fuzz_ratio)
                for child in children:
                    result, stack = self._get_result_for_child(child, soup, url)
                    stack["alias"] = alias
                    result_list += result
                    self.stack_list.append(stack)

        result_list = unique_hashable([item.text for item in result_list])
        self.stack_list = unique_stack_list(self.stack_list)

        # ★ 核心改动②：规则结果为空且 sklearn 可用时，启动 ML 训练
        if not result_list and use_ml and _ML_AVAILABLE:
            print("[AutoScraper] 规则模式未找到结果，切换到 ML 模式...")
            self._ml_active = True
            self._ml_build(soup, url, _wanted_dict, text_fuzz_ratio)
            # 返回 ML 在训练页的回放结果
            return self._ml_get_result(soup, url, threshold=0.3)

        self._ml_active = False
        return result_list

    # ── ML 内部方法 ──────────────────────────────────────

    @staticmethod
    def _node_signature(node) -> str:
        """节点结构签名：tag + 稳定class + 父tag，用于判断是否同类兄弟"""
        stable = tuple(sorted(_stable_classes(node.attrs.get('class', []))))
        parent_tag = node.parent.name if node.parent else ''
        return f"{node.name}|{stable}|{parent_tag}"

    def _expand_to_siblings(self, seed_nodes: list, all_nodes: list) -> set:
        """
        列表页核心：找出页面上所有结构相同的节点作为正样本。

        两层匹配策略：
        1. 严格模式：同一父节点下 tag+稳定class 相同（普通兄弟）
        2. 宽松模式：全页面 tag+稳定class+父tag 相同（列表项，如每个<li>里的<a>）
           额外要求：祖父节点的结构签名也相同，防止误匹配导航栏等无关节点
        """
        expanded = set()
        for seed in seed_nodes:
            sig = self._node_signature(seed)
            parent = seed.parent
            if parent is None:
                continue

            # 策略1：同一父节点下的直接兄弟
            same_parent_hits = set()
            for i, node in enumerate(all_nodes):
                if node.parent == parent and self._node_signature(node) == sig:
                    same_parent_hits.add(i)

            if len(same_parent_hits) > 1:
                # 有真实兄弟，直接用
                expanded |= same_parent_hits
            else:
                # 策略2：列表模式，父节点本身是可重复的（如<li>）
                # 要求父节点的结构签名也相同
                parent_sig = self._node_signature(parent)
                for i, node in enumerate(all_nodes):
                    if (self._node_signature(node) == sig
                            and node.parent is not None
                            and self._node_signature(node.parent) == parent_sig):
                        expanded.add(i)

        return expanded

    def _ml_build(self, soup, url, wanted_dict, fuzz_ratio=1.0):
        """训练随机森林分类器（支持列表页自动扩充同类兄弟节点）"""
        self._ml_classifiers = {}
        self._ml_vocabs = {}
        self._ml_n_positives = {}  # 记录每个 alias 的正样本数，用于单样本特殊处理

        all_nodes = [n for n in soup.find_all(True) if n.name]
        all_features = [_extract_node_features(n, soup) for n in all_nodes]

        for alias, targets in wanted_dict.items():
            targets = [normalize(t) for t in targets]
            seed_nodes = []  # 直接命中的种子节点

            for target in targets:
                for i, node in enumerate(all_nodes):
                    text = node.get_text(strip=True)
                    hit = (SequenceMatcher(None, target, text).ratio() >= fuzz_ratio
                           if fuzz_ratio < 1.0 else text == target)
                    if not hit:
                        for val in node.attrs.values():
                            if isinstance(val, str) and val.strip() == target:
                                hit = True
                                break
                    if hit:
                        # 只保留最小节点
                        is_anc = any(
                            node in list(sn.parents) for sn in seed_nodes
                        )
                        if not is_anc:
                            seed_nodes.append(node)

            if not seed_nodes:
                print(f"[ML] alias='{alias}' 未找到正样本，跳过")
                continue

            # ★ 列表页关键：把结构相同的所有兄弟节点都标为正样本
            positive_indices = self._expand_to_siblings(seed_nodes, all_nodes)

            # 也把种子本身加进去（防止无兄弟的情况）
            for sn in seed_nodes:
                if sn in all_nodes:
                    positive_indices.add(all_nodes.index(sn))

            print(f"[ML] alias='{alias}' 训练中，"
                  f"{len(seed_nodes)} 个种子 → 扩充到 {len(positive_indices)} 个正样本 "
                  f"/ {len(all_nodes)} 个节点")

            vocab = None
            vecs = []
            for feat in all_features:
                vec, vocab = _features_to_vector(feat, vocab)
                vecs.append(vec)

            X = np.array(vecs)
            y = np.array([1 if i in positive_indices else 0 for i in range(len(all_nodes))])

            clf = RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
            )
            clf.fit(X, y)

            self._ml_classifiers[alias] = clf
            self._ml_vocabs[alias] = vocab
            self._ml_n_positives[alias] = len(positive_indices)

    def _ml_get_result(self, soup, url=None, threshold=0.5, group_by_alias=False):
        """用 ML 分类器在 soup 上预测，自动去除父子重复节点"""
        all_nodes = [n for n in soup.find_all(True) if n.name]

        # 预先算好所有节点的特征向量（各 alias 共用）
        alias_results = {}

        for alias, clf in self._ml_classifiers.items():
            vocab = self._ml_vocabs[alias]
            vecs = []
            for node in all_nodes:
                vec, _ = _features_to_vector(_extract_node_features(node, soup), vocab)
                vecs.append(vec)

            X = np.array(vecs)
            expected = clf.n_features_in_
            if X.shape[1] < expected:
                X = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
            elif X.shape[1] > expected:
                X = X[:, :expected]

            proba = clf.predict_proba(X)
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1

            # ★ 单样本模式：训练时只有1个正样本，直接取概率最高的1个节点
            n_positive = self._ml_n_positives.get(alias, 0)
            if n_positive == 1:
                best_i = int(np.argmax(proba[:, pos_idx]))
                hit_indices = [best_i]
            else:
                hit_indices = [i for i in range(len(all_nodes))
                               if proba[i][pos_idx] >= threshold]

            hit_nodes = [all_nodes[i] for i in hit_indices]

            # 去除父子重复：只保留最小子节点
            def is_ancestor_hit(node):
                return any(node in list(h.parents) for h in hit_nodes if h != node)

            leaf_hits = [
                (all_nodes[i], proba[i][pos_idx])
                for i in hit_indices
                if not is_ancestor_hit(all_nodes[i])
            ]

            # 按 DOM 顺序排列
            leaf_hits.sort(key=lambda x: all_nodes.index(x[0]))

            seen = set()
            alias_results[alias] = []
            for node, _ in leaf_hits:
                text = node.get_text(strip=True)
                if text and text not in seen:
                    seen.add(text)
                    alias_results[alias].append(text)

        if group_by_alias:
            return alias_results

        # 合并所有 alias 结果（去重，保持顺序）
        all_results, seen = [], set()
        for items in alias_results.values():
            for text in items:
                if text not in seen:
                    seen.add(text)
                    all_results.append(text)
        return all_results

    # ── get_result_similar / exact / get_result（原版 + ML fallback）──

    def get_result_similar(self, url=None, html=None, soup=None, request_args=None,
                           grouped=False, group_by_alias=False, unique=None,
                           attr_fuzz_ratio=1.0, keep_blank=False, keep_order=False,
                           contain_sibling_leaves=False, ml_threshold=0.5):
        """
        原版参数完全兼容。新增：
          ml_threshold: float = 0.5  - ML 模式下的分类器概率阈值
        """
        # ★ 如果 build() 启用了 ML 模式，走 ML 预测
        if self._ml_active and self._ml_classifiers and _ML_AVAILABLE:
            if not soup:
                soup = self._get_soup(url=url, html=html, request_args=request_args)
            return self._ml_get_result(soup, url=url, threshold=ml_threshold,
                                       group_by_alias=group_by_alias)

        # 否则走原版规则路径
        func = self._get_result_with_stack
        return self._get_result_by_func(
            func, url, html, soup, request_args,
            grouped, group_by_alias, unique, attr_fuzz_ratio,
            keep_blank=keep_blank, keep_order=keep_order,
            contain_sibling_leaves=contain_sibling_leaves,
        )

    def get_result_exact(self, url=None, html=None, soup=None, request_args=None,
                         grouped=False, group_by_alias=False, unique=None,
                         attr_fuzz_ratio=1.0, keep_blank=False):
        func = self._get_result_with_stack_index_based
        return self._get_result_by_func(
            func, url, html, soup, request_args,
            grouped, group_by_alias, unique, attr_fuzz_ratio,
            keep_blank=keep_blank,
        )

    def get_result(self, url=None, html=None, request_args=None,
                   grouped=False, group_by_alias=False, unique=None, attr_fuzz_ratio=1.0):
        soup = self._get_soup(url=url, html=html, request_args=request_args)
        args = dict(url=url, soup=soup, grouped=grouped, group_by_alias=group_by_alias,
                    unique=unique, attr_fuzz_ratio=attr_fuzz_ratio)
        return self.get_result_similar(**args), self.get_result_exact(**args)

    # ── 规则管理（原版不变）──────────────────────────────

    def remove_rules(self, rules):
        self.stack_list = [x for x in self.stack_list if x["stack_id"] not in rules]

    def keep_rules(self, rules):
        self.stack_list = [x for x in self.stack_list if x["stack_id"] in rules]

    def set_rule_aliases(self, rule_aliases):
        id_to_stack = {stack["stack_id"]: stack for stack in self.stack_list}
        for rule_id, alias in rule_aliases.items():
            id_to_stack[rule_id]["alias"] = alias

    def generate_python_code(self):
        print("This function is deprecated. Please use save() and load() instead.")