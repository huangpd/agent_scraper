"""测试 _stack_to_xpath 精准 XPath 生成"""

import pytest
from lxml import html as lxml_html

from agent_scraper.rule_learner.learner import AutoScraper, _is_stable_id


# ─────────────────────────────────────────────────────────
# _is_stable_id 单元测试
# ─────────────────────────────────────────────────────────

class TestIsStableId:
    """动态 ID 检测"""

    @pytest.mark.parametrize("id_str", [
        "main-content", "repo-files", "sidebar", "nav", "footer",
        "app", "root", "content-wrapper",
    ])
    def test_stable_ids(self, id_str):
        assert _is_stable_id(id_str) is True

    @pytest.mark.parametrize("id_str", [
        ":r1:",                    # React 生成
        ":r2:panel",               # React 生成
        "a-3f2b1c9d",              # CSS-module
        "el-12345678",             # 长数字序列
        "item-a3b2c1d4",           # 尾部 hex hash
        "abcdef1234abcd",          # 纯 hex hash
        "ember-123",               # 框架前缀
        "react-tooltip-45",        # 框架前缀
        "vue-component",           # 框架前缀
        "",                        # 空字符串
    ])
    def test_dynamic_ids(self, id_str):
        assert _is_stable_id(id_str) is False


# ─────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────

def _build_and_xpath(html, wanted_dict):
    """构建 scraper 并返回 XPath 规则字典"""
    scraper = AutoScraper()
    scraper.build(html=html, wanted_dict=wanted_dict)
    return scraper.get_result_xpath_rule()


def _xpath_extract(html, xpath):
    """用 lxml 执行 XPath 并返回文本列表"""
    tree = lxml_html.fromstring(html)
    nodes = tree.xpath(xpath)
    return [n.text_content().strip() if hasattr(n, "text_content") else n for n in nodes]


# ─────────────────────────────────────────────────────────
# 核心 XPath 生成测试
# ─────────────────────────────────────────────────────────

class TestXPathGeneration:
    """XPath 生成精度测试"""

    def test_simple_list_matches_all_items(self):
        """简单列表：给一个样本应匹配所有同类元素"""
        html = """
        <html><body>
        <div id="main-content">
          <ul class="news-list">
            <li class="item"><a href="/p/1">Title-A</a></li>
            <li class="item"><a href="/p/2">Title-B</a></li>
            <li class="item"><a href="/p/3">Title-C</a></li>
          </ul>
        </div>
        </body></html>
        """
        rules = _build_and_xpath(html, {"title": ["Title-A"]})
        assert rules is not None
        texts = _xpath_extract(html, rules["title"])
        assert texts == ["Title-A", "Title-B", "Title-C"]

    def test_href_extraction(self):
        """属性提取：从 href 中提取所有链接"""
        html = """
        <html><body>
        <ul class="links">
          <li><a href="/page/1">Link1</a></li>
          <li><a href="/page/2">Link2</a></li>
          <li><a href="/page/3">Link3</a></li>
        </ul>
        </body></html>
        """
        rules = _build_and_xpath(html, {"url": ["/page/1"]})
        assert rules is not None
        hrefs = _xpath_extract(html, rules["url"])
        assert hrefs == ["/page/1", "/page/2", "/page/3"]

    def test_multi_container_disambiguation(self):
        """多容器歧义：只匹配目标容器，不跨容器"""
        html = """
        <html><body>
        <div class="sidebar">
          <ul class="links">
            <li class="item">Nav1</li>
            <li class="item">Nav2</li>
          </ul>
        </div>
        <div class="main">
          <ul class="links">
            <li class="item">Article-A</li>
            <li class="item">Article-B</li>
            <li class="item">Article-C</li>
          </ul>
        </div>
        </body></html>
        """
        rules = _build_and_xpath(html, {"article": ["Article-A"]})
        texts = _xpath_extract(html, rules["article"])
        assert "Nav1" not in texts
        assert "Nav2" not in texts
        assert len(texts) == 3

    def test_id_anchor(self):
        """ID 锚点：利用稳定 ID 缩短路径"""
        html = """
        <html><body>
        <div id="repo-content">
          <table>
            <tr><td class="name">file1.txt</td><td>100KB</td></tr>
            <tr><td class="name">file2.txt</td><td>200KB</td></tr>
            <tr><td class="name">file3.txt</td><td>300KB</td></tr>
          </table>
        </div>
        </body></html>
        """
        rules = _build_and_xpath(html, {"file": ["file1.txt"]})
        xpath = rules["file"]
        # 应使用 ID 作为锚点
        assert "@id='repo-content'" in xpath
        texts = _xpath_extract(html, xpath)
        assert texts == ["file1.txt", "file2.txt", "file3.txt"]

    def test_deep_nested_cards(self):
        """深层嵌套：卡片内的标题仍能匹配全部"""
        html = """
        <html><body>
        <div class="container">
          <ul class="card-list">
            <li class="card"><div class="body"><h3 class="title">Card1</h3></div></li>
            <li class="card"><div class="body"><h3 class="title">Card2</h3></div></li>
            <li class="card"><div class="body"><h3 class="title">Card3</h3></div></li>
          </ul>
        </div>
        </body></html>
        """
        rules = _build_and_xpath(html, {"card_title": ["Card1"]})
        texts = _xpath_extract(html, rules["card_title"])
        assert texts == ["Card1", "Card2", "Card3"]

    def test_dynamic_id_not_used(self):
        """动态 ID 过滤：React/框架生成的 ID 不应出现在 XPath 中"""
        html = """
        <html><body>
        <div id=":r1:react-panel">
          <span class="label">Price-100</span>
          <span class="label">Price-200</span>
        </div>
        </body></html>
        """
        rules = _build_and_xpath(html, {"price": ["Price-100"]})
        xpath = rules["price"]
        assert ":r1:" not in xpath
        texts = _xpath_extract(html, xpath)
        assert texts == ["Price-100", "Price-200"]

    def test_multiple_classes_all_used(self):
        """多 class 精度：所有稳定 class 都应参与匹配"""
        html = """
        <html><body>
        <div class="feed primary">
          <div class="post featured"><span>Post1</span></div>
          <div class="post featured"><span>Post2</span></div>
        </div>
        <div class="feed secondary">
          <div class="post"><span>Other</span></div>
        </div>
        </body></html>
        """
        rules = _build_and_xpath(html, {"post": ["Post1"]})
        xpath = rules["post"]
        # 应包含 feed 和 primary 两个 class
        assert "feed" in xpath
        assert "primary" in xpath
        texts = _xpath_extract(html, xpath)
        assert "Other" not in texts

    def test_word_boundary_class_matching(self):
        """词边界：class='item' 不应匹配 class='item-extra'"""
        html = """
        <html><body>
        <ul>
          <li class="item">Target1</li>
          <li class="item">Target2</li>
          <li class="item-extra">Noise</li>
        </ul>
        </body></html>
        """
        rules = _build_and_xpath(html, {"data": ["Target1"]})
        xpath = rules["data"]
        # 应使用词边界匹配
        assert "concat" in xpath
        texts = _xpath_extract(html, xpath)
        assert "Noise" not in texts
        assert len(texts) == 2

    def test_table_rows(self):
        """表格：匹配所有行的指定列"""
        html = """
        <html><body>
        <table class="data-table">
          <tr><td class="col-name">Alice</td><td class="col-age">30</td></tr>
          <tr><td class="col-name">Bob</td><td class="col-age">25</td></tr>
          <tr><td class="col-name">Carol</td><td class="col-age">28</td></tr>
        </table>
        </body></html>
        """
        rules = _build_and_xpath(html, {"name": ["Alice"]})
        texts = _xpath_extract(html, rules["name"])
        assert texts == ["Alice", "Bob", "Carol"]

    def test_empty_stack_returns_empty(self):
        """空 stack 返回空 dict 而非 None"""
        scraper = AutoScraper()
        assert scraper.get_result_xpath_rule() == {}

    def test_get_result_similar_empty_graceful(self):
        """get_result_similar 在无规则时不崩溃"""
        scraper = AutoScraper()
        result = scraper.get_result_similar(html="<html><body></body></html>")
        assert result == []

    def test_wanted_dict_multiple_aliases(self):
        """多 alias：每个字段独立生成 XPath"""
        html = """
        <html><body>
        <table>
          <tr>
            <td class="name">Alice</td>
            <td class="score">95</td>
          </tr>
          <tr>
            <td class="name">Bob</td>
            <td class="score">87</td>
          </tr>
        </table>
        </body></html>
        """
        rules = _build_and_xpath(html, {
            "name": ["Alice"],
            "score": ["95"],
        })
        assert "name" in rules
        assert "score" in rules
        names = _xpath_extract(html, rules["name"])
        scores = _xpath_extract(html, rules["score"])
        assert names == ["Alice", "Bob"]
        assert scores == ["95", "87"]
