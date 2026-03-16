
import os
import sys
import json

# 确保能找到本地 src 目录
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from autoscraper.auto_scraper import AutoScraper

# 模拟真实的 HTML 结构，带有 dataList 容器
DATA_LIST_HTML = """
<html>
<body>
    <div id="wrapper">
        <ul class="dataList">
            <li class="item-abc">
                <h3><a href="http://example.com/1" class="link-123">专人守护、一树一策 黄山多措并举保护古松树</a></h3>
            </li>
            <li class="item-def">
                <h3><a href="http://example.com/2" class="link-456">黄山：以营商“软实力”夯实发展“硬支撑”</a></h3>
            </li>
        </ul>
    </div>
</body>
</html>
"""

def test_pure_ml_xpath_derivation():
    """
    测试纯 ML (AutoScraper) 反推 XPath 的效果。
    不使用 LLM，通过 DOM 树结构学习生成高精度路径。
    """
    print("开始纯 ML XPath 反推测试...")
    
    wanted_dict = {
        "title": ["专人守护、一树一策 黄山多措并举保护古松树"],
        "url": ["http://example.com/1"],
    }

    scraper = AutoScraper()
    
    # 1. 运行 ML 反推
    # 这里 build_xpath 会内部调用 build() 学习 stack，然后转化为 XPath
    rules = scraper.build_xpath(html=DATA_LIST_HTML, wanted_dict=wanted_dict)
    
    print("\n--- ML 生成的 XPath 规则 ---")
    print(json.dumps(rules, indent=2, ensure_ascii=False))

    # 2. 核心验证
    # 验证是否识别出了 dataList 容器锚点，且过滤了 item-abc 这种哈希 class
    title_xpath = rules.get("title", "")
    url_xpath = rules.get("url", "")
    
    print(f"\nTitle XPath: {title_xpath}")
    print(f"URL XPath: {url_xpath}")

    # 验证逻辑：
    # 1. 应该包含锚点 dataList
    assert "dataList" in title_xpath, "未识别出 dataList 稳定锚点"
    # 2. 应该保持层级关系 h3/a
    assert "h3/a" in title_xpath, "层级关系不完整"
    # 3. 应该排除了 item-abc 这种带哈希的随机 class
    assert "item-abc" not in title_xpath, "未过滤随机类名"
    # 4. URL 应该包含属性提取 /@href
    assert "/@href" in url_xpath, "未识别出属性提取"

    # 3. 验证提取准确性
    from lxml import html
    tree = html.fromstring(DATA_LIST_HTML)
    
    titles = tree.xpath(title_xpath)
    urls = tree.xpath(url_xpath)
    
    print(f"\n提取出的 Title (共{len(titles)}条): {titles}")
    print(f"提取出的 URL (共{len(urls)}条): {urls}")

    assert len(titles) == 2, "提取数量不对"
    assert titles[0] == "专人守护、一树一策 黄山多措并举保护古松树"
    assert urls[1] == "http://example.com/2"
    
    print("\n[SUCCESS] 纯 ML 高精度 XPath 反推测试通过！")

if __name__ == "__main__":
    test_pure_ml_xpath_derivation()
