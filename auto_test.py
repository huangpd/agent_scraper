from playwright.sync_api import sync_playwright
from autoscraper import AutoScraper

def get_rendered_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')
        html = page.content()
        browser.close()
        return html

# ── 训练：用 autoscraper repo 作为样本 ──
train_url = 'https://github.com/alirezamika/autoscraper'
train_html = get_rendered_html(train_url)

scraper = AutoScraper()
scraper.build(
    url=train_url,
    html=train_html,
    wanted_dict={
        'description': ['A Smart, Automatic, Fast and Lightweight Web Scraper for Python'],
        'stars': ['7.1k'],
        'forks': ['719'],
    }
)

# ── 预测：换成 tutorials repo ──
target_url = 'https://github.com/alirezamika/tutorials'
target_html = get_rendered_html(target_url)

result = scraper.get_result_similar(url=target_url, html=target_html,ml_threshold=0.2,group_by_alias=True,)
print(result)
