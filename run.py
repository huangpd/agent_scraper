"""Agent Scraper 入口脚本"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from agent_scraper import AgentScraper
from agent_scraper.formatter import Formatter

instruction = """
步骤1: 打开网址 https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT
步骤2: 找到并点击 "Files and versions" 标签页
步骤3: 下滑页面到最底部，如果页面有 "Load more files" 请点击，直到按钮消失
步骤4: 遍历所有子子文件夹，
步骤5：加载全部文件，提取文件的文件名和下载URL，用json格式

样本数据:
{"file_name":".gitattributes","download_url":"https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT/resolve/main/.gitattributes"}
{"file_name":"README.md","download_url":"https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT/resolve/main/README.md"}
"""


async def main():
    scraper = AgentScraper(headless=False)
    result = await scraper.run(instruction)

    # 输出结果
    print(f"\n{'=' * 60}")
    print(f"提取到 {result.total_count} 个文件:")

    output = Formatter.to_json(result)
    print(output[:2000])  # 预览前2000字符

    # 保存到文件
    with open("scraped_result.json", "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\n结果已保存到 scraped_result.json")


if __name__ == "__main__":
    asyncio.run(main())
