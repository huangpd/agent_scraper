"""Agent Scraper 入口脚本"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

# 确保 browser-use 的日志完整输出到终端
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

from agent_scraper import AgentScraper
from agent_scraper.extraction.formatter import Formatter

instruction = """
步骤1: 打开网址 https://huggingface.co/baichuan-inc/Baichuan-M3-235B-FP8
步骤2: 找到并点击 "Files and versions" 标签页
步骤3: 下滑页面到最底部，如果页面有 "Load more files" 请点击，直到按钮消失
步骤4: 遍历所有子文件夹，
步骤5：提取页面的文件名和下载URL，用json格式

样本数据:
{"file_name":".gitattributes","download_url":"/baichuan-inc/Baichuan-M3-235B-FP8/blob/main/.gitattributes"}
{"file_name":"README.md","download_url":"/baichuan-inc/Baichuan-M3-235B-FP8/blob/main/README.md"}
"""
#
instruction = """
步骤1: 打开网址 https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a
步骤2: 再点击"Login"
步骤3: 等待输入 邮箱："limengzhu@fudan.edu.cn"
步骤3: 等待输入 密码："Aliyun0611"
步骤4: 点击"Sign in"
步骤5: 点击左上角"Download"[参考截图]，点击"COPY"下载链接的URL，保存为JSON
"""
instruction = """
步骤1: 打开网址 https://openneuro.org/datasets/ds005511/versions/1.0.1
步骤2: 提取文件的文件名，用json格式

# 样本数据:
{"file_name": "dataset_description.json"}
{"file_name": "README.md"}

# """


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
