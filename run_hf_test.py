from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI
from pydantic import BaseModel
import asyncio
import os


class Repo(BaseModel):
    url: str
    title: str


class Posts(BaseModel):
    posts: list[Repo]


async def main():

    browser = Browser()

    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        temperature=0,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


    agent = Agent(
        task="""
        步骤1: 打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html
        步骤2: 获取列表页URL和title
        步骤3: 点击"下一页"链接，获取前3页数据
        获取当前页面的所有项目：
        - URL:文章链接
        - title:文章标题
        """,
        llm=llm,
        browser=browser,
        output_model_schema=Posts   # 关键
    )

    result = await agent.run()

    print(result.model_json_schema())


if __name__ == "__main__":
    asyncio.run(main())