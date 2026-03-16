from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI
from pydantic import BaseModel
import asyncio
import os


class Repo(BaseModel):
    url: str
    stars: str
    forks: str


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
        打开 https://github.com/trending
        获取当前页面的所有项目：
        - 项目URL
        - stars
        - forks
        """,
        llm=llm,
        browser=browser,
        output_model_schema=Posts   # 关键
    )

    result = await agent.run()

    print(result.model_json_schema())


if __name__ == "__main__":
    asyncio.run(main())