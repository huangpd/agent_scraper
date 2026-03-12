"""共享 LLM 客户端工厂"""

import os

from openai import AsyncOpenAI


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "gpt-4o")


def create_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
