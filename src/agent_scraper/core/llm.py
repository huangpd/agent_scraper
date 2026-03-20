"""统一 LLM 服务网关 (AOP 层)"""

import os
import logging
import time
from typing import Any
from openai import AsyncOpenAI
from agent_scraper.core.trace import get_trace_id, increment_llm_count

logger = logging.getLogger(__name__)

def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "gpt-4o")

class LLMService:
    """LLM 服务网关：处理 Logging, Tracing, Retries, Costing"""
    
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = get_model_name()

    async def call(self, prompt: str, system_msg: str = "", temperature: float = 0.0, caller: str = "LLM") -> str:
        """统一调用接口"""
        trace_id = get_trace_id()
        increment_llm_count() # 自动累加计数
        start_time = time.perf_counter()
        
        # 1. 结构化 Prompt 日志 (Trace ID 绑定)
        logger.info("[%s][#%s] Prompt: %s...", caller, trace_id, prompt[:100].replace("\n", " "))
        
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})

        try:
            # 2. 调用转发
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
            )
            
            content = resp.choices[0].message.content.strip()
            duration = time.perf_counter() - start_time
            
            # 3. 结果日志
            logger.info("[%s][#%s] Success (%.2fs): %s...", caller, trace_id, duration, content[:10000].replace("\n", " "))
            
            # 这里可以扩展 Token 统计逻辑
            # self._record_tokens(resp.usage)
            
            return content

        except Exception as e:
            logger.error("[%s][#%s] Failed (%.2fs): %s", caller, trace_id, time.perf_counter() - start_time, str(e))
            raise
