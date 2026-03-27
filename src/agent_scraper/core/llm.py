"""统一 LLM 服务网关 (AOP 层)"""

import json
import os
import logging
import time
from typing import Any
from openai import AsyncOpenAI
from agent_scraper.core.trace import get_trace_id, increment_llm_count

logger = logging.getLogger(__name__)


def _fmt_json(obj) -> str:
    """安全格式化 JSON，非序列化对象降级为 str"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)

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
        
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})

        # 1. 完整输入日志
        logger.info(
            "[%s][#%s] ── LLM 输入 ──\n  model: %s\n  temperature: %s\n  messages:\n%s",
            caller, trace_id, self.model, temperature, _fmt_json(messages),
        )

        try:
            # 2. 调用转发
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
            )

            content = resp.choices[0].message.content.strip()
            duration = time.perf_counter() - start_time

            # 3. 完整输出日志
            logger.info(
                "[%s][#%s] ── LLM 输出 (%.2fs) ──\n%s",
                caller, trace_id, duration, content,
            )
            
            # 这里可以扩展 Token 统计逻辑
            # self._record_tokens(resp.usage)
            
            return content

        except Exception as e:
            logger.error("[%s][#%s] Failed (%.2fs): %s", caller, trace_id, time.perf_counter() - start_time, str(e))
            raise

    async def call_with_images(self, prompt: str, images: list[str], caller: str = "LLM") -> str:
        """多模态调用接口：文本 + 图片（OpenAI vision 格式）

        Args:
            prompt: 文本提示词
            images: 图片列表，每项为 base64 data URL 或 http(s) URL
            caller: 调用方标识（用于日志）
        """
        import base64
        import mimetypes

        trace_id = get_trace_id()
        increment_llm_count()
        start_time = time.perf_counter()

        # 构建 content 数组
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            if img.startswith("data:"):
                data_url = img
            elif img.startswith(("http://", "https://")):
                data_url = img
            else:
                # 本地文件路径 → base64 data URL
                mime = mimetypes.guess_type(img)[0] or "image/png"
                with open(img, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                data_url = f"data:{mime};base64,{b64}"
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            })

        messages = [{"role": "user", "content": content}]

        # 输入日志（图片只打 URL 前缀，不打完整 base64）
        log_content = []
        for part in content:
            if part["type"] == "text":
                log_content.append(part)
            else:
                url = part["image_url"]["url"]
                log_content.append({"type": "image_url", "image_url": {"url": url[:80] + "...", "detail": "high"}})
        logger.info(
            "[%s][#%s] ── Vision 输入 (%d images) ──\n  model: %s\n  content:\n%s",
            caller, trace_id, len(images), self.model, _fmt_json(log_content),
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
            )
            result = resp.choices[0].message.content.strip()
            duration = time.perf_counter() - start_time
            logger.info(
                "[%s][#%s] ── Vision 输出 (%.2fs) ──\n%s",
                caller, trace_id, duration, result,
            )
            return result
        except Exception as e:
            logger.error("[%s][#%s] VisionFailed (%.2fs): %s", caller, trace_id, time.perf_counter() - start_time, str(e))
            raise
