"""统一 LLM 服务网关 (AOP 层) — 基于 LiteLLM"""

import base64
import json
import mimetypes
import os
import logging
import time

import litellm
from agent_scraper.core.trace import get_trace_id, increment_llm_count

logger = logging.getLogger(__name__)

# 降低 LiteLLM 自身的日志噪音
litellm.suppress_debug_info = True
litellm.drop_params = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)


def _fmt_json(obj) -> str:
    """安全格式化 JSON，非序列化对象降级为 str"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def get_model_name() -> str:
    """默认模型（轻量调用：任务解析、评估等）"""
    return os.getenv("MODEL_NAME", "gpt-4o")


def get_strong_model_name() -> str:
    """强模型（关键调用：CSS 选择器提取、规则发现）"""
    return os.getenv("STRONG_MODEL_NAME", os.getenv("MODEL_NAME", "gpt-4o"))


class LLMService:
    """LLM 服务网关：Logging, Tracing, Model Routing"""

    def __init__(self):
        self.default_model = get_model_name()
        # LiteLLM 通过环境变量自动读取 API key
        # 如果用户配置了 OPENAI_BASE_URL，设置为 LiteLLM api_base
        api_base = os.getenv("OPENAI_BASE_URL")
        if api_base:
            litellm.api_base = api_base

    async def call(
        self,
        prompt: str,
        system_msg: str = "",
        temperature: float = 0.0,
        caller: str = "LLM",
        model: str | None = None,
    ) -> str:
        """单条 prompt 调用接口（向后兼容），内部委托 call_messages"""
        messages = [{"role": "user", "content": prompt}]
        return await self.call_messages(
            messages, system_msg=system_msg, temperature=temperature,
            caller=caller, model=model,
        )

    async def call_messages(
        self,
        messages: list[dict],
        system_msg: str = "",
        temperature: float = 0.0,
        caller: str = "LLM",
        model: str | None = None,
    ) -> str:
        """多消息调用接口（支持 few-shot 对话），model 参数可覆盖默认模型"""
        use_model = model or self.default_model
        trace_id = get_trace_id()
        increment_llm_count()
        start_time = time.perf_counter()

        full_messages = []
        if system_msg:
            full_messages.append({"role": "system", "content": system_msg})
        full_messages.extend(messages)

        # 输入日志
        logger.info(
            "[%s][#%s] ── LLM 输入 ──\n  model: %s\n  temperature: %s\n  messages: %d 条\n%s",
            caller, trace_id, use_model, temperature, len(full_messages),
            _fmt_json(full_messages),
        )

        try:
            resp = await litellm.acompletion(
                model=use_model,
                temperature=temperature,
                messages=full_messages,
            )

            content = resp.choices[0].message.content.strip()
            duration = time.perf_counter() - start_time

            logger.info(
                "[%s][#%s] ── LLM 输出 (%.2fs) ──\n%s",
                caller, trace_id, duration, content,
            )
            return content

        except Exception as e:
            logger.error(
                "[%s][#%s] Failed (%.2fs): %s",
                caller, trace_id, time.perf_counter() - start_time, str(e),
            )
            raise

    async def call_with_images(
        self,
        prompt: str,
        images: list[str],
        caller: str = "LLM",
        model: str | None = None,
    ) -> str:
        """多模态调用接口：文本 + 图片"""
        use_model = model or self.default_model
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

        # 输入日志（图片只打 URL 前缀）
        log_content = []
        for part in content:
            if part["type"] == "text":
                log_content.append(part)
            else:
                url = part["image_url"]["url"]
                log_content.append({"type": "image_url", "image_url": {"url": url[:80] + "..."}})
        logger.info(
            "[%s][#%s] ── Vision 输入 (%d images) ──\n  model: %s\n  content:\n%s",
            caller, trace_id, len(images), use_model, _fmt_json(log_content),
        )

        try:
            resp = await litellm.acompletion(
                model=use_model,
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
            logger.error(
                "[%s][#%s] VisionFailed (%.2fs): %s",
                caller, trace_id, time.perf_counter() - start_time, str(e),
            )
            raise
