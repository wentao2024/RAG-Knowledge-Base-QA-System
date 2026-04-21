"""
LLM 客户端(OpenAI 接口（兼容 gpt-4o / gpt-4o-mini / DeepSeek / Qwen 等）
重试策略：只重试网络/限流错误，认证/参数错误直接抛出不重试
"""
from typing import List, AsyncIterator
from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)
from app.config import settings
from app.models.schemas import Message


# ── 判断哪些异常值得重试 ────────────────────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """
    只重试：网络错误、限流(429)、服务端 5xx
    不重试：认证(401)、参数错误(400)、模型不存在(404)
    """
    cls_name = type(exc).__name__
    NON_RETRYABLE = {
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
        "UnprocessableEntityError",
        "InvalidRequestError",
    }
    return not any(n in cls_name for n in NON_RETRYABLE)


_RETRY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception(_is_retryable),
    reraise=True,  # 直接抛出原始异常，不包装成 RetryError
)


# ── LLM 客户端 ─────────────────────────────────────────────────────────────────

class LLMClient:

    def __init__(self):
        key = settings.openai_api_key
        if not key:
            raise ValueError(
                " OPENAI_API_KEY 未配置！请在 .env 文件中填入你的 OpenAI API Key。"
            )
        self.model = settings.llm_model
        self.client = AsyncOpenAI(
            api_key=key,
            base_url=settings.openai_base_url,
        )
        logger.info(f"LLM 初始化: {settings.openai_base_url} / {self.model}")

    # ─── 单次补全 ──────────────────────────────────────────────────────────────

    @retry(**_RETRY)
    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    # ─── 多轮对话 ──────────────────────────────────────────────────────────────

    @retry(**_RETRY)
    async def chat(
        self,
        system: str,
        messages: List[Message],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        msg_list = [{"role": m.role.value, "content": m.content} for m in messages]
        resp = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "system", "content": system}] + msg_list,
        )
        return resp.choices[0].message.content

    # ─── 流式输出 ──────────────────────────────────────────────────────────────

    async def stream_chat(
        self,
        system: str,
        messages: List[Message],
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        msg_list = [{"role": m.role.value, "content": m.content} for m in messages]
        stream = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}] + msg_list,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
