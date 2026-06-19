"""
LLM client using the OpenAI API (compatible with gpt-4o / gpt-4o-mini / DeepSeek / Qwen, etc.)
Retry strategy: only retry network/rate-limit errors; authentication/parameter errors are raised immediately.
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


# ── Determine which exceptions are worth retrying ──────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    """
    Retry only: network errors, rate limits (429), server 5xx.
    Do not retry: authentication (401), bad request (400), model not found (404).
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
    reraise=True,  # raise the original exception, not a RetryError wrapper
)


# ── LLM client ─────────────────────────────────────────────────────────────────

class LLMClient:

    def __init__(self):
        key = settings.openai_api_key
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Please add your OpenAI API key to the .env file."
            )
        self.model = settings.llm_model
        self.client = AsyncOpenAI(
            api_key=key,
            base_url=settings.openai_base_url,
        )
        logger.info(f"LLM initialised: {settings.openai_base_url} / {self.model}")

    # ─── Single completion ─────────────────────────────────────────────────────

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

    # ─── Multi-turn chat ───────────────────────────────────────────────────────

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

    # ─── Streaming output ──────────────────────────────────────────────────────

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
