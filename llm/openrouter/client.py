import os
import asyncio
import itertools
import logging
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class APIError(Exception):
    def __init__(self, status: int, body: dict):
        self.status = status
        self.body = body
        super().__init__(f"OpenRouter API error {status}: {body}")


class OpenRouterClient:
    """Async OpenRouter client with multi-key rotation and concurrency control.

    Keys are read from OPENROUTER_API_KEYS (comma-separated) with fallback
    to OPENROUTER_API_KEY (single key). Keys are rotated round-robin across
    requests so rate limits are distributed.

    Usage:
        async with OpenRouterClient() as client:
            resp = await client.complete([{"role": "user", "content": "Hi"}])

            # Parallel batch
            results = await client.complete_batch([
                [{"role": "user", "content": "Q1"}],
                [{"role": "user", "content": "Q2"}],
            ])
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
        base_url: str = OPENROUTER_BASE_URL,
    ):
        if keys is None:
            raw = os.getenv(
                "OPENROUTER_API_KEYS", os.getenv("OPENROUTER_API_KEY", "")
            )
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                "No API keys found. Set OPENROUTER_API_KEYS or OPENROUTER_API_KEY."
            )

        self._keys = keys
        self._key_cycle = itertools.cycle(keys)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_retries = max_retries
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    # -- lifecycle --

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any):
        await self.close()

    # -- key rotation --

    def _next_key(self) -> str:
        return next(self._key_cycle)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    # -- public API --

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send a chat completion request, respecting the concurrency semaphore."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools

        async with self._semaphore:
            return await self._request_with_retry(payload)

    async def complete_batch(
        self,
        message_batches: list[list[dict[str, str]]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[dict | APIError]:
        """Run many completions concurrently. Failed requests return APIError."""
        tasks = [
            self.complete(
                messages=msgs,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                **kwargs,
            )
            for msgs in message_batches
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # -- internals --

    async def _request_with_retry(self, payload: dict) -> dict:
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            key = self._next_key()
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }

            try:
                session = await self._get_session()
                async with session.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    body = await resp.json()

                    if resp.status == 200:
                        return body

                    if resp.status == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", 2**attempt)
                        )
                        logger.warning(
                            "Rate limited (key ...%s), retry in %.1fs",
                            key[-6:],
                            retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status >= 500:
                        logger.warning(
                            "Server error %d, attempt %d/%d",
                            resp.status,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(2**attempt)
                        continue

                    # 4xx (not 429) — don't retry
                    raise APIError(resp.status, body)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(
                    "Request failed: %s, attempt %d/%d",
                    e,
                    attempt + 1,
                    self._max_retries,
                )
                await asyncio.sleep(2**attempt)

        raise APIError(
            0,
            {
                "error": f"All {self._max_retries} retries exhausted",
                "last_error": str(last_error),
            },
        )
