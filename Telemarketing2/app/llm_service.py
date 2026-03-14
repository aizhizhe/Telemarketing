from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import OpenAI

from .config import get_settings


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        timeout=45.0,
        max_retries=0,
    )


class LlmService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_client()

    def chat(
        self,
        *,
        stage: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        model: str | None = None,
    ) -> dict[str, Any]:
        chosen_model = model or self.settings.chat_model
        response = self.client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=45,
        )
        content = response.choices[0].message.content or ""
        return {
            "stage": stage,
            "model": chosen_model,
            "request": messages,
            "response_text": content,
        }
