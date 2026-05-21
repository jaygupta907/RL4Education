"""Thin wrapper around OpenAI Chat Completions API.

Loads ``OPENAI_API_KEY`` from ``experiments/.env``. Same ``complete(system, user)``
interface as ``ClaudeClient`` for judges and dataset generation.
"""
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")


def _prefers_max_completion_tokens(model: str) -> bool:
    """GPT-5 / o-series chat models use ``max_completion_tokens`` not ``max_tokens``."""
    m = (model or "").lower()
    return any(
        m.startswith(p) or p in m
        for p in ("gpt-5", "chatgpt-5", "o1", "o3", "o4")
    )


class OpenAIClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
    ):
        key = (api_key or "").strip() or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set (use --openai-api-key, .env, or environment)"
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package required: pip install openai") from e
        self.client = OpenAI(api_key=key)
        self.model = (model or "").strip() or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._temperature_supported = True
        self._use_max_completion_tokens = _prefers_max_completion_tokens(self.model)

    @staticmethod
    def _extract_text(msg) -> str:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)
        return ""

    def _create_kwargs(
        self,
        system: str,
        user: str,
        cur_max: int,
        temp: Optional[float],
    ) -> dict:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": user},
            ],
        )
        if self._use_max_completion_tokens:
            kwargs["max_completion_tokens"] = cur_max
        else:
            kwargs["max_tokens"] = cur_max
        if temp is not None and self._temperature_supported:
            kwargs["temperature"] = temp
        return kwargs

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        retries: int = 3,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        cur_max = max_tokens or self.max_tokens
        last = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    **self._create_kwargs(system, user, cur_max, temp)
                )
                last = resp
                text = self._extract_text(resp.choices[0].message)
                if text:
                    return text
                if getattr(resp.choices[0], "finish_reason", "") == "length":
                    cur_max = min(cur_max * 4, 16384)
                    continue
                return ""
            except Exception as e:
                err = str(e).lower()
                if "max_completion_tokens" in err and "max_tokens" in err:
                    self._use_max_completion_tokens = True
                    continue
                if "temperature" in err and self._temperature_supported:
                    self._temperature_supported = False
                    continue
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        if last is not None and last.choices:
            return self._extract_text(last.choices[0].message)
        return ""
