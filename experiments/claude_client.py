"""Thin wrapper around Anthropic Messages API.

Loads ANTHROPIC_API_KEY from `experiments/.env`. Designed for short prompts.
"""
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent / ".env")


DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")


class ClaudeClient:
    def __init__(self, model: Optional[str] = None,
                 max_tokens: int = 1024,
                 temperature: Optional[float] = None):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment / .env")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._temperature_supported = True

    @staticmethod
    def _extract_text(msg) -> str:
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""

    def complete(self, system: str, user: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 retries: int = 3) -> str:
        temp = temperature if temperature is not None else self.temperature
        cur_max = max_tokens or self.max_tokens
        last_msg = None
        for attempt in range(retries):
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=cur_max,
                    system=system or "You are a helpful assistant.",
                    messages=[{"role": "user", "content": user}],
                )
                if temp is not None and self._temperature_supported:
                    kwargs["temperature"] = temp
                msg = self.client.messages.create(**kwargs)
                last_msg = msg
                text = self._extract_text(msg)
                if text:
                    return text
                if getattr(msg, "stop_reason", "") == "max_tokens":
                    cur_max = min(cur_max * 4, 4096)
                    continue
                return ""
            except anthropic.BadRequestError as e:
                if "temperature" in str(e).lower() and self._temperature_supported:
                    self._temperature_supported = False
                    continue
                raise
            except (anthropic.RateLimitError, anthropic.APIStatusError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return self._extract_text(last_msg) if last_msg is not None else ""
