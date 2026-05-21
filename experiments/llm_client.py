"""Shared CLI and factory for Claude / OpenAI judge and generation backends."""
from __future__ import annotations

import argparse
from typing import Any, Optional, Union

from claude_client import ClaudeClient
from openai_client import OpenAIClient


def add_llm_cli(ap: argparse.ArgumentParser, *, default_provider: str = "claude") -> None:
    """Register ``--llm-provider``, model override, and API key flags on *ap*."""
    ap.add_argument(
        "--llm-provider",
        dest="llm_provider",
        choices=("claude", "openai"),
        default=default_provider,
        help="LLM backend for judges in generate_dataset.py, eval_pipeline.py, "
        "eval_rl.py, eval_base.py, and train_rl.py (default: %(default)s).",
    )
    ap.add_argument(
        "--llm-model",
        dest="llm_model",
        default="",
        help="Model id override (else CLAUDE_MODEL / OPENAI_MODEL env or provider default).",
    )
    ap.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        default="",
        help="OpenAI API key (else OPENAI_API_KEY in .env / environment).",
    )
    ap.add_argument(
        "--anthropic-api-key",
        dest="anthropic_api_key",
        default="",
        help="Anthropic API key (else ANTHROPIC_API_KEY in .env / environment).",
    )


def make_llm_client(
    provider: str,
    *,
    model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: Optional[float] = None,
) -> Union[ClaudeClient, OpenAIClient]:
    """Build a client with ``complete(system, user, ...)`` for judges / generation."""
    p = (provider or "claude").strip().lower()
    m = (model or "").strip() or None
    if p == "openai":
        return OpenAIClient(
            model=m,
            api_key=(openai_api_key or "").strip() or None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if p == "claude":
        return ClaudeClient(
            model=m,
            api_key=(anthropic_api_key or "").strip() or None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise ValueError(f"Unknown llm provider: {provider!r} (use claude or openai)")


def llm_client_from_args(args: Any) -> Union[ClaudeClient, OpenAIClient]:
    """Instantiate client from argparse namespace (needs ``add_llm_cli`` fields)."""
    return make_llm_client(
        getattr(args, "llm_provider", "claude"),
        model=getattr(args, "llm_model", "") or None,
        openai_api_key=getattr(args, "openai_api_key", "") or None,
        anthropic_api_key=getattr(args, "anthropic_api_key", "") or None,
    )


def describe_llm_client(client: Union[ClaudeClient, OpenAIClient]) -> str:
    provider = "openai" if isinstance(client, OpenAIClient) else "claude"
    return f"{provider} model={client.model}"
