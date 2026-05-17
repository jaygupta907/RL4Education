"""Shared helpers for <reasoning>...</reasoning> blocks in dataset / SFT / eval."""
from __future__ import annotations

import re

_REASONING_RE = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


def strip_reasoning_prefix(text: str) -> str:
    """Remove the first <reasoning>...</reasoning> block; return trimmed remainder.

    If no block is found, returns ``text.strip()`` unchanged. Used after
    model generation so Claude judges only see the word problem.
    """
    if not text:
        return ""
    m = _REASONING_RE.search(text)
    if not m:
        return text.strip()
    return (text[: m.start()] + text[m.end() :]).strip()
