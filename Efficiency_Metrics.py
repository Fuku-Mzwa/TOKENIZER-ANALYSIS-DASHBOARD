"""Efficiency metrics: token count, fertility, compression, token premium."""
from __future__ import annotations

from typing import Dict, List, Optional


def token_count(tokens) -> int:
    """Number of tokens produced."""
    return len(tokens)


def fertility(text: str, tokens) -> float:
    """Average number of tokens per whitespace-separated word."""
    n_tokens = len(tokens)
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    return n_tokens / word_count


def compression(text: str, tokens) -> float:
    """Bytes (UTF-8) per token — higher means denser packing."""
    n_tokens = len(tokens)
    bytes_count = len(text.encode("utf-8"))
    if n_tokens == 0:
        return 0.0
    return bytes_count / n_tokens


def token_premium(tokens, english_tokens) -> float:
    """
    tokens(lang) / tokens(English) on parallel text.
    > 1 means the language is more expensive than English.
    """
    n = len(tokens)
    n_eng = len(english_tokens)
    if n_eng == 0:
        return 0.0
    return n / n_eng


def compute_all(
    text: str,
    tokens: List[str],
    english_tokens: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Convenience wrapper used by the UI."""
    result = {
        "token_count": token_count(tokens),
        "word_count": len(text.split()) if text.strip() else 0,
        "fertility": fertility(text, tokens),
        "compression": compression(text, tokens),
    }
    if english_tokens is not None:
        result["token_premium"] = token_premium(tokens, english_tokens)
    return result
