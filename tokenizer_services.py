"""Hugging Face tokenizer loading and tokenization utilities."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

TOKENIZER_CACHE: Dict[str, Any] = {}

# Common subword markers to strip when measuring surface character length
_MARKER_PREFIXES = ("##", "▁", "Ġ", "Ċ", "\u0120")


def load_tokenizer(model_id: str, use_fast: bool = True):
    """Load a Hugging Face tokenizer by model ID (cached)."""
    if model_id is None or not str(model_id).strip():
        raise ValueError("Tokenizer identifier cannot be empty.")

    model_id = str(model_id).strip()
    if model_id in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[model_id]
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast,
        trust_remote_code=True,
    )
    TOKENIZER_CACHE[model_id] = tokenizer
    logger.info("Loaded tokenizer: %s", model_id)
    return tokenizer



def clear_tokenizer_cache() -> None:
    TOKENIZER_CACHE.clear()


def tokenize_text(tokenizer, text: str) -> Dict[str, Any]:
    """Tokenize text; return tokens, ids, and basic counts."""
    if text is None:
        text = ""
    text = str(text)

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = encoded["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "tokens": tokens,
        "ids": ids,
        "token_count": len(tokens),
        "char_count": len(text),
        "byte_count": len(text.encode("utf-8")),
    }


def clean_token_surface(token: str) -> str:
    """Strip common subword markers so len() reflects surface characters."""
    if token is None:
        return ""
    t = str(token)
    for prefix in _MARKER_PREFIXES:
        if t.startswith(prefix):
            t = t[len(prefix) :]
    # Byte-fallback tokens like <0xE2> → approximate as 1 char
    if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t):
        return "?"
    return t


def get_token_boundary_positions(tokens: List[str], word: str) -> set:
    """
    Map tokenizer pieces onto character offsets inside `word`.

    Returns the set of boundary positions measured as the number of
    characters consumed so far (1 .. len(word)-1), matching MBI.py.
    """
    boundaries = set()
    char_count = 0
    surfaces = [clean_token_surface(t) for t in tokens]

    for i, surface in enumerate(surfaces[:-1]):
        char_count += len(surface)
        if 0 < char_count < len(word):
            boundaries.add(char_count)

    # Fallback: if reconstruction overshoots / undershoots badly, try
    # equal-split approximation so metrics still run
    if not boundaries and len(tokens) > 1 and len(word) > 1:
        # Use cumulative ratio of surface lengths if available
        total = sum(max(len(s), 1) for s in surfaces) or 1
        acc = 0
        for s in surfaces[:-1]:
            acc += max(len(s), 1)
            pos = int(round(acc / total * len(word)))
            if 0 < pos < len(word):
                boundaries.add(pos)

    return boundaries
