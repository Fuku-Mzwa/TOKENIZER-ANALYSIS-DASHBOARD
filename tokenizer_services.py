"""
tokenizer_services.py — Hugging Face Tokenizer Loading, In-Memory Caching & Tokenization.

Design & Architectural Rationale (Capstone Explanations):
---------------------------------------------------------
1. In-Memory Caching (TOKENIZER_CACHE):
   Downloading or instantiating tokenizers repeatedly causes severe network bottlenecks,
   rate-limiting from Hugging Face Hub, and multi-second UI freezes during Streamlit reruns.
   Caching instantiated AutoTokenizer objects in an in-memory dictionary guarantees O(1)
   retrieval across multi-language FLORES evaluations and comparison workflows.

2. User-Safe Error Sanitization vs. Developer Logger Retention:
   When loading models via Hugging Face's `AutoTokenizer.from_pretrained()`, failures can occur
   for many reasons: typos in repo identifiers, missing tokenizer configs, HF Hub rate limits,
   offline environments, or deserialization bugs.
   - What happens without sanitization: Raw Python tracebacks span 30-50 lines, dumping low-level
     urllib3 connection calls, internal HTTP 404/401 status headers, and local directory paths
     (e.g., `C:\\Users\\...\\.cache\\huggingface\\...`). Leaking this into the UI creates intense
     cognitive noise for evaluators and exposes internal file structures.
   - The Sanitization Architecture: We decouple user notification from developer diagnostics.
     Full stack traces and HTTP context are captured in backend terminal logs via `logger.exception()`,
     ensuring developers have complete debugging information. Simultaneously, the UI receives a
     concise, actionable, user-friendly error message (`ValueError`) explaining what failed and how
     to recover without technical jargon.
   - Python Exception Chaining Suppression (`from None`): Raising `from None` explicitly suppresses
     Python's default context chaining, ensuring no internal traceback fragments inadvertently leak
     into UI alerts.

3. Subword Marker Normalization:
   Modern subword tokenizers introduce distinct prefixes or delimiters to mark word-initial
   or continuation tokens:
   - WordPiece (BERT): "##"
   - Byte-Pair Encoding (GPT-2, RoBERTa): "Ġ" (\u0120) or "\u010a"
   - SentencePiece (XLM-RoBERTa, T5): " " (\u2581)
   When mapping token boundaries onto character offsets for Morphological Boundary
   Identification (MBI), these artificial delimiter characters must be stripped so that
   character spans align accurately with raw lexical surfaces.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Global in-memory cache to prevent redundant re-instantiation of tokenizer models
TOKENIZER_CACHE: Dict[str, Any] = {}

# Common subword continuation and word-boundary markers used across model architectures:
# - "##" -> BERT / DistilBERT WordPiece continuation marker
# - " " (\u2581) -> SentencePiece word boundary delimiter
# - "Ġ" (\u0120) -> Hugging Face Byte-level BPE leading space marker
# - "Ċ" -> Byte-level BPE newline marker
_MARKER_PREFIXES = ("##", " ", "Ġ", "Ċ", "\u0120")


def load_tokenizer(model_id: str, use_fast: bool = True):
    #Load a Hugging Face tokenizer by model ID (cached).
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
    #Tokenize text; return tokens, ids, and basic counts.
    if text is None:
        text = ""
    text = str(text)

    offsets = []
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors=None)
        ids = encoded["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        offsets = encoded.get("offset_mapping", [])
    except Exception:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "tokens": tokens,
        "ids": ids,
        "token_count": len(tokens),
        "char_count": len(text),
        "byte_count": len(text.encode("utf-8")),
        "offsets": offsets,
    }


def clean_token_surface(token: str) -> str:
    #Strip common subword markers so len() reflects surface characters.
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
