"""
Morphological Boundary Identification (MBI).

Treats boundary detection as binary classification between consecutive characters.
Compares gold NCHLT segmentations (underscores) against tokenizer cuts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tokenizer_services import clean_token_surface, get_token_boundary_positions, tokenize_text


def get_gold_boundaries(gold_segmentation: str) -> set:
    """
    Extract boundary positions from a gold string like 'ba_ya_hamba'.

    Positions count only non-underscore characters, so for 'ba_ya_hamba'
    boundaries are at character offsets 2 and 4 (1-based cumulative length
    of surface characters before each '_').
    """
    char_count = 0
    boundaries = set()
    for char in gold_segmentation:
        if char == "_":
            if char_count > 0:
                boundaries.add(char_count)
        else:
            char_count += 1
    return boundaries


def get_tokenizer_boundaries(tokens: List[str], word: str = "") -> set:
    """
    Extract boundary positions from tokenizer output.

    Strips ## / ▁ / Ġ markers so lengths match surface characters.
    If `word` is provided, positions are clamped to valid range.
    """
    if word:
        return get_token_boundary_positions(tokens, word)

    # Legacy path when word is unknown
    char_count = 0
    boundaries = set()
    for token in tokens[:-1]:
        char_count += len(clean_token_surface(token))
        boundaries.add(char_count)
    return boundaries


def get_boundary_counts(
    text: str, tokens: List[str], gold_segmentation: str
) -> Tuple[int, int, int, int]:
    """Return (tp, fp, tn, fn) for one word."""
    gold_boundaries = get_gold_boundaries(gold_segmentation)
    tokenizer_boundaries = get_tokenizer_boundaries(tokens, word=text)

    # Possible cut sites: after char 1 .. after char n-1
    possible_positions = set(range(1, len(text))) if len(text) > 1 else set()

    tp = len(gold_boundaries & tokenizer_boundaries)
    fp = len(tokenizer_boundaries - gold_boundaries)
    tn = len(possible_positions - (gold_boundaries | tokenizer_boundaries))
    fn = len(gold_boundaries - tokenizer_boundaries)

    return tp, fp, tn, fn


def precision(tp: int, fp: int) -> float:
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)


def f1(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def evaluate_word(tokenizer, word: str, gold_segmentation: str) -> Dict:
    """Score one word; return metrics and debug fields."""
    if not word or len(word) < 2:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    result = tokenize_text(tokenizer, word)
    tokens = result["tokens"]
    tp, fp, tn, fn = get_boundary_counts(word, tokens, gold_segmentation)
    p = precision(tp, fp)
    r = recall(tp, fn)
    return {
        "precision": p,
        "recall": r,
        "f1": f1(p, r),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tokens": tokens,
        "gold_segmentation": gold_segmentation,
        "word": word,
    }


def evaluate_corpus(
    tokenizer,
    gold_dict: Dict[str, str],
    max_words: Optional[int] = 500,
) -> Dict:
    """
    Macro-average P/R/F1 over alphabetic words in the gold dictionary.

    gold_dict: surface_word -> gold_segmentation_with_underscores
    """
    candidates = [
        (w, seg)
        for w, seg in gold_dict.items()
        if len(w) >= 3 and any(c.isalpha() for c in w)
    ]
    if max_words is not None:
        candidates = candidates[:max_words]

    sum_p = sum_r = sum_f = 0.0
    total_tp = total_fp = total_fn = 0
    count = 0

    for word, seg in candidates:
        m = evaluate_word(tokenizer, word, seg)
        sum_p += m["precision"]
        sum_r += m["recall"]
        sum_f += m["f1"]
        total_tp += m["tp"]
        total_fp += m["fp"]
        total_fn += m["fn"]
        count += 1

    if count == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "n_words": 0,
        }

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f = f1(micro_p, micro_r)

    return {
        "precision": sum_p / count,
        "recall": sum_r / count,
        "f1": sum_f / count,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f,
        "n_words": count,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
