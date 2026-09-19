"""
Tokenizer Analysis Dashboard (TAD) — Centralized Ranking Engine & Methodology.

This module provides the authoritative ranking and scoring methodology for evaluating
and comparing tokenizers across multiple quantitative dimensions:
1. Metric Normalization: Scaling disparate raw metrics (e.g. token counts in thousands,
   fertility ratios in small floats, and precision in percentages) to a uniform [0.0, 1.0] scale.
2. Directional Alignment: Inverting "lower is better" metrics (e.g. token count, fertility,
   token premium) so that 1.0 universally signifies optimal desirability across all metrics.
3. Centralized Composite Weighting: Combining efficiency (50%) and linguistic alignment (50%)
   components into a unified [0.0, 100.0] score.
4. Dynamic Missing-Component Fallback: Gracefully reallocating weights (e.g. 100% to efficiency
   when morphology gold data is unavailable for English or Afrikaans) to avoid unfair penalties,
   NaN values, or ZeroDivisionError exceptions.
5. Deterministic Leaderboard Ranking: Sorting tokenizers in descending order of overall score
   and assigning continuous 1-based ranks.

Design Rationale & Methodology (Capstone Explanations):
- Disparate Scale Disparity: In tokenizer analysis, raw metrics span completely different
  orders of magnitude. Without anchor-bound normalization, a metric like `token_count` (1000s)
  would completely dominate a delicate metric like `mbi_f1` (0.0 to 1.0) by orders of magnitude.
  Fixed reference anchor bounds clamp each dimension to [0.0, 1.0] to preserve relative
  sensitivity without sample-dependent normalization jitter.
- Directionality Inversion: In computational linguistics and LLM serving, fewer tokens per word
  (low fertility) and lower relative token counts directly reduce inference latency and API cost.
  However, human users intuitive expect "higher score = better model". Mapping lower-is-better
  metrics via `1.0 - norm` guarantees that every dimension aligns with desirability while strictly
  preserving linear distances.
- 50/50 Dual-Axis Equilibrium: Efficiency metrics (cost, speed, memory) and linguistic metrics
  (morphological boundary preservation, morphological fidelity) represent the fundamental trade-off
  in subword tokenization. A tokenizer could achieve trivial compression by treating entire phrases
  as single tokens, or achieve 100% boundary accuracy by tokenizing at character level. Equal weighting
  ensures that winning models must strike a sound balance between compression and grammatical integrity.
- Fallback Fairness: Languages without NCHLT morphological gold standards (such as English or Afrikaans)
  cannot compute MBI metrics. Penalizing these models with a 0% linguistic score would artificially
  condemn them to the bottom of the leaderboard. Conversely, dropping them prevents cross-lingual
  comparison. Dynamically allocating 100% weight to efficiency ensures fair evaluation on available
  evidence while explicitly recording the missing component in the transparent audit trail.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# Centralized Metric Definitions, Directions, and Reference Anchor Bounds
# ===========================================================================

# Directionality indicators:
# "lower"  -> Lower raw value is superior (inverted: 1.0 - normalized)
# "higher" -> Higher raw value is superior (preserved: normalized)
#
# Capstone Rationale — Why Invert Lower-is-Better Metrics?
# In LLM serving and computational linguistics, efficiency is defined by compactness:
# 1. Fertility (tokens per word): Lower is better. Fewer tokens per word mean smaller sequence
#    lengths, which dramatically speeds up autoregressive decoding and reduces quadratic KV-cache memory.
# 2. Token Count: Lower is better. Total subwords needed to represent the benchmark text directly
#    dictates compute throughput.
# 3. Token Premium (tokens_lang / tokens_eng): Lower is better. Parallel sentences in African languages
#    routinely produce 2x-4x more tokens than English in commercial models, imposing an unfair "tax"
#    in API billing and latency. Parity (1.0) is the ideal target.
# Conversely, users intuitively expect higher leaderboard scores to reflect better models.
# Inverting lower-is-better dimensions via `1.0 - norm` aligns all metrics along a single unified axis
# where 1.0 universally represents optimum desirability, enabling sound linear weighting.
METRIC_DIRECTION: Dict[str, str] = {
    "token_count": "lower",
    "fertility": "lower",
    "token_premium": "lower",
    "compression": "higher",
    "mbi_precision": "higher",
    "mbi_recall": "higher",
    "mbi_f1": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",
}

# Reference Anchor Bounds:
# Defined based on empirical ranges observed across South African and multilingual corpora.
#
# Capstone Rationale — Why Fixed Empirical Anchors Instead of Dynamic Batch Min-Max?
# If we normalized metrics using dynamic min-max across the currently loaded models in a session,
# the leaderboard would suffer from extreme "ranking jitter": adding or removing a single tokenizer
# would alter the min/max denominators, causing every other model's score to fluctuate and breaking
# reproducibility across independent runs.
#
# To provide an invariant academic benchmark, we ground our normalization in fixed empirical bounds:
# - fertility: (1.0, 5.0)
#   * 1.0 is the theoretical minimum for whitespace-segmented text (1 token per word, word-level limit).
#   * For agglutinative Bantu languages (e.g. isiZulu), standard Western tokenizers heavily fragment words
#     due to poor vocabulary coverage of concordial prefixes, reaching 3.5 to 5.0 tokens per word.
#     Values > 5.0 indicate degeneration into character-by-character spelling.
# - compression: (1.0, 6.0)
#   * Measured in UTF-8 bytes per token.
#   * 1.0 represents raw byte-fallback (1 byte per token, zero compression).
#   * 6.0 represents dense subword packing where common multicharacter morphemes/words are unified.
# - token_count: (0.0, 10000.0)
#   * Slices of our standardized FLORES-dev test corpus produce 1,500 to 7,500 tokens for standard models.
#   * 10,000 serves as a realistic empirical ceiling for severely fragmented tokenizers.
# - token_premium: (1.0, 4.0)
#   * 1.0 represents true cross-lingual parity with English (the goal of equitable multilingual tokenization).
#   * 4.0 captures the severe 3x-4x penalty typically observed when western LLM tokenizers process African text.
# - morphological metrics (precision, recall, f1): (0.0, 1.0)
#   * Standard statistical classification probabilities naturally bounded in [0.0, 1.0].
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "fertility": (1.0, 5.0),
    "compression": (1.0, 6.0),
    "token_count": (0.0, 10000.0),
    "token_premium": (1.0, 4.0),
    "mbi_precision": (0.0, 1.0),
    "mbi_recall": (0.0, 1.0),
    "mbi_f1": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "f1": (0.0, 1.0),
}

# High-level category composite weights:
# Capstone Rationale — Why 50/50 Dual-Axis Equilibrium?
# Subword tokenization involves a fundamental trade-off:
# - A tokenizer could maximize efficiency by memorizing whole phrases into an excessively large vocabulary,
#   yielding low fertility but destroying generalization on unseen words.
# - Conversely, a character-level or highly fragmented tokenizer could align cleanly with morpheme boundaries
#   but produce catastrophic token counts, causing severe inference slowdown and KV-cache exhaustion.
# The 50/50 split between efficiency (cost, latency, memory) and linguistic alignment (morphological boundary
# respect against NCHLT gold segmentations) forces winning models to achieve genuine balance.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "efficiency": 0.50,
    "linguistic": 0.50,
}

# Metric sub-weights within categories:
# - Efficiency: 4 metrics equally balanced at 0.125 each (total 0.50)
# - Linguistic: Morphological F1 weighted primarily (0.25) as the harmonic mean of precision (0.125) and recall (0.125)
DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
    "token_count": 0.125,
    "fertility": 0.125,
    "compression": 0.125,
    "token_premium": 0.125,
    "mbi_precision": 0.125,
    "mbi_recall": 0.125,
    "mbi_f1": 0.250,
    "precision": 0.125,
    "recall": 0.125,
    "f1": 0.250,
}


# ===========================================================================
# Metric Normalization & Directionality Inversion Functions
# ===========================================================================

def normalize_metric(
    value: Any,
    metric_name: str,
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """
    Clamps and linearly scales a raw metric value to [0.0, 1.0] using reference anchor bounds.

    Args:
        value: Raw numeric metric value. Handles None, strings, and edge case extremes safely.
        metric_name: Name of the metric (e.g. 'fertility', 'compression', 'mbi_f1').
        custom_bounds: Optional dictionary overriding default anchor bounds for specific metrics.

    Returns:
        A float strictly clamped between 0.0 and 1.0.

    Mathematical Rationale:
        Linear min-max normalization: `norm = (value - min) / (max - min)`.
        Values below `min` are clamped to 0.0; values exceeding `max` are clamped to 1.0.
        When min equals max (degenerate anchor bound), division by zero is intercepted
        and 1.0 is returned if value >= min, else 0.0.
    """
    if value is None:
        return 0.0

    try:
        num_val = float(value)
    except (ValueError, TypeError):
        return 0.0

    if math.isnan(num_val) or num_val == float("-inf"):
        return 0.0
    if num_val == float("inf"):
        return 1.0

    # Determine anchor bounds
    norm_key = metric_name.lower().strip()
    bounds: Optional[Tuple[float, float]] = None

    if custom_bounds:
        if metric_name in custom_bounds:
            bounds = custom_bounds[metric_name]
        elif norm_key in custom_bounds:
            bounds = custom_bounds[norm_key]

    if bounds is None:
        bounds = DEFAULT_BOUNDS.get(norm_key, (0.0, 1.0))

    min_b, max_b = bounds
    min_val = float(min(min_b, max_b))
    max_val = float(max(min_b, max_b))

    # Guard against zero range (degenerate anchor bounds)
    if max_val == min_val:
        return 1.0 if num_val >= min_val else 0.0

    # Linear normalization and clamping to [0.0, 1.0]
    scaled = (num_val - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, scaled))


def invert_if_lower_is_better(normalized_value: float, metric_name: str) -> float:
    """
    Inverts normalized score if lower raw values indicate superior performance.

    Args:
        normalized_value: Float in [0.0, 1.0].
        metric_name: Name of the metric.

    Returns:
        1.0 - normalized_value for "lower is better" metrics (token_count, fertility, token_premium).
        normalized_value for "higher is better" metrics (compression, precision, recall, f1).

    Mathematical Rationale:
        To construct a coherent composite score, every contributing component must be
        monotonically aligned with desirability. For lower-is-better metrics:
        - raw minimum -> normalized 0.0 -> inverted 1.0 (perfect score).
        - raw maximum -> normalized 1.0 -> inverted 0.0 (poorest score).
    """
    try:
        norm = max(0.0, min(1.0, float(normalized_value)))
    except (ValueError, TypeError):
        norm = 0.0

    norm_key = metric_name.lower().strip()
    direction = METRIC_DIRECTION.get(norm_key, "higher")

    if direction == "lower" or norm_key in ("token_count", "fertility", "token_premium"):
        return max(0.0, min(1.0, 1.0 - norm))
    return norm


# ===========================================================================
# Centralized Composite Score Computation
# ===========================================================================

def compute_composite_score(
    metrics: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Computes an overall ranking score [0.0 - 100.0] combining efficiency (50%)
    and linguistic alignment (50%) components with transparent breakdown.

    Handles missing metrics (e.g. English/Afrikaans without NCHLT gold morphology data)
    dynamically by reallocating weight gracefully to available components.

    Args:
        metrics: Dictionary of raw metric values.
        weights: Optional dictionary specifying category weights (e.g. {"efficiency": 0.8, "linguistic": 0.2}).
        custom_bounds: Optional dictionary overriding anchor bounds for normalization.

    Returns:
        Detailed score breakdown dictionary:
        {
            "overall_score": float [0.0 - 100.0],
            "efficiency_score": float [0.0 - 100.0],
            "linguistic_score": Optional[float] [0.0 - 100.0] (None if missing),
            "normalized_metrics": Dict[str, float],
            "inverted_metrics": Dict[str, float],
            "weights_used": Dict[str, float],
            "missing_components": List[str]
        }
    """
    if not isinstance(metrics, dict):
        metrics = {}

    normalized_metrics: Dict[str, float] = {}
    inverted_metrics: Dict[str, float] = {}
    missing_components: List[str] = []

    # -----------------------------------------------------------------------
    # Step 1: Normalize and invert all provided numeric metrics
    # -----------------------------------------------------------------------
    for k, v in metrics.items():
        if v is not None:
            try:
                f_val = float(v)
                if not (math.isnan(f_val) or math.isinf(f_val)):
                    norm = normalize_metric(f_val, k, custom_bounds=custom_bounds)
                    inv = invert_if_lower_is_better(norm, k)
                    normalized_metrics[k] = norm
                    inverted_metrics[k] = inv
            except (ValueError, TypeError):
                continue

    # -----------------------------------------------------------------------
    # Step 2: Compute Efficiency Component Score (0.0 - 100.0)
    # -----------------------------------------------------------------------
    # Efficiency metrics: token_count, fertility, compression, token_premium.
    # When available, each contributes equally (0.125 of total, or 0.25 of efficiency).
    eff_keys = ["token_count", "fertility", "compression", "token_premium"]
    available_eff = [k for k in eff_keys if k in inverted_metrics]

    efficiency_score: Optional[float] = None
    if available_eff:
        eff_sum = sum(inverted_metrics[k] for k in available_eff)
        efficiency_score = (eff_sum / len(available_eff)) * 100.0
    else:
        missing_components.append("efficiency")

    # -----------------------------------------------------------------------
    # Step 3: Compute Linguistic Alignment Component Score (0.0 - 100.0)
    # -----------------------------------------------------------------------
    # Morphological Boundary Identification metrics: precision, recall, f1.
    # Can be prefixed with 'mbi_' or bare. F1 is weighted primarily (50%) when all 3 present.
    f1_key = "mbi_f1" if "mbi_f1" in inverted_metrics else ("f1" if "f1" in inverted_metrics else None)
    prec_key = "mbi_precision" if "mbi_precision" in inverted_metrics else ("precision" if "precision" in inverted_metrics else None)
    rec_key = "mbi_recall" if "mbi_recall" in inverted_metrics else ("recall" if "recall" in inverted_metrics else None)

    linguistic_score: Optional[float] = None
    ling_components = {}
    if f1_key is not None:
        ling_components[f1_key] = (inverted_metrics[f1_key], 0.50)
    if prec_key is not None:
        ling_components[prec_key] = (inverted_metrics[prec_key], 0.25)
    if rec_key is not None:
        ling_components[rec_key] = (inverted_metrics[rec_key], 0.25)

    if ling_components:
        # Dynamically re-normalize subweights for whichever linguistic metrics are available
        total_subweight = sum(weight for _, weight in ling_components.values())
        if total_subweight > 0:
            ling_norm_sum = sum(val * (weight / total_subweight) for val, weight in ling_components.values())
            linguistic_score = ling_norm_sum * 100.0
    else:
        missing_components.append("linguistic")

    # -----------------------------------------------------------------------
    # Step 4: Centralized Composite Weighting & Dynamic Fallback
    # -----------------------------------------------------------------------
    # Default: 50% Efficiency, 50% Linguistic
    req_eff_w = DEFAULT_WEIGHTS["efficiency"]
    req_ling_w = DEFAULT_WEIGHTS["linguistic"]

    if weights and isinstance(weights, dict):
        if "efficiency" in weights:
            req_eff_w = float(weights["efficiency"])
        if "linguistic" in weights:
            req_ling_w = float(weights["linguistic"])

    weights_used: Dict[str, float] = {}
    overall_score: float = 0.0

    if efficiency_score is not None and linguistic_score is not None:
        # Both components are available: apply weights directly
        tot = req_eff_w + req_ling_w
        if tot > 0:
            w_eff = req_eff_w / tot
            w_ling = req_ling_w / tot
        else:
            w_eff, w_ling = 0.5, 0.5

        overall_score = (w_eff * efficiency_score) + (w_ling * linguistic_score)
        weights_used = {"efficiency": w_eff, "linguistic": w_ling}

    elif efficiency_score is not None and linguistic_score is None:
        # Fallback Case 1: Linguistic metrics missing (e.g. English, Afrikaans).
        #
        # Capstone Rationale — Why Fall Back to 100% Efficiency Weight?
        # In South African NLP, morphological gold segmentations (NCHLT) are only available for
        # 9 indigenous Bantu languages. English and Afrikaans lack morphological gold annotations.
        # If we penalized models evaluated on English or Afrikaans by assigning 0.0 to the linguistic
        # component, an exceptional English model would have its score halved (~85 -> ~42.5), unfairly
        # pushing it to the bottom of the leaderboard through no fault of its own.
        # Conversely, disqualifying them would prevent essential cross-lingual comparisons against
        # multilingual models.
        # By dynamically reallocating 100% of the weight to efficiency, the model is evaluated fairly
        # strictly on the evidence available, while `missing_components` explicitly records that
        # linguistic alignment could not be computed.
        overall_score = efficiency_score
        weights_used = {"efficiency": 1.0, "linguistic": 0.0}

    elif efficiency_score is None and linguistic_score is not None:
        # Fallback Case 2: Efficiency metrics missing
        # Reallocate 100% weight to linguistic alignment
        overall_score = linguistic_score
        weights_used = {"efficiency": 0.0, "linguistic": 1.0}

    else:
        # Fallback Case 3: Degenerate case with no usable metrics
        overall_score = 0.0
        weights_used = {"efficiency": 0.0, "linguistic": 0.0}

    # Clamp overall score strictly to [0.0, 100.0]
    overall_score = max(0.0, min(100.0, float(overall_score)))

    return {
        "overall_score": round(overall_score, 4),
        "efficiency_score": round(efficiency_score, 4) if efficiency_score is not None else None,
        "linguistic_score": round(linguistic_score, 4) if linguistic_score is not None else None,
        "normalized_metrics": normalized_metrics,
        "inverted_metrics": inverted_metrics,
        "weights_used": weights_used,
        "missing_components": missing_components,
    }


# ===========================================================================
# Leaderboard Sorting & Rank Assignment
# ===========================================================================

def rank_tokenizers(leaderboard_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sorts leaderboard entries descending by overall_score and assigns 1-based ranks.

    - If overall_score is already present in an entry, it is respected.
    - If overall_score is missing, it is computed via compute_composite_score(entry['metrics']).
    - Preserves all existing entry metadata (model_id, added_at, languages, tags, etc.).
    - Ties are broken deterministically by model_id to prevent UI jitter.
    - An empty list returns [].

    Args:
        leaderboard_entries: List of tokenizer entry dictionaries.

    Returns:
        A new list of ranked entry dictionaries with assigned 'rank' (1, 2, ...).
    """
    if not leaderboard_entries:
        return []

    prepared_entries: List[Dict[str, Any]] = []

    for entry in leaderboard_entries:
        if not isinstance(entry, dict):
            continue

        item = dict(entry)
        score: Optional[float] = None

        if item.get("overall_score") is not None:
            try:
                val = float(item["overall_score"])
                if not (math.isnan(val) or math.isinf(val)):
                    score = val
            except (ValueError, TypeError):
                score = None

        if score is None:
            raw_metrics = item.get("metrics") or {}
            score_data = compute_composite_score(raw_metrics)
            score = score_data["overall_score"]
            item["overall_score"] = score
            if "efficiency_score" not in item:
                item["efficiency_score"] = score_data["efficiency_score"]
            if "linguistic_score" not in item:
                item["linguistic_score"] = score_data["linguistic_score"]
            if "score_breakdown" not in item:
                item["score_breakdown"] = score_data
        else:
            item["overall_score"] = score

        prepared_entries.append(item)

    # Sort descending by overall_score, breaking ties deterministically by model_id
    sorted_entries = sorted(
        prepared_entries,
        key=lambda x: (-float(x.get("overall_score", 0.0)), str(x.get("model_id", ""))),
    )

    # Assign 1-based continuous rank
    for idx, item in enumerate(sorted_entries, start=1):
        item["rank"] = idx

    return sorted_entries
