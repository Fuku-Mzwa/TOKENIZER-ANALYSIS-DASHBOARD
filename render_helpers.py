"""
render_helpers.py - Presentation & Formatting Transformation Layer for TAD.

This module provides pure, side-effect-free helper functions for formatting,
structuring, and transforming tokenizer evaluation metrics for dashboard display.

Design Rationale & Architectural Decisions (Capstone Explanations):
---------------------------------------------------------------------
1. Why Decouple Pure Display Helpers from Streamlit Rendering?
   A major architectural challenge in Streamlit development is that the framework tightly
   couples reactive UI rendering (`st.metric`, `st.dataframe`, `st.markdown`) with Python
   script execution. Mixing data formatting, delta computation, and string manipulation
   directly inside Streamlit widget calls introduces three severe drawbacks:
   
   - Fragile, Slow Testing: Testing Streamlit UI code directly requires either a running
     web server with headless browser drivers (Playwright/Selenium) or fragile monkey-patching
     of Streamlit's internal protobuf session context. By isolating all presentation logic
     into pure, deterministic Python functions with zero dependencies on `streamlit`, we can
     run comprehensive unit and regression tests (e.g. `tests/test_render_results.py`) in
     under 50 milliseconds using standard `python -m unittest`.
   - Single Source of Truth Across Dashboard Pages: The TAD application presents metrics in
     multiple contexts: single-model analysis in 'Inspect', pairwise differential analysis
     in 'Compare', and multi-model global benchmarking in 'Rank'. If formatting rules (such as
     3 decimal places for fertility, 2 for compression, comma grouping for token counts, and
     medal emojis for top ranks) were hardcoded inside each page script, styling inconsistencies
     and duplicated code would inevitably proliferate. `render_helpers.py` provides a unified,
     reusable presentation layer.
   - Fault Isolation & UI Crash Prevention: Real-world NLP benchmarks generate diverse numerical
     edge cases: missing language baselines, zero-word corpora, division-by-zero, `None` values,
     and IEEE 754 `NaN`/`inf` floats. By filtering and sanitizing these values in pure helper
     functions with defensive fallbacks ('N/A', '-', neutral badges), the Streamlit UI layer is
     guaranteed never to crash with unhandled TypeErrors or ValueError exceptions during rendering.

2. Consistent Precision Standards:
   - Fertility: Formatted to 3 decimal places (e.g. 2.145 tokens/word) because subword fragmentation
     differences between competitive tokenizers are subtle but computationally significant.
   - Compression: Formatted to 2 decimal places (bytes/token) since byte density operates across
     coarser intervals.
   - Token & Word Counts: Formatted with thousands-separator commas (e.g. `12,450`) to avoid visual
     fatigue when comparing large benchmark splits.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd


def format_metric(val: Any, decimals: int = 2) -> str:
    """
    Format a floating-point metric value cleanly with specified decimal precision.

    Rationale:
    In dashboard scorecards and leaderboards, displaying raw floating-point values
    (e.g., 3.14159265) creates cognitive noise. Fixed decimal rounding ensures
    uniform scannability. Missing or invalid values return 'N/A' instead of crashing
    or rendering unsightly Python 'None' strings.
    """
    if val is None:
        return "N/A"
    try:
        f_val = float(val)
        if math.isnan(f_val):
            return "N/A"
        return f"{f_val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def format_int(val: Any) -> str:
    """
    Format an integer count with thousands-separator commas.

    Rationale:
    Corpus token counts frequently range in the tens or hundreds of thousands.
    Adding commas (e.g., '12,345' instead of '12345') significantly reduces visual
    parsing fatigue during tokenizer comparison.
    Defensively handles infinite, NaN, and overflowing values gracefully.
    """
    if val is None:
        return "0"
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return str(val)
        return f"{int(round(f_val)):,}"
    except (ValueError, TypeError, OverflowError):
        return str(val)


def get_status_color(f1_score: Optional[float]) -> str:
    """
    Return a semantic status color name based on morphological F1 alignment.

    Rationale:
    MBI F1 reflects how closely tokenizer boundaries align with linguistic morpheme
    boundaries from NCHLT gold standards:
    - F1 >= 0.60: Strong alignment (green)
    - 0.35 <= F1 < 0.60: Moderate alignment (orange)
    - F1 < 0.35: Weak or poor alignment (red)
    - Missing/unsupported (e.g., English/Afrikaans): Neutral gray
    """
    if f1_score is None:
        return "gray"
    try:
        val = float(f1_score)
        if math.isnan(val):
            return "gray"
        if val >= 0.60:
            return "green"
        elif val >= 0.35:
            return "orange"
        else:
            return "red"
    except (ValueError, TypeError):
        return "gray"


def get_rank_badge(rank: int) -> str:
    """
    Return a visual badge/icon for a leaderboard rank position.

    Rationale:
    Top 3 placements in benchmarks deserve immediate visual distinction. We use
    podium medals (🥇, 🥈, 🥉) for top 3, while subsequent ranks display standard
    numerical rank badges (e.g., '#4'). Invalid or non-positive ranks return '-'.
    """
    try:
        r = int(rank)
    except (ValueError, TypeError):
        return "-"

    if r == 1:
        return "🥇"
    elif r == 2:
        return "🥈"
    elif r == 3:
        return "🥉"
    elif r > 0:
        return f"#{r}"
    return "-"


def format_token_chips(
    tokens: Optional[Sequence[str]], max_show: int = 60
) -> Dict[str, Any]:
    """
    Format tokenizer tokens into backtick-enclosed markdown chips for visual inspection.

    Rationale:
    Displaying tokens as individual code chips (e.g. `hel` `##lo`) makes word-splitting
    behavior immediately apparent to human evaluators.
    To avoid browser DOM bloat and layout breakage when tokenizing full paragraphs,
    we cap display to `max_show` tokens and calculate the `overflow` count so the UI
    can inform the user if tokens were truncated.
    """
    if not tokens:
        return {
            "empty": True,
            "total": 0,
            "overflow": 0,
            "chips_markdown": "",
        }

    total = len(tokens)
    shown = tokens[:max_show]
    overflow = max(0, total - len(shown))
    chips_markdown = " ".join(f"`{t}`" for t in shown)

    return {
        "empty": False,
        "total": total,
        "overflow": overflow,
        "chips_markdown": chips_markdown,
    }


def format_efficiency_metrics(
    metrics: Optional[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Format efficiency metrics dictionary into scorecard data items.

    Rationale:
    The standard scorecard displays 4 metric tiles:
    1. Token count (integer with commas)
    2. Fertility (3 decimal places: tokens / word)
    3. Compression (2 decimal places: bytes / token)
    4. Token premium (3 decimal places: tokens_lang / tokens_eng) when parallel English
       is available; otherwise falls back to Word count so layout remains 4 columns.
    Defensively formats token counts and metrics even if values are None, infinite, or NaN.
    """
    if not metrics or not isinstance(metrics, dict):
        return []

    token_val = metrics.get("token_count", 0)
    token_val = 0 if token_val is None else token_val

    fertility_val = metrics.get("fertility", 0.0)
    fertility_val = 0.0 if fertility_val is None else fertility_val

    compression_val = metrics.get("compression", 0.0)
    compression_val = 0.0 if compression_val is None else compression_val

    cards = [
        {"label": "Token count", "value": format_int(token_val)},
        {"label": "Fertility", "value": format_metric(fertility_val, 3)},
        {"label": "Compression (B/tok)", "value": format_metric(compression_val, 2)},
    ]

    # Token premium requires parallel English; free-text or standalone runs lack it.
    if "token_premium" in metrics and metrics["token_premium"] is not None:
        cards.append({
            "label": "Token premium",
            "value": format_metric(metrics["token_premium"], 3),
        })
    else:
        word_val = metrics.get("word_count", 0)
        word_val = 0 if word_val is None else word_val
        cards.append({
            "label": "Words",
            "value": format_int(word_val),
        })

    return cards


def format_morphology_metrics(
    morphology: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format Morphological Boundary Identification (MBI) metrics for UI display.

    Rationale:
    Morphology metrics are only valid if gold segmentation data exists and at least
    one word was successfully evaluated (`n_words > 0`). If no morphology data exists
    (e.g., for English or Afrikaans) or evaluation had zero words, we set `show=False`
    so the UI does not render misleading zeros.
    """
    if not morphology or morphology.get("n_words", 0) <= 0:
        return {"show": False, "cards": [], "caption": ""}

    precision_val = float(morphology.get("precision", 0.0) or 0.0)
    recall_val = float(morphology.get("recall", 0.0) or 0.0)
    f1_val = float(morphology.get("f1", 0.0) or 0.0)
    micro_f1_val = float(morphology.get("micro_f1", 0.0) or 0.0)
    n_words = int(morphology.get("n_words", 0) or 0)

    cards = [
        {"label": "Precision", "value": f"{precision_val:.3f}"},
        {"label": "Recall", "value": f"{recall_val:.3f}"},
        {"label": "F1", "value": f"{f1_val:.3f}"},
    ]
    caption = f"Evaluated on {n_words:,} words · micro-F1 = {micro_f1_val:.3f}"

    return {
        "show": True,
        "cards": cards,
        "caption": caption,
    }


def build_comparison_rows(
    name_a: str,
    eff_a: Dict[str, Any],
    morph_a: Optional[Dict[str, Any]],
    name_b: str,
    eff_b: Dict[str, Any],
    morph_b: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build tabular comparison rows for Tokenizer A and Tokenizer B.

    Rationale:
    Rounds floating-point metrics to standard precision (Fertility: 3, Compression: 2,
    Token premium: 3). MBI columns (Precision, Recall, F1) are appended only when
    morphology data with positive word counts is present, preventing clutter when
    evaluating languages without gold segmentations.
    Defensively handles None or invalid values, formatting missing metrics as 'N/A'.
    """
    def _safe_cell(val: Any, decimals: int) -> Union[float, str]:
        if val is None:
            return "N/A"
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return "N/A"
            return round(f, decimals)
        except (ValueError, TypeError, OverflowError):
            return "N/A"

    rows: List[Dict[str, Any]] = []

    for prefix, name, eff, morph in (
        ("A", name_a, eff_a, morph_a),
        ("B", name_b, eff_b, morph_b),
    ):
        eff_dict = eff if isinstance(eff, dict) else {}
        token_count = eff_dict.get("token_count")
        if token_count is None:
            tokens_val: Any = "N/A"
        else:
            try:
                f_tc = float(token_count)
                tokens_val = int(round(f_tc)) if not (math.isnan(f_tc) or math.isinf(f_tc)) else "N/A"
            except (ValueError, TypeError, OverflowError):
                tokens_val = "N/A"

        row: Dict[str, Any] = {
            "Tokenizer": f"{prefix}: {name}",
            "Tokens": tokens_val,
            "Fertility": _safe_cell(eff_dict.get("fertility"), 3),
            "Compression": _safe_cell(eff_dict.get("compression"), 2),
            "Token premium": _safe_cell(eff_dict.get("token_premium"), 3),
        }
        if morph and isinstance(morph, dict) and (morph.get("n_words", 0) or 0) > 0:
            row["MBI Precision"] = _safe_cell(morph.get("precision"), 3)
            row["MBI Recall"] = _safe_cell(morph.get("recall"), 3)
            row["MBI F1"] = _safe_cell(morph.get("f1"), 3)
        rows.append(row)

    return rows


def build_comparison_chart_data(
    eff_a: Dict[str, Any],
    eff_b: Dict[str, Any],
    morph_a: Optional[Dict[str, Any]] = None,
    morph_b: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Prepare multi-series grouped bar chart data comparing Tokenizer A and Tokenizer B.

    Rationale:
    Visual bar charts give immediate intuition on relative efficiency. Fertility and
    Token premium are always plotted. MBI F1 is plotted if and only if both tokenizers
    have valid morphology results, ensuring balanced side-by-side comparison bars.
    Defensively handles None or missing metric values by coalescing to 0.0.
    """
    def _chart_float(val: Any) -> float:
        if val is None:
            return 0.0
        try:
            f = float(val)
            return 0.0 if (math.isnan(f) or math.isinf(f)) else f
        except (ValueError, TypeError, OverflowError):
            return 0.0

    dict_a = eff_a if isinstance(eff_a, dict) else {}
    dict_b = eff_b if isinstance(eff_b, dict) else {}

    series: List[Dict[str, Any]] = [
        {
            "name": "Fertility",
            "values": [
                _chart_float(dict_a.get("fertility")),
                _chart_float(dict_b.get("fertility")),
            ],
        },
        {
            "name": "Token premium",
            "values": [
                _chart_float(dict_a.get("token_premium")),
                _chart_float(dict_b.get("token_premium")),
            ],
        },
    ]

    has_morph_a = bool(
        morph_a
        and isinstance(morph_a, dict)
        and (morph_a.get("n_words", 0) or 0) > 0
        and morph_a.get("f1") is not None
    )
    has_morph_b = bool(
        morph_b
        and isinstance(morph_b, dict)
        and (morph_b.get("n_words", 0) or 0) > 0
        and morph_b.get("f1") is not None
    )

    if has_morph_a and has_morph_b:
        series.append({
            "name": "MBI F1",
            "values": [
                _chart_float(morph_a.get("f1")),
                _chart_float(morph_b.get("f1")),
            ],
        })

    return {
        "labels": ["A", "B"],
        "series": series,
    }


def format_sample_header(token_count: int, text: str, max_chars: int = 90) -> str:
    """
    Format a clean header label for a sample sentence expander.

    Rationale:
    When listing sample tokenization results, long sentences cause horizontal overflow
    in expander titles. Truncating text to `max_chars` followed by a unicode ellipsis
    ('…') keeps the header concise while displaying the token count front and center.
    """
    clean_text = text if text is not None else ""
    if len(clean_text) > max_chars:
        truncated = clean_text[:max_chars] + "…"
    else:
        truncated = clean_text
    return f"{token_count} tokens — {truncated}"


def format_token_ids_preview(
    ids: Optional[Sequence[int]], max_show: int = 40
) -> str:
    """
    Format numerical token IDs into a preview string.

    Rationale:
    In debugging view, showing all hundreds of token IDs can crowd the UI. Truncating
    the list to `max_show` elements gives developers immediate visibility into token
    ID distributions without clutter.
    """
    if ids is None:
        return "[]"
    return str(list(ids[:max_show]))


def create_delta_dataframe(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    name_a: str = "Tokenizer A",
    name_b: str = "Tokenizer B",
) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame highlighting differences (deltas)
    between Tokenizer A and Tokenizer B.

    Rationale:
    Instead of making users manually subtract metric values between two rows,
    this delta table aligns metrics side-by-side with an explicit Delta (B - A)
    column. This provides immediate clarity on which tokenizer is more fertile,
    which compresses better, and which has superior morphological alignment.
    """
    # Canonical display mapping for metric keys
    display_configs = [
        ("token_count", "Tokens", 0),
        ("fertility", "Fertility", 3),
        ("compression", "Compression", 2),
        ("token_premium", "Token premium", 3),
        ("precision", "MBI Precision", 3),
        ("recall", "MBI Recall", 3),
        ("f1", "MBI F1", 3),
    ]

    records: List[Dict[str, Any]] = []
    handled_keys = set()

    for raw_key, label, decimals in display_configs:
        val_a = metrics_a.get(raw_key, metrics_a.get(label))
        val_b = metrics_b.get(raw_key, metrics_b.get(label))

        if val_a is None and val_b is None:
            continue

        handled_keys.add(raw_key)
        handled_keys.add(label)

        delta = None
        if val_a is not None and val_b is not None:
            try:
                fa = float(val_a)
                fb = float(val_b)
                diff = fb - fa
                if math.isnan(diff) or math.isinf(diff):
                    delta = diff
                elif decimals > 0:
                    delta = round(diff, decimals)
                else:
                    delta = int(round(diff))
            except (ValueError, TypeError, OverflowError):
                delta = None

        records.append({
            "Metric": label,
            name_a: val_a,
            name_b: val_b,
            "Delta": delta,
        })

    # Also handle any arbitrary custom metrics passed in dictionaries
    all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
    for key in sorted(all_keys - handled_keys, key=str):
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        delta = None
        if va is not None and vb is not None:
            try:
                fa = float(va)
                fb = float(vb)
                diff = fb - fa
                if math.isnan(diff) or math.isinf(diff):
                    delta = diff
                else:
                    delta = round(diff, 3)
            except (ValueError, TypeError, OverflowError):
                delta = None
        records.append({
            "Metric": str(key),
            name_a: va,
            name_b: vb,
            "Delta": delta,
        })

    return pd.DataFrame(records)
