#!/usr/bin/env python
# coding: utf-8
"""
Tokenizer Analysis Dashboard (TAD)
CSC3003S Capstone — Home · Inspect · Compare · Rank

Architectural Overview:
-----------------------
- Framework: Streamlit interactive analytical application.
- Services:
  * tokenizer_services.py: Hugging Face AutoTokenizer loading, caching & sanitized errors.
  * Efficiency_Metrics.py: Pure math formulas for Token count, Fertility, Compression, Token premium.
  * MBI.py: Morphological Boundary Identification against NCHLT gold standard corpora.
  * ranking_engine.py: Normalized, directionally inverted composite ranking with score transparency.
  * storage.py: Atomic local JSON persistence (data/leaderboard.json) with startup hydration & confirmed reset.
  * render_helpers.py: Decoupled formatting, badges, and delta comparison transformations.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

# Quiet noisy library warnings so the Streamlit UI stays clean and readable
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# Core analytics & persistence services
import ranking_engine
import render_helpers
import storage
# Data loaders + language list (FLORES / NCHLT)
from datasets import (
    LANGUAGE_NAMES,
    has_morphology,
    load_flores,
    load_nchlt_gold,
)
# Efficiency numbers: fertility, compression, token premium, etc.
from Efficiency_Metrics import compute_all
# Morphological boundary P/R/F1 against NCHLT gold
from MBI import evaluate_corpus
# Hugging Face load + tokenize helpers (with in-memory caching and error sanitization)
from tokenizer_services import load_tokenizer, tokenize_text

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
# Capstone UX Rationale — Why initial_sidebar_state="collapsed"?
# In complex analytical dashboards, horizontal screen real estate is the most precious resource.
# The TAD application uses a top horizontal navigation bar (streamlit_option_menu) across
# Home, Inspect, Compare, and Rank, rendering the traditional 25% grey sidebar redundant.
# Collapsing the sidebar by default immediately grants full 100% viewport width for side-by-side
# metric cards (4 columns), multi-language comparison delta tables, and Plotly charts, completely
# eliminating awkward horizontal scrolling and visual compression on standard laptops.
st.set_page_config(
    page_title="Tokenizer Analysis Dashboard",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
# Friendly label -> Hugging Face model id (shortlist for demos)
POPULAR_TOKENIZERS = {
    "BERT (multilingual)": "bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "DistilBERT (multilingual)": "distilbert-base-multilingual-cased",
    "GPT-2": "gpt2",
    "RoBERTa": "roberta-base",
    "T5": "google-t5/t5-base",
    "Llama 3.2 1B": "meta-llama/Llama-3.2-1B",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def init_session():
    """
    Initialize persistent session state attributes.

    Capstone Rationale — Why Session State Caching?
    Streamlit operates on an immediate-mode execution architecture: every widget interaction,
    tab switch, or checkbox toggle re-executes the entire Python script from line 1.
    Without caching active model objects (`tokenizer`, `tokenizer_b`) and multi-language
    corpus evaluation payloads (`inspect_results`, `cmp_results`) in `st.session_state`:
    1. Switching between 'Inspect' and 'Compare' would wipe all benchmarked data, forcing the
       user to sit through expensive re-tokenization loops over thousands of sentences.
    2. Changing a display toggle (like selecting a different Plotly bar chart metric) would
       trigger wasteful recomputation instead of rendering immediately (<10ms).
    Session state caching guarantees a smooth, responsive, desktop-grade user experience.
    """
    defaults = {
        "tokenizer": None,          # Tokenizer instance A (Inspect + Compare A)
        "tokenizer_name": None,     # HF model ID string for A
        "tokenizer_b": None,        # Tokenizer instance B (Compare only)
        "tokenizer_b_name": None,   # HF model ID string for B
        "inspect_results": None,    # Cached results for Inspect tab
        "cmp_results": None,        # Cached results for Compare tab
        "confirm_reset": False,     # Two-step confirmation flag for Reset Leaderboard
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Hydrate analyzed tokenizers registry from persistent store on startup.
    #
    # Capstone Rationale — Why the R2 Eligibility Gate?
    # Fulfills Requirement R2 (§3 & §3.1): A core academic tenet of the TAD project is that
    # tokenizers must only appear on the Rank leaderboard after they have been genuinely loaded
    # and evaluated against empirical test data in Inspect or Compare.
    # The Rank dashboard must never eagerly download models or populate speculative scores
    # for unanalyzed models. By hydrating `st.session_state["analyzed_tokenizers"]` from the
    # local JSON store (`data/leaderboard.json`) at startup, we restore the registry of
    # previously verified models without triggering any eager network requests or heavy recomputations.
    if "analyzed_tokenizers" not in st.session_state:
        persisted_entries = storage.load_leaderboard()
        st.session_state["analyzed_tokenizers"] = set(
            entry["model_id"]
            for entry in persisted_entries
            if isinstance(entry, dict) and "model_id" in entry
        )


def show_token_chips(tokens, max_show=60):
    # Draw subword pieces as inline code chips so splits are easy to see
    if not tokens:
        st.write("_No tokens_")
        return
    # Cap length so a long sentence doesn't dominate the page
    chips = " ".join(f"`{t}`" for t in tokens[:max_show])
    st.markdown(chips)
    if len(tokens) > max_show:
        st.caption(f"… and {len(tokens) - max_show} more")


def metric_row(efficiency: dict, morphology: dict | None = None):
    # Scorecards for efficiency (always) and MBI (only when we have gold data)
    cols = st.columns(4)
    cols[0].metric("Token count", f"{efficiency.get('token_count', 0):,}")
    cols[1].metric("Fertility", f"{efficiency.get('fertility', 0):.3f}")
    cols[2].metric("Compression (B/tok)", f"{efficiency.get('compression', 0):.2f}")
    # Premium needs parallel English; free text usually won't have it
    if "token_premium" in efficiency:
        cols[3].metric("Token premium", f"{efficiency['token_premium']:.3f}")
    else:
        cols[3].metric("Words", f"{efficiency.get('word_count', 0):,}")

    # Skip MBI block for English/Afrikaans or empty eval
    if morphology and morphology.get("n_words", 0) > 0:
        st.markdown("**Morphological Boundary Identification (MBI)**")
        mcols = st.columns(3)
        mcols[0].metric("Precision", f"{morphology['precision']:.3f}")
        mcols[1].metric("Recall", f"{morphology['recall']:.3f}")
        mcols[2].metric("F1", f"{morphology['f1']:.3f}")
        st.caption(
            f"Evaluated on {morphology['n_words']:,} words · "
            f"micro-F1 = {morphology.get('micro_f1', 0):.3f}"
        )


def tokenizer_picker(label: str, key_prefix: str) -> str | None:
    """Popular + custom HF ID picker. Returns model id or None."""
    # key_prefix keeps Inspect / Compare A / Compare B widgets from clashing
    st.markdown(f"**{label}**")
    names = list(POPULAR_TOKENIZERS.keys()) + ["Other (custom HF ID)"]
    choice = st.selectbox(
        "Popular tokenizers",
        names,
        key=f"{key_prefix}_popular",
    )
    if choice != "Other (custom HF ID)":
        return POPULAR_TOKENIZERS[choice]
    # Free-typed Hugging Face repo id
    custom = st.text_input(
        "Hugging Face model ID",
        placeholder="e.g. google/gemma-2b",
        key=f"{key_prefix}_custom",
    )
    return custom.strip() if custom and custom.strip() else None


def load_and_store(model_id: str, state_tok_key: str, state_name_key: str) -> bool:
    """
    Load tokenizer into session_state with robust, sanitized error handling.

    Why?
    Fulfills Requirement R4: If a model fails to download (e.g. invalid repo ID,
    offline environment, or network timeout), the UI displays a clean, actionable
    alert without dumping raw urllib/HuggingFace stack traces. Full exception details
    are safely routed to the backend console logger for developer diagnosis.
    """
    try:
        with st.spinner(f"Loading `{model_id}` …"):
            tok = load_tokenizer(model_id)
        st.session_state[state_tok_key] = tok
        st.session_state[state_name_key] = model_id
        st.success(f"Loaded **{model_id}**")
        return True
    except ValueError as e:
        # Sanitized user-friendly error raised by load_tokenizer
        st.session_state[state_tok_key] = None
        st.session_state[state_name_key] = None
        st.error(str(e))
        return False
    except Exception as e:
        # Catch-all for unexpected failures: log full exception to console while presenting clean UI message
        logger.exception("Unexpected error while loading tokenizer %s: %s", model_id, e)
        st.session_state[state_tok_key] = None
        st.session_state[state_name_key] = None
        st.error(f"Unable to load tokenizer '{model_id}'. Please check the model ID or internet connection.")
        return False


def register_analyzed_model(
    model_id: str,
    eff_results: Dict[str, Any],
    morph_results: Optional[Dict[str, Any]] = None,
    languages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregates multi-language evaluation metrics, computes composite ranking score,
    persists the tokenizer to data/leaderboard.json, and registers it in session state.

    Why aggregate across languages?
    The leaderboard ranks tokenizers on their holistic performance across South African
    languages. Computing the mean across evaluated languages balances cross-lingual
    fertility and compression with morphological alignment.
    """
    metrics_list = []
    for lang_data in eff_results.values():
        if isinstance(lang_data, dict) and "efficiency" in lang_data:
            metrics_list.append(lang_data["efficiency"])

    agg_metrics: Dict[str, float] = {}
    if metrics_list:
        for key in ("token_count", "fertility", "compression", "token_premium"):
            vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
            if vals:
                agg_metrics[key] = sum(vals) / len(vals)

    if morph_results:
        morph_list = [m for m in morph_results.values() if isinstance(m, dict) and m.get("n_words", 0) > 0]
        if morph_list:
            for key in ("precision", "recall", "f1"):
                vals = [m[key] for m in morph_list if key in m and m[key] is not None]
                if vals:
                    agg_metrics[f"mbi_{key}"] = sum(vals) / len(vals)

    # Compute composite score via centralized ranking engine
    score_info = ranking_engine.compute_composite_score(agg_metrics)

    entry = {
        "model_id": model_id,
        "display_name": model_id,
        "languages": list(languages or list(eff_results.keys())),
        "metrics": agg_metrics,
        "overall_score": score_info["overall_score"],
        "efficiency_score": score_info["efficiency_score"],
        "linguistic_score": score_info["linguistic_score"],
        "score_breakdown": score_info,
    }

    # Persist to local JSON store (atomic write)
    storage.add_or_update_entry(entry)

    # Register model in session state registry (eligibility gate for Rank)
    if "analyzed_tokenizers" not in st.session_state:
        st.session_state["analyzed_tokenizers"] = set()
    st.session_state["analyzed_tokenizers"].add(model_id)

    return entry


def run_efficiency_on_flores(tokenizer, language: str, max_sentences: int):
    # Pull parallel FLORES lines for the target language and for English
    sentences = load_flores(language, max_sentences=max_sentences)
    eng = load_flores("English", max_sentences=max_sentences)
    # Join into one blob so fertility/compression are corpus-level
    text = "\n".join(sentences)
    eng_text = "\n".join(eng)
    tok_result = tokenize_text(tokenizer, text)
    eng_result = tokenize_text(tokenizer, eng_text)
    # eng tokens are only used for token premium
    metrics = compute_all(text, tok_result["tokens"], eng_result["tokens"])
    # Keep a few sentence-level tokenizations for the UI samples
    samples = []
    for s in sentences[:5]:
        r = tokenize_text(tokenizer, s)
        samples.append({"text": s, "tokens": r["tokens"], "token_count": r["token_count"]})
    return metrics, samples, len(sentences)


def run_efficiency_multi(tokenizer, languages , max_sentences):
    results = {}
    for lang in languages:
        eff, samples, n = run_efficiency_on_flores(tokenizer, lang, max_sentences)
        results[lang] = {"efficiency": eff, "samples": samples, "n_sentences": n}
    return results 

    
def run_morphology(tokenizer, language: str, max_words: int):
    # No NCHLT gold for some languages — caller should treat None as "skip MBI"
    if not has_morphology(language):
        return None
    gold = load_nchlt_gold(language)
    return evaluate_corpus(tokenizer, gold, max_words=max_words)

def run_morphology_multi(tokenizer, languages, max_words):
    results = {}
    for lang in languages:
        if has_morphology(lang):
            results[lang] = run_morphology(tokenizer, lang, max_words)
    return results

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
init_session()

st.title("📌 Tokenizer Analysis Dashboard")

# Horizontal menu drives which page body runs below
selected = option_menu(
    menu_title=None,
    options=["Home", "Inspect", "Compare", "Rank"],
    icons=["house", "eye", "arrow-left-right", "trophy"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# =============================================================================
# HOME
# =============================================================================
if selected == "Home":
    # Tighten top padding a bit so the title sits closer to the menu
    st.markdown(
        """
        <style>div.block-container{padding-top:1.5rem;}</style>
        """,
        unsafe_allow_html=True,
    )

    # Short pitch: why tokenizers matter for SA languages
    st.markdown(
        """
        ### Analyse how LLM tokenizers treat South African languages

        Tokenizers split text into subword units before a language model sees it.
        Different tokenizers fragment languages very differently — especially
        morphologically rich ones like **isiZulu** and **isiXhosa**. That affects
        cost, context length, and how well meaning is preserved.
        """
    )

    # Three cards — one per mode
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**Inspect**\n\n"
            "Load one Hugging Face tokenizer and measure efficiency "
            "(FLORES) and morphological alignment (NCHLT)."
        )
    with c2:
        st.info(
            "**Compare**\n\n"
            "Run two tokenizers side-by-side on the same language "
            "and see fertility, premium, and MBI differences."
        )
    with c3:
        st.info(
            "**Rank**\n\n"
            "Compare and rank all analyzed tokenizers on an analytical "
            "leaderboard with transparent composite scoring and detailed metric breakdowns."
        )

    st.markdown("---")
    st.subheader("Supported languages")
    st.write(
        "Afrikaans · English · isiXhosa · isiZulu · Sesotho · "
        "Sepedi · Setswana · siSwati · Xitsonga"
    )

    st.subheader("Metrics")
    m1, m2 = st.columns(2)
    with m1:
        # Efficiency family (FLORES parallel text)
        st.markdown(
            """
            **Efficiency (FLORES-dev)**  
            - **Token count** — total pieces produced  
            - **Fertility** — tokens per word  
            - **Compression** — UTF-8 bytes per token  
            - **Token premium** — tokens(lang) / tokens(English) on parallel text  
            """
        )
    with m2:
        # Linguistic family (NCHLT gold boundaries)
        st.markdown(
            """
            **Linguistic (NCHLT)**  
            - **Morphological Boundary Identification (MBI)**  
            - Precision · Recall · F1  
            - Gold morpheme cuts vs tokenizer cuts between characters  
            """
        )

    st.subheader("Datasets")
    st.markdown(
        """
        | Dataset | Role |
        |---------|------|
        | **FLORES-dev** | Same 997 sentences translated into 9 SA languages — efficiency metrics |
        | **NCHLT** | Linguist-annotated morpheme splits — morphological alignment |

        Use the top navigation to open **Inspect**, **Compare**, or **Rank**.
        """
    )

# =============================================================================
# INSPECT
# =============================================================================
elif selected == "Inspect":
    st.markdown(
        '<style>div.block-container{padding-top:1.5rem;}</style>',
        unsafe_allow_html=True,
    )
    st.header("Inspect a tokenizer")
    st.caption(
        "Select a Hugging Face tokenizer, choose a language, and analyse "
        "default corpora or your own text."
    )

    # --- Tokenizer selection ---
    model_id = tokenizer_picker("Tokenizer", "inspect")
    col_load, col_status = st.columns([1, 2])
    with col_load:
        if st.button("Load tokenizer", type="primary", key="inspect_load"):
            if model_id:
                load_and_store(model_id, "tokenizer", "tokenizer_name")
            else:
                st.warning("Select or enter a tokenizer ID first.")
    with col_status:
        # Mirror whatever is currently in session_state
        if st.session_state.tokenizer_name:
            st.success(f"Active: `{st.session_state.tokenizer_name}`")
        else:
            st.warning("No tokenizer loaded yet.")

    st.markdown("---")

    # --- Language & source ---
    # Default to isiZulu so MBI is available out of the box in demos
    langs = st.multiselect("Languages", LANGUAGE_NAMES, default=["isiZulu"])
    source = st.radio(
        "Analysis source",
        ["Default FLORES corpus", "Manual text", "Upload text file"],
        horizontal=True,
    )


    # Uploading the whole dataset
    use_full = st.checkbox("Use full dataset", key="inspect_full")
    if use_full:
        max_sent = None
        max_morph = None
    else:
        # Smaller limits = faster interactive runs; user can raise them
        max_sent = st.slider("Max FLORES sentences", 50, 997, 150, 50)
        max_morph = st.slider("Max NCHLT words (morphology)", 50, 2000, 400, 50)


    # Source-specific inputs (shown before Run so values exist on click)
    manual_text = ""
    upload_text = ""
    if source == "Manual text":
        manual_text = st.text_area("Enter text to tokenize", height=160, key="inspect_manual_text")
    elif source == "Upload text file":
        up = st.file_uploader("Upload a UTF-8 `.txt` file", type=["txt"], key="inspect_up")
        if up is not None:
            # Replace undecodable bytes instead of crashing
            upload_text = up.read().decode("utf-8", errors="replace")
            st.caption(f"Loaded {len(upload_text):,} characters")

    run = st.button("Run analysis", type="primary", key="inspect_run")

    if run:
        if st.session_state.tokenizer is None:
            st.error("Load a tokenizer first.")
        elif not langs and source == "Default FLORES corpus":
            st.warning("Select at least one language.")
        else:
            tok = st.session_state.tokenizer
            name = st.session_state.tokenizer_name

            if source == "Default FLORES corpus":
                spinner_msg = (
                    "Tokenizing full dataset — this may take a while …"
                    if max_sent is None
                    else "Tokenizing FLORES and computing metrics …"
                )
                try:
                    with st.spinner(spinner_msg):
                        eff_results = run_efficiency_multi(tok, langs, max_sent)
                        morph_results = run_morphology_multi(tok, langs, max_morph)

                    # Auto-register analyzed model into persistent leaderboard store and session registry
                    # Why? Fulfills R2: Models become eligible for Rank only after being loaded & evaluated here
                    register_analyzed_model(name, eff_results, morph_results, langs)

                    # Cache results in session state so adjusting widgets doesn't wipe them
                    st.session_state["inspect_results"] = {
                        "source": "Default FLORES corpus",
                        "name": name,
                        "langs": langs,
                        "eff_results": eff_results,
                        "morph_results": morph_results,
                    }
                except Exception as e:
                    logger.exception("Failed to execute Inspect FLORES analysis for %s: %s", name, e)
                    st.error("An error occurred during dataset evaluation. Please check logs or reduce sentence count.")

            elif source == "Manual text":
                text = manual_text or ""
                if not text.strip():
                    st.warning("Enter some text above, then click Run analysis.")
                else:
                    try:
                        with st.spinner("Tokenizing …"):
                            result = tokenize_text(tok, text)
                            # No parallel English reference available for standalone manual text,
                            # so token premium is omitted while fertility and compression are computed
                            efficiency = compute_all(text, result["tokens"])
                        st.session_state["inspect_results"] = {
                            "source": "Manual text",
                            "name": name,
                            "result": result,
                            "efficiency": efficiency,
                        }
                    except Exception as e:
                        logger.exception("Failed to tokenize manual text for %s: %s", name, e)
                        st.error("Unable to tokenize input text. Please check the text input.")

            else:  # Upload text file
                text = upload_text or ""
                if not text.strip():
                    st.warning("Upload a UTF-8 text file above, then click Run analysis.")
                else:
                    try:
                        with st.spinner("Analysing uploaded corpus …"):
                            result = tokenize_text(tok, text)
                            efficiency = compute_all(text, result["tokens"])
                        st.session_state["inspect_results"] = {
                            "source": "Upload text file",
                            "name": name,
                            "result": result,
                            "efficiency": efficiency,
                        }
                    except Exception as e:
                        logger.exception("Failed to analyze uploaded file for %s: %s", name, e)
                        st.error("Unable to process uploaded file. Please ensure it is valid UTF-8 text.")

    # Render cached analysis results
    if st.session_state.get("inspect_results"):
        ires = st.session_state["inspect_results"]
        isource = ires.get("source")
        iname = ires.get("name")

        if isource == "Default FLORES corpus":
            ilangs = ires.get("langs", [])
            ieff = ires.get("eff_results", {})
            imorph = ires.get("morph_results", {})

            # 1. Cross-Language Summary Visualization (Grouped Bar Chart)
            # Why? Answers: "How do fertility and compression compare across languages?"
            # Visualizes language equity across agglutinative and morphologically varied SA languages.
            st.markdown("### Cross-Language Summary")
            fig = go.Figure()
            fertilities = [ieff[l]["efficiency"]["fertility"] for l in ilangs if l in ieff]
            compressions = [ieff[l]["efficiency"]["compression"] for l in ilangs if l in ieff]
            plot_langs = [l for l in ilangs if l in ieff]

            fig.add_trace(go.Bar(
                name="Fertility (tokens/word)",
                x=plot_langs,
                y=fertilities,
                marker_color="#1f77b4",
            ))
            fig.add_trace(go.Bar(
                name="Compression (bytes/token)",
                x=plot_langs,
                y=compressions,
                marker_color="#2ca02c",
            ))
            fig.update_layout(
                barmode="group",
                title=f"Fertility & Compression by Language — {iname}",
                xaxis_title="Language",
                yaxis_title="Metric Value",
                height=360,
                margin=dict(t=40, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 2. Per-Language Detail Tabs (one per selected language)
            # Capstone UX Rationale — Why Tabbed Multi-Language Navigation?
            # When evaluating tokenizers across up to 9 South African official languages simultaneously,
            # stacking 9 complete metric scorecards and sample tokenization chips vertically creates
            # excessive page height (several thousand pixels of vertical scroll). Users lose situational
            # awareness of the overarching model performance.
            # By structuring the language breakdown into compact tabs (`st.tabs(ilangs)`), users can quickly
            # switch between languages with a single click, keeping scorecards, MBI alignment metrics, and
            # expandable token chips in a clean, stationary viewport.
            st.markdown("### Language Details")
            if ilangs:
                lang_tabs = st.tabs(ilangs)
                for tab, lang in zip(lang_tabs, ilangs):
                    with tab:
                        result = ieff[lang]
                        morphology = imorph.get(lang)

                        # Formatted header: "{lang} — {name}" with em dash
                        st.subheader(f"{lang} — {iname}")
                        st.caption(f"FLORES sentences evaluated: {result['n_sentences']}")
                        metric_row(result["efficiency"], morphology)

                        st.markdown("##### Sample tokenizations")
                        for s in result["samples"][:3]:
                            with st.expander(
                                render_helpers.format_sample_header(s["token_count"], s["text"]),
                                expanded=False,
                            ):
                                show_token_chips(s["tokens"])

        elif isource == "Manual text":
            st.subheader(f"Manual Text Results — {iname}")
            metric_row(ires["efficiency"])
            st.markdown("#### Tokens")
            show_token_chips(ires["result"]["tokens"])
            st.markdown("#### Token IDs (first 40)")
            st.code(render_helpers.format_token_ids_preview(ires["result"]["ids"]))

        elif isource == "Upload text file":
            st.subheader(f"Uploaded Corpus Results — {iname}")
            metric_row(ires["efficiency"])
            st.markdown("#### First Tokens Preview")
            show_token_chips(ires["result"]["tokens"][:60])

# =============================================================================
# COMPARE
# =============================================================================
elif selected == "Compare":
    st.markdown(
        '<style>div.block-container{padding-top:1.5rem;}</style>',
        unsafe_allow_html=True,
    )
    st.header("Compare two tokenizers")
    st.caption("Side-by-side efficiency and morphological alignment on the same language.")

    # Two independent loaders; A uses tokenizer / tokenizer_name,
    # B uses tokenizer_b / tokenizer_b_name in session_state
    col_a, col_b = st.columns(2)
    with col_a:
        id_a = tokenizer_picker("Tokenizer A", "cmp_a")
        if st.button("Load A", key="load_a"):
            if id_a:
                load_and_store(id_a, "tokenizer", "tokenizer_name")
            else:
                st.warning("Select tokenizer A.")
        if st.session_state.tokenizer_name:
            st.caption(f"A active: `{st.session_state.tokenizer_name}`")

    with col_b:
        id_b = tokenizer_picker("Tokenizer B", "cmp_b")
        if st.button("Load B", key="load_b"):
            if id_b:
                load_and_store(id_b, "tokenizer_b", "tokenizer_b_name")
            else:
                st.warning("Select tokenizer B.")
        if st.session_state.tokenizer_b_name:
            st.caption(f"B active: `{st.session_state.tokenizer_b_name}`")

    st.markdown("---")
    langs = st.multiselect(
        "Languages",
        LANGUAGE_NAMES,
        default=["isiZulu"],
        key="cmp_lang",
    )
    use_full = st.checkbox("Use full dataset", key="cmp_full")
    if use_full:
        max_sent = None
        max_morph = None
    else:
        max_sent = st.slider("Max FLORES sentences", 50, 997, 100, 50, key="cmp_sent")
        max_morph = st.slider("Max NCHLT words", 50, 2000, 300, 50, key="cmp_morph")

    if st.button("Compare", type="primary", key="cmp_run"):
        if st.session_state.tokenizer is None or st.session_state.tokenizer_b is None:
            st.error("Load **both** tokenizers before comparing.")
        elif not langs:
            st.warning("Select at least one language.")
        else:
            tok_a = st.session_state.tokenizer
            tok_b = st.session_state.tokenizer_b
            name_a = st.session_state.tokenizer_name
            name_b = st.session_state.tokenizer_b_name

            spinner_msg = (
                "Tokenizing full dataset for both tokenizers — this may take a while …"
                if max_sent is None
                else "Analysing both tokenizers across selected languages …"
            )
            try:
                # Same language + same sentence budget for a fair, controlled comparison
                with st.spinner(spinner_msg):
                    eff_a_multi = run_efficiency_multi(tok_a, langs, max_sent)
                    eff_b_multi = run_efficiency_multi(tok_b, langs, max_sent)
                    morph_a_multi = run_morphology_multi(tok_a, langs, max_morph) 
                    morph_b_multi = run_morphology_multi(tok_b, langs, max_morph) 

                # Auto-register and persist BOTH evaluated models into persistent leaderboard store
                # Why? Fulfills R2: Comparing models produces genuine benchmarks that make both models
                # eligible for leaderboard ranking.
                register_analyzed_model(name_a, eff_a_multi, morph_a_multi, langs)
                register_analyzed_model(name_b, eff_b_multi, morph_b_multi, langs)

                st.session_state["cmp_results"] = {
                    "langs": langs,
                    "name_a": name_a,
                    "name_b": name_b,
                    "eff_a_multi": eff_a_multi,
                    "eff_b_multi": eff_b_multi,
                    "morph_a_multi": morph_a_multi,
                    "morph_b_multi": morph_b_multi,
                }
            except Exception as e:
                logger.exception("Failed to execute Compare analysis between %s and %s: %s", name_a, name_b, e)
                st.error("An error occurred during tokenizer comparison. Please check logs or reduce sentence count.")

    if "cmp_results" in st.session_state and st.session_state["cmp_results"]:
        r = st.session_state["cmp_results"]
        langs = r["langs"]
        name_a, name_b = r["name_a"], r["name_b"]
        eff_a_multi, eff_b_multi = r["eff_a_multi"], r["eff_b_multi"]
        morph_a_multi, morph_b_multi = r["morph_a_multi"], r["morph_b_multi"]

        st.subheader("Side-by-Side Comparison")

        # Aggregate / Overall delta comparison across selected languages
        # Why? Provides an executive summary delta table so users do not need to mentally
        # compute differences across multiple language tabs.
        agg_a: Dict[str, float] = {}
        agg_b: Dict[str, float] = {}
        for key in ("token_count", "fertility", "compression", "token_premium"):
            vals_a = [eff_a_multi[l]["efficiency"][key] for l in langs if key in eff_a_multi[l]["efficiency"] and eff_a_multi[l]["efficiency"][key] is not None]
            vals_b = [eff_b_multi[l]["efficiency"][key] for l in langs if key in eff_b_multi[l]["efficiency"] and eff_b_multi[l]["efficiency"][key] is not None]
            if vals_a:
                agg_a[key] = sum(vals_a) / len(vals_a)
            if vals_b:
                agg_b[key] = sum(vals_b) / len(vals_b)

        # Morphology F1/Precision/Recall across languages with NCHLT gold standard data
        morph_langs = [l for l in langs if has_morphology(l) and l in morph_a_multi and l in morph_b_multi]
        if morph_langs:
            for key in ("precision", "recall", "f1"):
                v_a = [morph_a_multi[l][key] for l in morph_langs if key in morph_a_multi[l] and morph_a_multi[l][key] is not None]
                v_b = [morph_b_multi[l][key] for l in morph_langs if key in morph_b_multi[l] and morph_b_multi[l][key] is not None]
                if v_a:
                    agg_a[f"mbi_{key}"] = sum(v_a) / len(v_a)
                if v_b:
                    agg_b[f"mbi_{key}"] = sum(v_b) / len(v_b)

        # Render Delta table for summary metrics
        st.markdown("#### Overall Average Metrics & Deltas (B − A)")
        summary_df = render_helpers.create_delta_dataframe(
            agg_a, agg_b, name_a=f"A: {name_a}", name_b=f"B: {name_b}"
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Grouped bar chart by language
        # Why? Answers: "Where does one tokenizer outperform another across individual languages?"
        metric_options = ["Fertility", "Compression", "Token premium"]
        if any(has_morphology(l) for l in langs):
            metric_options.append("MBI F1")

        metric = st.selectbox("Metric to plot across languages", metric_options, key="cmp_metric")
        metric_key = {
            "Fertility": "fertility",
            "Compression": "compression",
            "Token premium": "token_premium",
        }

        if metric == "MBI F1":
            # Only languages with NCHLT gold data can plot F1
            chart_langs = [l for l in langs if has_morphology(l)]
            y_a = [morph_a_multi[l]["f1"] for l in chart_langs]
            y_b = [morph_b_multi[l]["f1"] for l in chart_langs]
        else:
            chart_langs = langs
            key = metric_key[metric]
            y_a = [eff_a_multi[l]["efficiency"].get(key, 0) for l in chart_langs]
            y_b = [eff_b_multi[l]["efficiency"].get(key, 0) for l in chart_langs]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=f"A: {name_a}", x=chart_langs, y=y_a, marker_color="#1f77b4"))
        fig.add_trace(go.Bar(name=f"B: {name_b}", x=chart_langs, y=y_b, marker_color="#ff7f0e"))
        fig.update_layout(
            barmode="group",
            title=f"{metric} Comparison by Language",
            xaxis_title="Language",
            yaxis_title=metric,
            height=380,
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Language-specific comparison tabs
        # Why? Breaking down by tab prevents vertical scroll fatigue while grouping delta analysis
        # alongside collapsed sample tokenizations.
        st.markdown("#### Language-Specific Comparison & Sample Tokens")
        cmp_tabs = st.tabs(langs)
        for tab, lang in zip(cmp_tabs, langs):
            with tab:
                eff_a = eff_a_multi[lang]["efficiency"]
                eff_b = eff_b_multi[lang]["efficiency"]
                m_a = morph_a_multi.get(lang) or {}
                m_b = morph_b_multi.get(lang) or {}

                metrics_lang_a = dict(eff_a)
                metrics_lang_b = dict(eff_b)
                if m_a:
                    metrics_lang_a.update(m_a)
                if m_b:
                    metrics_lang_b.update(m_b)

                lang_delta_df = render_helpers.create_delta_dataframe(
                    metrics_lang_a, metrics_lang_b, name_a=f"A: {name_a}", name_b=f"B: {name_b}"
                )
                st.dataframe(lang_delta_df, use_container_width=True, hide_index=True)

                # Collapsed sample chips into expanders per language
                # Why? Capping DOM footprint and preventing huge paragraphs from taking over the screen.
                samples_a = eff_a_multi[lang]["samples"]
                samples_b = eff_b_multi[lang]["samples"]
                with st.expander(f"View Sample Tokenizations ({lang})", expanded=False):
                    for i in range(min(3, len(samples_a), len(samples_b))):
                        st.markdown(f"**Text:** {samples_a[i]['text'][:120]}…")
                        ca, cb = st.columns(2)
                        with ca:
                            st.caption(f"A · {samples_a[i]['token_count']} tokens · `{name_a}`")
                            show_token_chips(samples_a[i]["tokens"])
                        with cb:
                            st.caption(f"B · {samples_b[i]['token_count']} tokens · `{name_b}`")
                            show_token_chips(samples_b[i]["tokens"])

# =============================================================================
# RANK (Full Analytical Leaderboard)
# =============================================================================
elif selected == "Rank":
    st.markdown(
        '<style>div.block-container{padding-top:1.5rem;}</style>',
        unsafe_allow_html=True,
    )
    st.header("🏆 Tokenizer Benchmark Leaderboard")
    st.caption(
        "Permanent multi-tokenizer benchmark ranking models on efficiency "
        "and morphological alignment across South African languages."
    )

    # --- Confirmable Reset Leaderboard ---
    # Capstone Rationale — Why a 2-Step Confirmation Dialog?
    # Fulfills Requirement R3 & §5: The local JSON store (`data/leaderboard.json`) persists
    # tokenizer evaluations across sessions. Benchmarking multiple models over FLORES and NCHLT
    # represents significant compute time and analytical effort.
    # An unconfirmed single-click reset button would introduce catastrophic risk: an accidental
    # click would irreversibly wipe all benchmarking history.
    # The two-step confirmation state machine (`confirm_reset`: False -> True -> Action) requires
    # deliberate user confirmation ("Are you sure?"), protecting user data integrity while
    # offering an immediate 'Cancel' escape hatch.
    if "confirm_reset" not in st.session_state:
        st.session_state["confirm_reset"] = False

    col_reset, _ = st.columns([2, 5])
    with col_reset:
        if not st.session_state["confirm_reset"]:
            if st.button("🗑️ Reset Leaderboard", type="secondary", key="btn_request_reset"):
                st.session_state["confirm_reset"] = True
                st.rerun()
        else:
            st.warning("Are you sure? This will clear all persisted rankings.")
            c_confirm, c_cancel = st.columns(2)
            with c_confirm:
                if st.button("Confirm Reset", type="primary", key="btn_confirm_reset"):
                    storage.clear_leaderboard()
                    if "analyzed_tokenizers" in st.session_state:
                        st.session_state["analyzed_tokenizers"] = set()
                    st.session_state["confirm_reset"] = False
                    st.success("Leaderboard has been reset.")
                    st.rerun()
            with c_cancel:
                if st.button("Cancel", key="btn_cancel_reset"):
                    st.session_state["confirm_reset"] = False
                    st.rerun()

    st.markdown("---")

    # Strict No-Eager-Loading Policy (§3.1):
    # Retrieve persisted tokenizers directly from disk. NEVER trigger Hugging Face
    # downloads, re-tokenizations, or network requests on the Rank page.
    leaderboard_entries = storage.load_leaderboard()

    if not leaderboard_entries:
        # Friendly empty state (§12)
        st.info(
            "No tokenizers on the leaderboard yet. "
            "Load and analyze tokenizers in Inspect or Compare to rank them."
        )
    else:
        # Deterministic ranking calculation via ranking_engine (§2.2)
        ranked_models = ranking_engine.rank_tokenizers(leaderboard_entries)

        # Render scannable leaderboard table (§6)
        table_data = []
        for entry in ranked_models:
            r = entry.get("rank", "-")
            badge = render_helpers.get_rank_badge(r)
            model_name = entry.get("display_name") or entry.get("model_id", "Unknown")
            overall = render_helpers.format_metric(entry.get("overall_score"), 2)
            eff = render_helpers.format_metric(entry.get("efficiency_score"), 2)
            ling = entry.get("linguistic_score")
            ling_str = render_helpers.format_metric(ling, 2) if ling is not None else "N/A"
            langs_evaluated = ", ".join(entry.get("languages", [])) or "All Selected"

            table_data.append({
                "Rank": badge,
                "Model": model_name,
                "Overall Score (0–100)": overall,
                "Efficiency Score (0–100)": eff,
                "Linguistic Score (0–100)": ling_str,
                "Languages Evaluated": langs_evaluated,
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # --- Score Breakdown Expander ---
        # Why? Fulfills score transparency requirement (R1 & §2.3).
        # Users can inspect exactly how normalized metrics, inverted directions,
        # and weights combined into the final score for any model.
        st.markdown("### Score Breakdown & Ranking Transparency")
        with st.expander("🔍 View Granular Scoring Calculations & Methodology", expanded=False):
            st.markdown(
                """
                **Composite Ranking Methodology:**
                - **Efficiency (50%)**: Balances Token Count, Fertility, Compression Ratio, and Token Premium.
                - **Linguistic Alignment (50%)**: Evaluates Morphological Boundary Identification (MBI) Precision, Recall, and F1 against linguist-annotated NCHLT gold segmentations.
                - **Directional Inversion**: Lower-is-better metrics (Fertility, Token Count, Token Premium) are inverted ($1.0 - \\text{normalized}$) so that higher scores always represent superior models.
                - **Missing Data Fallback**: When morphology data is unavailable (e.g. English, Afrikaans), 100% weight is dynamically reallocated to efficiency to avoid penalizing models.
                """
            )
            model_names = [e.get("model_id") for e in ranked_models]
            selected_mid = st.selectbox("Select model to inspect score breakdown:", model_names, key="rank_model_select")
            selected_entry = next((e for e in ranked_models if e.get("model_id") == selected_mid), ranked_models[0])

            breakdown = selected_entry.get("score_breakdown")
            if not breakdown:
                breakdown = ranking_engine.compute_composite_score(selected_entry.get("metrics", {}))

            m_cols = st.columns(3)
            m_cols[0].metric("Overall Composite", f"{render_helpers.format_metric(breakdown.get('overall_score'), 2)} / 100")
            m_cols[1].metric("Efficiency Component", f"{render_helpers.format_metric(breakdown.get('efficiency_score'), 2)} / 100")
            ling_score = breakdown.get("linguistic_score")
            m_cols[2].metric("Linguistic Component", f"{render_helpers.format_metric(ling_score, 2)} / 100" if ling_score is not None else "N/A (Skipped)")

            # Granular sub-score breakdown table
            raw_m = selected_entry.get("metrics", {})
            norm_m = breakdown.get("normalized_metrics", {})
            inv_m = breakdown.get("inverted_metrics", {})
            bd_rows = []
            for k, v in raw_m.items():
                direction = ranking_engine.METRIC_DIRECTION.get(k.lower(), "higher")
                dir_desc = "Lower is better (inverted)" if direction == "lower" else "Higher is better"
                cat = "Efficiency" if k.lower() in ("token_count", "fertility", "compression", "token_premium") else "Linguistic"
                bd_rows.append({
                    "Category": cat,
                    "Metric": k,
                    "Raw Value": render_helpers.format_metric(v, 3) if isinstance(v, float) else str(v),
                    "Directionality": dir_desc,
                    "Normalized [0.0–1.0]": render_helpers.format_metric(norm_m.get(k), 4),
                    "Inverted Score [0.0–1.0]": render_helpers.format_metric(inv_m.get(k), 4),
                })
            if bd_rows:
                st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)

            weights_used = breakdown.get("weights_used", {})
            missing = breakdown.get("missing_components", [])
            if "linguistic" in missing:
                st.info("ℹ️ Note: NCHLT morphology gold data was not available for the evaluated languages of this model. Weight was fairly reallocated 100% to efficiency.")
            else:
                st.caption(f"Category Weights Applied: Efficiency {weights_used.get('efficiency', 0.5) * 100:.0f}% · Linguistic {weights_used.get('linguistic', 0.5) * 100:.0f}%")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "TAD · CSC3003S Capstone · Efficiency on FLORES-dev · "
    "Morphological Boundary Identification on NCHLT."
)