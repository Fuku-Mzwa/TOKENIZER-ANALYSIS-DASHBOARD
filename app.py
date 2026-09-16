#!/usr/bin/env python
# coding: utf-8
"""
Tokenizer Analysis Dashboard (TAD)
CSC3003S Capstone — Home · Inspect · Compare
"""
from __future__ import annotations

import warnings

# Quiet noisy library warnings so the Streamlit UI stays readable
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

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
# Hugging Face load + tokenize helpers (with caching)
from tokenizer_services import load_tokenizer, tokenize_text

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
# Wide layout gives Inspect/Compare room for metrics and charts
st.set_page_config(
    page_title="Tokenizer Analysis Dashboard",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded",
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
    # Streamlit reruns the script on every click; session_state keeps
    # loaded tokenizers alive so we don't download them again each time
    defaults = {
        "tokenizer": None,       # tokenizer A (Inspect + Compare A)
        "tokenizer_name": None,  # HF id string for A
        "tokenizer_b": None,     # tokenizer B (Compare only)
        "tokenizer_b_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    """Load tokenizer into session_state. Returns True on success."""
    # Wraps tokenizer_services.load_tokenizer and stores the result for later runs
    try:
        with st.spinner(f"Loading `{model_id}` …"):
            tok = load_tokenizer(model_id)
        st.session_state[state_tok_key] = tok
        st.session_state[state_name_key] = model_id
        st.success(f"Loaded **{model_id}**")
        return True
    except ValueError as e:
        # Bad id or download failure — clear any half-set state
        st.session_state[state_tok_key] = None
        st.session_state[state_name_key] = None
        st.error(str(e))
        return False


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
            "Leaderboard of multiple tokenizers — *coming soon*."
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

        Use the top navigation to open **Inspect** or **Compare**.
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
        elif not langs:
            st.warning("Select at least one language.")
        else:
            tok = st.session_state.tokenizer
            name = st.session_state.tokenizer_name

            if source == "Default FLORES corpus":
                with st.spinner("Tokenizing FLORES and computing metrics …"):
                    eff_results = run_efficiency_multi(tok, langs, max_sent)
                    morph_results = run_morphology_multi(tok, langs, max_morph)

                for lang in langs:
                    result = eff_results[lang]
                    morphology = morph_results.get(lang)

                    st.subheader(f"{lang} . '{name}'")
                    st.caption(f"FLORES sentences used: {result['n_sentences']}")
                    metric_row(result["efficiency"], morphology)

                    st.markdown("### Sample tokenizations")
                    for s in result["samples"][:3]:
                        with st.expander(
                            f"{s['token_count']} tokens - {s['text'][:90]}..."
                        ):
                            show_token_chips(s["tokens"])
                    
                    st.markdown("---")

                

            elif source == "Manual text":
                text = manual_text or ""
                if not text.strip():
                    st.warning("Enter some text above, then click Run analysis.")
                else:
                    with st.spinner("Tokenizing …"):
                        result = tokenize_text(tok, text)
                        # No English baseline here, so no token premium
                        efficiency = compute_all(text, result["tokens"])
                    st.subheader("Results")
                    metric_row(efficiency)
                    st.markdown("#### Tokens")
                    show_token_chips(result["tokens"])
                    st.markdown("#### Token IDs (first 40)")
                    st.code(str(result["ids"][:40]))

            else:  # Upload
                text = upload_text or ""
                if not text.strip():
                    st.warning("Upload a UTF-8 text file above, then click Run analysis.")
                else:
                    with st.spinner("Analysing uploaded corpus …"):
                        result = tokenize_text(tok, text)
                        efficiency = compute_all(text, result["tokens"])
                    st.subheader("Results")
                    metric_row(efficiency)
                    st.markdown("#### First tokens")
                    # Preview only — full file can be huge
                    show_token_chips(result["tokens"][:60])

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

            # Same language + same sentence budget for a fair comparison
            with st.spinner("Analysing both tokenizers …"):
                eff_a_multi = run_efficiency_multi(tok_a, langs, max_sent)
                eff_b_multi = run_efficiency_multi(tok_b, langs, max_sent)
                morph_a_multi = run_morphology_multi(tok_a, langs, max_morph) 
                morph_b_multi = run_morphology_multi(tok_b, langs, max_morph) 
            
            st.session_state["cmp_results"] = {
                "langs": langs,
                "name_a": name_a,
                "name_b": name_b,
                "eff_a_multi": eff_a_multi,
                "eff_b_multi": eff_b_multi,
                "morph_a_multi": morph_a_multi,
                "morph_b_multi": morph_b_multi,
            }

    if "cmp_results" in st.session_state:
        r =  st.session_state["cmp_results"]
        langs = r["langs"]
        name_a, name_b = r["name_a"], r["name_b"]
        eff_a_multi, eff_b_multi = r["eff_a_multi"], r["eff_b_multi"]
        morph_a_multi, morph_b_multi = r["morph_a_multi"], r["morph_b_multi"]

        st.subheader("Comparison")

        # Build a small table: one row per tokenizer
        rows = []
        for lang in langs:
            for label, name, eff, morph in (
                ("A", name_a, eff_a_multi[lang]["efficiency"], morph_a_multi.get(lang)),
                ("B", name_b,  eff_b_multi[lang]["efficiency"], morph_b_multi.get(lang)),
            ):
                row = {
                    "Language": lang,
                    "Tokenizer": f"{label}: {name}",
                    "Tokens": eff["token_count"],
                    "Fertility": round(eff["fertility"], 3),
                    "Compression": round(eff["compression"], 2),
                    "Token premium": round(eff.get("token_premium", 0), 3),
                }
                if morph:
                    row["MBI Precision"] = round(morph["precision"], 3)
                    row["MBI Recall"] = round(morph["recall"], 3)
                    row["MBI F1"] = round(morph["f1"], 3)
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Grouped bars for the metrics people care about most in a demo
        metric_options = ["Fertility", "Compression" , "Token premium"]
        if any(has_morphology(l) for l in langs):
            metric_options.append("MBI F1")
        
        metric = st.selectbox("Metric to plot", metric_options, key="cmp_metric")

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
        fig.add_trace(go.Bar(name=f"A: {name_a}", x=chart_langs, y=y_a))
        fig.add_trace(go.Bar(name=f"B: {name_b}", x=chart_langs, y=y_b))
        fig.update_layout(
            barmode="group",
            title=f"{metric} by language",
            height=380,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

       
        st.markdown("#### Sample tokenizations")
        for lang in langs:
            samples_a = eff_a_multi[lang]["samples"]
            samples_b = eff_b_multi[lang]["samples"]
            st.markdown(f"**{lang}**")
            for i in range(min(3, len(samples_a), len(samples_b))):
                st.markdown(f"**Text:** {samples_a[i]['text'][:120]}…")
                ca, cb = st.columns(2)
                with ca:
                    st.caption(f"A · {samples_a[i]['token_count']} tokens · `{name_a}`")
                    show_token_chips(samples_a[i]["tokens"])
                with cb:
                    st.caption(f"B · {samples_b[i]['token_count']} tokens · `{name_b}`")
                    show_token_chips(samples_b[i]["tokens"])
            st.markdown("---")
# =============================================================================
# RANK (skipped for now)
# =============================================================================
elif selected == "Rank":
    st.markdown(
        '<style>div.block-container{padding-top:1.5rem;}</style>',
        unsafe_allow_html=True,
    )
    st.header("Rank")
    # Left as a placeholder for a later milestone
    st.info(
        "The **Rank / leaderboard** feature is not implemented yet. "
        "Use **Inspect** and **Compare** for now."
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "TAD · CSC3003S Capstone · Efficiency on FLORES-dev · "
    "Morphological Boundary Identification on NCHLT"
)