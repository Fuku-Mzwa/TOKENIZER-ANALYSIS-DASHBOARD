# Tokenizer Analysis Dashboard (TAD)

**CSC3003S Capstone Project** — An interactive, evidence-grounded analytical platform for evaluating and benchmarking Hugging Face subword tokenizers on South African official languages.

---

## Table of Contents

- [Overview](#overview)
- [Application Modes](#application-modes)
- [Supported Languages & Datasets](#supported-languages--datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Ranking Engine & Methodology](#ranking-engine--methodology)
- [Leaderboard Persistence & Reset](#leaderboard-persistence--reset)
- [Project Architecture & Directory Layout](#project-architecture--directory-layout)
- [Installation & Quick Start](#installation--quick-start)

---

## Overview

The **Tokenizer Analysis Dashboard (TAD)** addresses the linguistic and economic disparities faced by low-resource and agglutinative languages in modern Large Language Models (LLMs). While state-of-the-art tokenizers are optimized heavily for English and major European languages, they frequently over-segment South African languages—inflating token counts, multiplying API inference costs, consuming quadratic attention context windows, and splitting words across unnatural morphological boundaries.

TAD provides a rigorous, transparent platform to:
1. **Quantify tokenization efficiency** (fertility, byte compression, token premium vs. English).
2. **Evaluate linguistic fidelity** against linguist-annotated morphological gold standards (NCHLT).
3. **Rank and benchmark tokenizers** using a normalized, multi-axis composite leaderboard.

---

## Application Modes

| Mode | Description |
| :--- | :--- |
| **🏠 Home** | Project overview, architectural motivation, language coverage summary, and metric reference guide. |
| **🔍 Inspect** | In-depth single-tokenizer evaluation. Analyze multi-language FLORES corpora, custom free-text, or uploaded text files. Features cross-language grouped visualizations, scorecards, and expandable token chips. |
| **⚖️ Compare** | Side-by-side comparative analysis of two tokenizers on identical corpora. Features automatic delta computation ($B - A$), cross-language grouped metric plots, and differential sample inspection. |
| **🏆 Rank** | Comprehensive benchmark leaderboard ranking evaluated tokenizers on efficiency and morphology. Features podium badges, sortable tables, granular scoring breakdowns, and atomic persistence. |

---

## Supported Languages & Datasets

TAD evaluates tokenizers across **9 South African official languages** alongside English:

| Language | Family / Typology | Morphology Gold Data (NCHLT) | Parallel Corpus (FLORES-dev) |
| :--- | :--- | :---: | :---: |
| **isiZulu** | Nguni (Agglutinative / Conjunctive) | ✅ Yes | ✅ 997 sentences |
| **isiXhosa** | Nguni (Agglutinative / Conjunctive) | ✅ Yes | ✅ 997 sentences |
| **siSwati** | Nguni (Agglutinative / Conjunctive) | ✅ Yes | ✅ 997 sentences |
| **Sesotho** | Sotho-Tswana (Disjunctive) | ✅ Yes | ✅ 997 sentences |
| **Sepedi** | Sotho-Tswana (Disjunctive) | ✅ Yes | ✅ 997 sentences |
| **Setswana** | Sotho-Tswana (Disjunctive) | ✅ Yes | ✅ 997 sentences |
| **Xitsonga** | Tswa-Ronga (Agglutinative) | ✅ Yes | ✅ 997 sentences |
| **Afrikaans** | West Germanic | ℹ️ *Skipped (No gold)* | ✅ 997 sentences |
| **English** | West Germanic (Baseline) | ℹ️ *Baseline reference* | ✅ 997 sentences (Parallel baseline) |

---

## Evaluation Metrics

### 1. Efficiency Metrics (`Efficiency_Metrics.py`)
- **Token Count**: Total subword tokens produced on a corpus split. Fewer tokens mean higher throughput.
- **Fertility Ratio**: $\frac{\text{tokens}}{\text{word}}$. Average subword tokens per whitespace-separated word. Measures over-segmentation.
- **Compression Ratio**: $\frac{\text{UTF-8 bytes}}{\text{token}}$. Byte packing density. Higher values indicate denser, more efficient subword units.
- **Token Premium**: $\frac{\text{tokens}(\text{Target Language})}{\text{tokens}(\text{English})}$ on parallel sentences. Quantifies the unfair "fertility tax" imposed on African language speakers compared to English. A premium of $2.5\times$ means an African speaker pays $2.5\times$ more money and latency for the same semantic content.

### 2. Linguistic Alignment Metrics (`MBI.py`)
- **Morphological Boundary Identification (MBI)**: Evaluates whether subword tokenizer cut sites align with linguistic morpheme boundaries annotated in the NCHLT gold standard corpora.
- Computed as binary classification over character cut candidates:
  - **Precision**: $\frac{TP}{TP + FP}$ — Proportion of tokenizer cuts that correspond to real morpheme boundaries.
  - **Recall**: $\frac{TP}{TP + FN}$ — Proportion of real morpheme boundaries captured by the tokenizer.
  - **F1 Score**: Harmonic mean $\frac{2 \cdot P \cdot R}{P + R}$. Reflects grammatical respect for concordial prefixes, roots, and verbal suffixes.

---

## Ranking Engine & Methodology (`ranking_engine.py`)

The ranking engine combines disparate metrics into an invariant, intuitive composite score $[0.0 - 100.0]$:

### 1. Fixed Empirical Anchor Normalization
Metrics operate on vastly different scales (e.g. token counts in thousands vs. F1 in $[0.0, 1.0]$). Instead of dynamic min-max scaling (which causes score jitter whenever new models are added), TAD uses fixed empirical reference bounds:
- **Fertility**: $[1.0, 5.0]$ (1.0 = ideal 1 token/word; 5.0 = severe over-fragmentation/character spelling).
- **Compression**: $[1.0, 6.0]$ bytes/token (1.0 = byte-fallback; 6.0 = dense morpheme packing).
- **Token Count**: $[0, 10000]$ tokens (standard FLORES slice ceiling).
- **Token Premium**: $[1.0, 4.0]$ (1.0 = cross-lingual parity; 4.0 = severe 4x penalty).
- **Morphological P/R/F1**: $[0.0, 1.0]$.

### 2. Directional Inversion (Lower is Better)
For **Token Count**, **Fertility**, and **Token Premium**, lower values represent superior efficiency. These metrics are normalized and inverted via:
$$\text{norm}_{\text{inv}} = 1.0 - \text{norm}$$
This maps the best performance to $1.0$ and the poorest to $0.0$, ensuring all metrics monotonically increase with desirability.

### 3. Dual-Axis 50/50 Weighted Composition
To prevent models from "gaming" the benchmark by memorizing huge vocabularies (destroying generalization) or tokenizing at character levels (destroying efficiency), the composite score balances efficiency and linguistic integrity equally:
$$\text{Overall Score} = (0.50 \times \text{Efficiency Score}) + (0.50 \times \text{Linguistic Score})$$
- **Efficiency (50%)**: Token Count (12.5%), Fertility (12.5%), Compression (12.5%), Token Premium (12.5%).
- **Linguistic (50%)**: MBI F1 (25%), MBI Precision (12.5%), MBI Recall (12.5%).

### 4. Dynamic Missing-Component Fallback
Languages like English and Afrikaans lack NCHLT gold standard morphology annotations. Rather than penalizing models with $0$ linguistic points (halving their score) or disqualifying them, the ranking engine dynamically reallocates $100\%$ weight to efficiency. The score breakdown explicitly flags `"missing_components": ["linguistic"]` for full transparency.

---

## Leaderboard Persistence & Reset (`storage.py`)

### 1. Local Zero-Dependency JSON Storage
Persistent state is stored in `data/leaderboard.json` using Python's standard library (`json`, `tempfile`, `os`, `pathlib`). This eliminates external database service dependencies, ensuring effortless portability across developer and evaluator machines.

### 2. Windows NTFS Durability via Atomic Writes
To prevent file corruption during server reloads, script reruns, or unexpected power interruptions, updates follow a strict atomic sequence:
1. `tempfile.mkstemp` creates a temporary file in the *same* directory (`data/`), ensuring identical filesystem partition allocation.
2. Content is serialized and flushed to disk with `os.fsync(fd)` to commit hardware buffers.
3. The temporary file is explicitly closed to satisfy Windows NTFS mandatory file-locking semantics.
4. `os.replace` atomically renames the temporary file over `data/leaderboard.json`.

### 3. Graceful Corruption Recovery
If the storage file is truncated, damaged, or contains malformed syntax, `load_leaderboard()` logs a detailed diagnostic warning via `logger.warning` and safely falls back to `[]`. Subsequent saves automatically rehabilitate the store without crashing the UI.

### 4. R2 Tokenizer Availability Gate & No Eager Loading
- **Eligibility Gate**: Tokenizers only become eligible for ranking after they have been genuinely loaded and analyzed in **Inspect** or **Compare**.
- **No Eager Loading**: Opening the Rank dashboard never downloads Hugging Face models or executes heavy tokenization loops; all leaderboard data is loaded directly from stored benchmark records.

### 5. Confirmable Leaderboard Reset
Clearing the leaderboard destroys accumulated benchmark records. A two-step state machine (`confirm_reset`: False $\rightarrow$ True $\rightarrow$ Action) prompts the user with an explicit confirmation warning before purging `data/leaderboard.json`.

---

## Project Architecture & Directory Layout

```text
tokenizer-analysis-dashboard/
├── app.py                      # Main Streamlit dashboard application (run this)
├── ranking_engine.py           # Centralized ranking methodology, normalization & composite scoring
├── storage.py                  # Local JSON persistence, atomic writes & reset operations
├── render_helpers.py           # Decoupled display, formatting & delta transformation helpers
├── tokenizer_services.py       # HF AutoTokenizer loader, in-memory caching & error sanitization
├── Efficiency_Metrics.py       # Pure math formulas for Token count, Fertility, Compression, Premium
├── MBI.py                      # Morphological Boundary Identification & corpus evaluation
├── datasets.py                 # Language registry, FLORES-dev & NCHLT gold standard loaders
├── requirements.txt            # Python package dependencies
├── README.md                   # System documentation & developer guide
├── data/
│   ├── leaderboard.json        # Persistent benchmark store
│   ├── FLORES dev/             # Parallel evaluation text across 9 languages
│   └── NCHLT/                  # Linguist-annotated morphological segmentations
└── tests/
    ├── run_harness.py          # Unified test discovery runner
    ├── test_ranking_engine.py  # Normalization, inversion, weights & ranking unit tests
    ├── test_storage.py         # Schema, atomic write, corruption recovery & reset tests
    ├── test_render_results.py  # Display formatting, badges & delta table unit tests
    ├── test_render_stress.py   # Stress & edge case testing for presentation layer
    ├── test_Efficiency_Metrics.py # Mathematical verification of efficiency calculations
    ├── test_MBI.py             # Verification of morphological boundary detection
    └── test_e2e_workflow.py    # Multi-tier end-to-end integration and workflow test suite
```

---

## Installation & Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository and navigate to the project root
cd tokenizer-analysis-dashboard-main

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

Run the Streamlit application using `app.py`:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (typically `http://localhost:8501`).

---



