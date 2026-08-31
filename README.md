# Tokenizer Analysis Dashboard (TAD)

CSC3003S Capstone Project — interactive dashboard for analysing Hugging Face tokenizers on South African languages.

## Features

| Mode | Status |
|------|--------|
| **Home** | Project overview, supported languages, metrics explained |
| **Inspect** | Load one tokenizer → FLORES efficiency + NCHLT morphology, or free text / upload |
| **Compare** | Side-by-side metrics and sample tokenizations for two tokenizers |
| **Rank** | Placeholder (not implemented yet) |

### Supported languages

Afrikaans · English · isiXhosa · isiZulu · Sesotho · Sepedi · Setswana · siSwati · Xitsonga

### Metrics

**Efficiency (FLORES-dev)**  
Token count · Fertility (tokens/word) · Compression (bytes/token) · Token premium (vs English)

**Linguistic (NCHLT)**  
Morphological Boundary Identification — Precision, Recall, F1

## Project structure

```
tokenizer-analysis-dashboard/
├── Capstone5.py              # Main Streamlit app (run this)
├── tokenizer_services.py     # HF tokenizer load + tokenize + boundary helpers
├── Efficiency_Metrics.py     # Token count, fertility, compression, premium
├── MBI.py                    # Morphological boundary P/R/F1
├── datasets.py               # FLORES + NCHLT loaders
├── requirements.txt
├── data/
│   ├── FLORES dev/           # 997 parallel sentences × 9 languages
│   └── NCHLT/                # Gold morpheme segmentations
└── frontend/                 # Older UI drafts (not used)
```

## Installation

```bash
cd tokenizer-analysis-dashboard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run Capstone5.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## How the pieces connect

```
Capstone5.py (UI)
    │
    ├─► tokenizer_services.load_tokenizer / tokenize_text
    ├─► datasets.load_flores / load_nchlt_gold
    ├─► Efficiency_Metrics.compute_all
    └─► MBI.evaluate_corpus
```

1. User selects a Hugging Face model ID and clicks **Load tokenizer**.
2. On **Run analysis**, the app loads FLORES (and NCHLT if available).
3. Text is tokenized; efficiency metrics are computed against English parallel text.
4. For languages with NCHLT gold, each word's gold `_` boundaries are compared to tokenizer cut positions → P/R/F1.

## Notes

- FLORES folder name contains a space (`FLORES dev`) — paths in `datasets.py` handle this.
- Afrikaans and English have no NCHLT gold; morphology metrics are skipped for them.
- Rank / leaderboard is intentionally left as a placeholder for a later milestone.
