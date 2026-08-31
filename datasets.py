"""Load FLORES-dev and NCHLT gold segmentations."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
FLORES_DIR = BASE_DIR / "data" / "FLORES dev"
NCHLT_DIR = BASE_DIR / "data" / "NCHLT"

# Display name -> (code, FLORES filename, NCHLT filename or None)
LANGUAGES = {
    "Afrikaans": ("afr", "afr_Latn.dev", None),
    "English": ("eng", "eng_Latn.dev", None),
    "isiXhosa": ("xho", "xho_Latn.dev", "xho.txt"),
    "isiZulu": ("zul", "zul_Latn.dev", "zul.txt"),
    "Sesotho": ("sot", "sot_Latn.dev", "sot.txt"),
    "Sepedi": ("nso", "nso_Latn.dev", "nso.txt"),
    "Setswana": ("tsn", "tsn_Latn.dev", "tsn.txt"),
    "siSwati": ("ssw", "ssw_Latn.dev", "ssw.txt"),
    "Xitsonga": ("tso", "tso_Latn.dev", "tso.txt"),
}

LANGUAGE_NAMES = list(LANGUAGES.keys())


def load_flores(language_name: str, max_sentences: Optional[int] = None) -> List[str]:
    """Load FLORES-dev sentences for a language (by display name)."""
    if language_name not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language_name}")
    _, filename, _ = LANGUAGES[language_name]
    path = FLORES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"FLORES file not found: {path}")

    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
                if max_sentences is not None and len(sentences) >= max_sentences:
                    break
    return sentences


def load_nchlt_gold(language_name: str) -> Dict[str, str]:
    """
    Load NCHLT gold segmentations.

    Returns dict: surface_word -> gold_segmentation_string (with '_' between morphs).
    Example: "bayahamba" -> "ba_ya_hamba"
    """
    if language_name not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language_name}")
    _, _, nchlt_file = LANGUAGES[language_name]
    if nchlt_file is None:
        raise ValueError(f"No NCHLT gold data for {language_name}")

    path = NCHLT_DIR / nchlt_file
    if not path.exists():
        raise FileNotFoundError(f"NCHLT file not found: {path}")

    gold: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            word, seg = line.split("\t", 1)
            if word and seg:
                gold[word] = seg
    return gold


def has_morphology(language_name: str) -> bool:
    """True if NCHLT gold exists for this language."""
    return language_name in LANGUAGES and LANGUAGES[language_name][2] is not None
