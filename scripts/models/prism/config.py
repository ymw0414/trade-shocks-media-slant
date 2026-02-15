"""
PRISM Cross-Encoder Model — Configuration

Topic-decomposed newspaper slant using cross-encoder scoring
against R/D indicator speeches, with GST procedural filtering
and partisan phrase-based topic labeling.
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# ── Cross-encoder model ──────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-base"
CROSS_ENCODER_BATCH_SIZE = 64

# ── Topic mining ─────────────────────────────────────────────
N_CLUSTERS = 200              # k-means clusters per window
N_POLARIZED_TOPICS = 30       # top polarized clusters to keep
TOP_M_TOPICS = 5              # relevant topics per article for scoring
INDICATOR_MAX_WORDS = 100     # truncate indicator speeches
MIN_CLUSTER_SIZE = 5          # minimum speeches per cluster

# ── Sample scope (same as SBERT) ─────────────────────────────
CONGRESS_RANGE = (99, 105)
SCORE_CONGRESSES = [100, 101, 102, 103, 104, 105]
WINDOW_SIZE = 2

# ── Speech filtering ─────────────────────────────────────────
PARTISAN_CORE_ONLY = True
MIN_WORDS_AFTER_CLEAN = 20
COMMON_WORD_RATIO_THRESHOLD = 0.05
PROCEDURAL_RATIO_THRESHOLD = 0.3  # max fraction of bigrams that are procedural

# ── Newspaper sampling ────────────────────────────────────────
NEWSPAPER_SAMPLE_PER_CONGRESS = 10_000

# ── Paths (inputs — reuse SBERT embeddings) ───────────────────
SBERT_DIR = BASE_DIR / "data" / "processed" / "models" / "sbert"
SPEECHES_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "01_speeches_merged.parquet"
LABELS_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "04_speeches_with_partisan_core.parquet"
RAW_NEWSPAPERS = BASE_DIR / "data" / "intermediate" / "newspapers"

# Existing slant for comparison
TFIDF_SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers"
BERT_DIR = BASE_DIR / "data" / "processed" / "models" / "finetuned_bert"

# ── GST official data ─────────────────────────────────────────
GST_VOCAB_DIR = BASE_DIR / "data" / "raw" / "speeches" / "vocabulary"
GST_PROCEDURAL_PATH = GST_VOCAB_DIR / "procedural.txt"
GST_MASTER_LIST_PATH = GST_VOCAB_DIR / "master_list.txt"
GST_PARTISAN_DIR = BASE_DIR / "data" / "raw" / "speeches" / "phrase_partisanship"

# ── Paths (outputs) ───────────────────────────────────────────
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "models" / "prism"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

# ── Regression panel ──────────────────────────────────────────
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel" / "14_regression_panel.parquet"

# ── Common words (Level 3 filter, same as SBERT) ─────────────
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been", "has",
    "had", "did", "said",
}

# ── State FIPS -> Census Division (for regressions) ──────────
STATE_TO_DIVISION = {
    9: 1, 23: 1, 25: 1, 33: 1, 44: 1, 50: 1,
    34: 2, 36: 2, 42: 2,
    17: 3, 18: 3, 26: 3, 39: 3, 55: 3,
    19: 4, 20: 4, 27: 4, 29: 4, 31: 4, 38: 4, 46: 4,
    10: 5, 11: 5, 12: 5, 13: 5, 24: 5, 37: 5, 45: 5, 51: 5, 54: 5,
    1: 6, 21: 6, 28: 6, 47: 6,
    5: 7, 22: 7, 40: 7, 48: 7,
    4: 8, 8: 8, 16: 8, 30: 8, 32: 8, 35: 8, 49: 8, 56: 8,
    2: 9, 6: 9, 15: 9, 41: 9, 53: 9,
}


def get_windows():
    """Generate 2-congress windows: [(99,100), (100,101), ..., (104,105)]."""
    return [(c - 1, c) for c in SCORE_CONGRESSES]


def congress_to_years(congress):
    """Congress number → (year1, year2)."""
    year1 = 1787 + congress * 2
    return (year1, year1 + 1)
