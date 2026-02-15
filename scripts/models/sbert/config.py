"""
SBERT Projection Model — Configuration

Shared settings for the SBERT cosine-projection pipeline.
Separated from the main NLP pipeline to avoid interference.
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# ── Model settings ────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # 384d, fast
EMBED_DIM = 384
BATCH_SIZE = 256

# ── Sample scope ──────────────────────────────────────────────
CONGRESS_RANGE = (99, 105)         # 99-105 inclusive (99 needed for window)
SCORE_CONGRESSES = [100, 101, 102, 103, 104, 105]  # congresses to score newspapers
WINDOW_SIZE = 2                    # (prev, current) congress pair

# ── Speech filtering ──────────────────────────────────────────
PARTISAN_CORE_ONLY = True          # Nokken-Poole top/bottom 20%
MIN_WORDS_AFTER_CLEAN = 20         # Level 2: min words after parl phrase removal
COMMON_WORD_RATIO_THRESHOLD = 0.05 # Level 3: ratio of top-100 words to total

# ── Newspaper sampling ────────────────────────────────────────
NEWSPAPER_SAMPLE_PER_CONGRESS = 10_000  # random sample per congress for speed

# ── Paths (inputs — shared with main pipeline) ───────────────
SPEECHES_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "01_speeches_merged.parquet"
LABELS_PATH   = BASE_DIR / "data" / "intermediate" / "speeches" / "04_speeches_with_partisan_core.parquet"
RAW_NEWSPAPERS = BASE_DIR / "data" / "intermediate" / "newspapers"

# Existing TF-IDF slant for comparison
TFIDF_SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers"

# ── Paths (outputs — isolated to models/sbert/) ──────────────
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "models" / "sbert"

# ── Top 100 common English words (for Level 3 filter) ────────
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


def get_windows():
    """Generate 2-congress windows: [(99,100), (100,101), ..., (104,105)]."""
    return [(c - 1, c) for c in SCORE_CONGRESSES]


def congress_to_years(congress):
    """Congress number → (year1, year2)."""
    year1 = 1787 + congress * 2
    return (year1, year1 + 1)
