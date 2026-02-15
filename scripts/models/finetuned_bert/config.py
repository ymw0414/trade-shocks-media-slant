"""
Fine-Tuned BERT (Method 3) -- Configuration

Shared settings for the BERT fine-tuning pipeline.
Reuses paths and window structure from SBERT config.
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# -- Model settings --
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 256
NUM_LABELS = 2  # R vs D

# -- Training hyperparameters --
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# -- Sample scope (same as SBERT) --
CONGRESS_RANGE = (99, 105)
SCORE_CONGRESSES = [100, 101, 102, 103, 104, 105]
WINDOW_SIZE = 2

# -- Speech filtering (same as SBERT) --
PARTISAN_CORE_ONLY = True
MIN_WORDS_AFTER_CLEAN = 20
COMMON_WORD_RATIO_THRESHOLD = 0.05

# -- Newspaper sampling --
NEWSPAPER_SAMPLE_PER_CONGRESS = 10_000

# -- Paths (inputs) --
SPEECHES_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "01_speeches_merged.parquet"
LABELS_PATH = BASE_DIR / "data" / "intermediate" / "speeches" / "04_speeches_with_partisan_core.parquet"
RAW_NEWSPAPERS = BASE_DIR / "data" / "intermediate" / "newspapers"

# Existing slant data for comparison
TFIDF_SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers"
SBERT_DIR = BASE_DIR / "data" / "processed" / "models" / "sbert"

# -- Paths (outputs) --
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "models" / "finetuned_bert"
DATA_DIR = OUTPUT_DIR / "data"         # prepared train/val splits
MODEL_DIR = OUTPUT_DIR / "models"      # saved fine-tuned models

# -- Common words for Level 3 filter --
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
    """Congress number -> (year1, year2)."""
    year1 = 1787 + congress * 2
    return (year1, year1 + 1)
