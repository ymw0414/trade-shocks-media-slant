"""
01_embed_speeches.py — Load, filter, and embed congressional speeches with SBERT.

Filtering pipeline:
  Level 1: Remove parliamentary procedure phrases (regex)
  Level 2: Drop speeches with < 20 words after cleaning
  Level 3: Drop speeches with common-word ratio < 0.05

Output per congress:
  speech_embeddings_cong_{N}.npz  (n_speeches × 384)
  speech_meta_cong_{N}.parquet    (speech_id, party, congress, is_core_R, is_core_D, word_count)
"""

import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.text_analyzer import PARLIAMENTARY_PHRASES
from models.sbert.config import (
    MODEL_NAME, BATCH_SIZE, CONGRESS_RANGE,
    PARTISAN_CORE_ONLY, MIN_WORDS_AFTER_CLEAN,
    COMMON_WORD_RATIO_THRESHOLD, COMMON_WORDS,
    SPEECHES_PATH, LABELS_PATH, OUTPUT_DIR, EMBED_DIM,
)


def build_parl_regex():
    """Compile parliamentary phrase removal regex."""
    return re.compile(
        "|".join(re.escape(p) for p in PARLIAMENTARY_PHRASES),
        re.IGNORECASE,
    )


def clean_text(text, parl_re):
    """Level 1: Remove parliamentary phrases, return cleaned text."""
    return parl_re.sub(" ", text).strip()


def word_count(text):
    """Count words (simple whitespace split)."""
    return len(text.split())


def common_word_ratio(text):
    """Level 3: Ratio of top-100 common English words to total tokens."""
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    common_count = sum(1 for t in tokens if t in COMMON_WORDS)
    return common_count / len(tokens)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    print("Loading speeches...")
    speeches = pd.read_parquet(SPEECHES_PATH, columns=["speech_id", "speech", "congress"])
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    speeches["congress"] = speeches["congress"].astype(int)

    print("Loading labels...")
    labels = pd.read_parquet(LABELS_PATH)
    labels["speech_id"] = labels["speech_id"].astype(str)

    # Merge text + metadata
    df = speeches.merge(
        labels[["speech_id", "congress_int", "party", "label_rep_core", "label_dem_core"]],
        left_on="speech_id",
        right_on="speech_id",
        how="inner",
    )
    del speeches, labels

    # ── Filter scope ──────────────────────────────────────────
    lo, hi = CONGRESS_RANGE
    df = df[df["congress_int"].between(lo, hi)].copy()
    print(f"Speeches in congress {lo}-{hi}: {len(df):,}")

    # Party filter (R or D)
    df = df[df["party"].isin(["R", "D"])].copy()

    # Partisan core only
    if PARTISAN_CORE_ONLY:
        df = df[(df["label_rep_core"] == 1) | (df["label_dem_core"] == 1)].copy()
        print(f"Partisan core only: {len(df):,}")

    # ── Level 1: Parliamentary phrase removal ─────────────────
    print("Level 1: Removing parliamentary phrases...")
    parl_re = build_parl_regex()
    df["speech_clean"] = df["speech"].apply(lambda x: clean_text(str(x), parl_re))
    del df["speech"]

    # ── Level 2: Minimum word count ───────────────────────────
    df["wc_clean"] = df["speech_clean"].apply(word_count)
    n_before = len(df)
    df = df[df["wc_clean"] >= MIN_WORDS_AFTER_CLEAN].copy()
    print(f"Level 2: {n_before:,} -> {len(df):,} (dropped {n_before - len(df):,} short speeches)")

    # ── Level 3: Common-word ratio ────────────────────────────
    df["cw_ratio"] = df["speech_clean"].apply(common_word_ratio)
    n_before = len(df)
    df = df[df["cw_ratio"] >= COMMON_WORD_RATIO_THRESHOLD].copy()
    print(f"Level 3: {n_before:,} -> {len(df):,} (dropped {n_before - len(df):,} low-ratio speeches)")

    # ── SBERT encoding ────────────────────────────────────────
    print(f"Loading SBERT model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    for cong in sorted(df["congress_int"].unique()):
        mask = df["congress_int"] == cong
        subset = df[mask]
        n = len(subset)
        print(f"\nCongress {cong}: {n:,} speeches")
        if n == 0:
            continue

        # Party breakdown
        n_r = (subset["label_rep_core"] == 1).sum() if PARTISAN_CORE_ONLY else (subset["party"] == "R").sum()
        n_d = (subset["label_dem_core"] == 1).sum() if PARTISAN_CORE_ONLY else (subset["party"] == "D").sum()
        print(f"  R core: {n_r:,}  |  D core: {n_d:,}")

        texts = subset["speech_clean"].tolist()
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Save
        np.savez_compressed(
            OUTPUT_DIR / f"speech_embeddings_cong_{cong}.npz",
            embeddings=embeddings,
        )
        meta = subset[["speech_id", "congress_int", "party",
                        "label_rep_core", "label_dem_core", "wc_clean"]].copy()
        meta = meta.rename(columns={"wc_clean": "word_count_clean"})
        meta.to_parquet(OUTPUT_DIR / f"speech_meta_cong_{cong}.parquet", index=False)

        print(f"  Saved: embeddings {embeddings.shape}, meta {len(meta):,}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
