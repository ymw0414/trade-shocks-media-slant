"""
01_prepare_data.py -- Load speeches, filter, and create train/val splits per window.

Reuses the 3-level filtering from SBERT pipeline.
Outputs train/val parquet files per window for BERT fine-tuning.
"""

import re
import sys
import time
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.text_analyzer import PARLIAMENTARY_PHRASES
from models.finetuned_bert.config import (
    CONGRESS_RANGE, PARTISAN_CORE_ONLY,
    MIN_WORDS_AFTER_CLEAN, COMMON_WORD_RATIO_THRESHOLD, COMMON_WORDS,
    SPEECHES_PATH, LABELS_PATH, DATA_DIR, VAL_SPLIT, RANDOM_SEED,
    get_windows,
)


def build_parl_regex():
    return re.compile(
        "|".join(re.escape(p) for p in PARLIAMENTARY_PHRASES),
        re.IGNORECASE,
    )


def clean_text(text, parl_re):
    return parl_re.sub(" ", text).strip()


def common_word_ratio(text):
    tokens = text.lower().split()
    if len(tokens) == 0:
        return 0.0
    return sum(1 for t in tokens if t in COMMON_WORDS) / len(tokens)


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    print("Loading speeches...")
    speeches = pd.read_parquet(SPEECHES_PATH, columns=["speech_id", "speech", "congress"])
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    speeches["congress"] = speeches["congress"].astype(int)

    print("Loading labels...")
    labels = pd.read_parquet(LABELS_PATH)
    labels["speech_id"] = labels["speech_id"].astype(str)

    df = speeches.merge(
        labels[["speech_id", "congress_int", "party", "label_rep_core", "label_dem_core"]],
        on="speech_id", how="inner",
    )
    del speeches, labels

    # -- Filter scope --
    lo, hi = CONGRESS_RANGE
    df = df[df["congress_int"].between(lo, hi)].copy()
    df = df[df["party"].isin(["R", "D"])].copy()
    print(f"R/D speeches in congress {lo}-{hi}: {len(df):,}")

    if PARTISAN_CORE_ONLY:
        df = df[(df["label_rep_core"] == 1) | (df["label_dem_core"] == 1)].copy()
        print(f"Partisan core only: {len(df):,}")

    # -- Level 1: Parliamentary phrase removal --
    print("Level 1: Removing parliamentary phrases...")
    parl_re = build_parl_regex()
    df["text"] = df["speech"].apply(lambda x: clean_text(str(x), parl_re))
    del df["speech"]

    # -- Level 2: Minimum word count --
    df["wc"] = df["text"].apply(lambda x: len(x.split()))
    n_before = len(df)
    df = df[df["wc"] >= MIN_WORDS_AFTER_CLEAN].copy()
    print(f"Level 2: {n_before:,} -> {len(df):,} (dropped {n_before - len(df):,})")

    # -- Level 3: Common-word ratio --
    df["cw_ratio"] = df["text"].apply(common_word_ratio)
    n_before = len(df)
    df = df[df["cw_ratio"] >= COMMON_WORD_RATIO_THRESHOLD].copy()
    print(f"Level 3: {n_before:,} -> {len(df):,} (dropped {n_before - len(df):,})")

    # -- Create label column --
    df["label"] = (df["label_rep_core"] == 1).astype(int)  # 1=R, 0=D

    # -- Save per window --
    print(f"\nCreating train/val splits per window...")

    for prev_cong, curr_cong in get_windows():
        mask = df["congress_int"].isin([prev_cong, curr_cong])
        window_df = df[mask][["speech_id", "congress_int", "party", "text", "wc", "label"]].copy()

        n_r = (window_df["label"] == 1).sum()
        n_d = (window_df["label"] == 0).sum()

        train_df, val_df = train_test_split(
            window_df, test_size=VAL_SPLIT,
            stratify=window_df["label"], random_state=RANDOM_SEED,
        )

        train_path = DATA_DIR / f"train_{prev_cong}_{curr_cong}.parquet"
        val_path = DATA_DIR / f"val_{prev_cong}_{curr_cong}.parquet"
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

        print(f"  Window ({prev_cong},{curr_cong}): {len(window_df):,} total "
              f"(R={n_r:,}, D={n_d:,}) -> train={len(train_df):,}, val={len(val_df):,}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
