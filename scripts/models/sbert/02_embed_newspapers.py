"""
02_embed_newspapers.py — Load, sample, and embed newspaper articles with SBERT.

Per congress:
  1. Load newspapers for both years in that congress
  2. Random sample (default 10,000 per congress)
  3. Combine title + text
  4. Encode with same SBERT model as speeches

Output per congress:
  newspaper_embeddings_cong_{N}.npz  (n_articles × 384)
  newspaper_meta_cong_{N}.parquet    (date, paper, title, word_count, year, congress)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sbert.config import (
    MODEL_NAME, BATCH_SIZE, SCORE_CONGRESSES,
    NEWSPAPER_SAMPLE_PER_CONGRESS,
    RAW_NEWSPAPERS, OUTPUT_DIR,
    congress_to_years,
)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading SBERT model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    for cong in SCORE_CONGRESSES:
        year1, year2 = congress_to_years(cong)
        print(f"\n{'='*60}")
        print(f"Congress {cong}: years {year1}-{year2}")

        # Load both years
        dfs = []
        for yr in [year1, year2]:
            path = RAW_NEWSPAPERS / f"newspapers_{yr}.parquet"
            if not path.exists():
                print(f"  WARNING: {path.name} not found, skipping year {yr}")
                continue
            df_yr = pd.read_parquet(path)
            df_yr["year"] = yr
            dfs.append(df_yr)

        if not dfs:
            print(f"  No newspaper data for congress {cong}, skipping")
            continue

        articles = pd.concat(dfs, ignore_index=True)
        articles["congress"] = cong
        print(f"  Total articles: {len(articles):,}")

        # Random sample
        n_sample = min(NEWSPAPER_SAMPLE_PER_CONGRESS, len(articles))
        articles = articles.sample(n=n_sample, random_state=42).reset_index(drop=True)
        print(f"  Sampled: {n_sample:,}")

        # Combine title + text
        texts = (
            articles["title"].fillna("") + " " + articles["text"].fillna("")
        ).str.strip().tolist()

        # Encode
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Save
        np.savez_compressed(
            OUTPUT_DIR / f"newspaper_embeddings_cong_{cong}.npz",
            embeddings=embeddings,
        )
        meta = articles[["date", "paper", "title", "word_count", "year", "congress"]].copy()
        meta.to_parquet(OUTPUT_DIR / f"newspaper_meta_cong_{cong}.parquet", index=False)

        print(f"  Saved: embeddings {embeddings.shape}, meta {len(meta):,}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
