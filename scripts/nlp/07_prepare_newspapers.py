"""
07_prepare_newspapers.py

Apply the pre-trained vectorizer (from step 05) to pre-cleaned
newspaper articles, grouped by congress window.

Widmer methodology (default):
  CountVectorizer → raw counts → L1 normalize to relative frequencies.
  This matches the speech feature space from step 05.

Legacy mode:
  TfidfVectorizer → TF-IDF values (no additional normalization).

Year-to-Congress mapping:
  Congress 100 -> 1987-1988
  Congress 101 -> 1989-1990
  ...
  Congress 108 -> 2003-2004

Inputs:
  - data/intermediate/newspapers/newspapers_{year}.parquet
  - data/processed/speeches/05_tfidf_vectorizer.joblib

Outputs (per congress):
  - 07_newspaper_tfidf_cong_{cong}.npz  (sparse: relative freq or TF-IDF)
  - 07_newspaper_meta_cong_{cong}.parquet
"""

import gc
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

# Ensure text_analyzer is importable (needed for unpickling the vectorizer)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import text_analyzer  # noqa: F401 — registers module for joblib.load

# ------------------------------------------------------------------
# Paths (from pipeline_config — change RUN_NAME there for new runs)
# ------------------------------------------------------------------
import pipeline_config as cfg

NEWSPAPER_DIR   = cfg.RAW_NEWSPAPERS
VECTORIZER_PATH = cfg.SPEECH_DIR / "05_tfidf_vectorizer.joblib"
OUT_DIR         = cfg.NEWS_DIR

USE_RELATIVE_FREQ = cfg.CONFIG.get("use_relative_freq", True)

# ------------------------------------------------------------------
# Year-to-Congress mapping (dynamic from config)
# ------------------------------------------------------------------
CONGRESS_YEARS = {}
for cong in cfg.get_congresses():
    year1 = 1787 + (cong * 2)
    CONGRESS_YEARS[cong] = (year1, year1 + 1)

N_JOBS = 8  # parallel workers per congress; lower = less memory


def _transform_chunk(vectorizer, texts):
    """Transform a text chunk in a worker process."""
    return vectorizer.transform(texts)


def normalize_to_relative_freq(count_matrix):
    """L1 normalize rows to relative frequencies, handling zero rows."""
    # sklearn's normalize handles zero rows gracefully (returns zero rows)
    return normalize(count_matrix.astype(np.float64), norm='l1', axis=1)


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    # 1. Load vectorizer
    print("Loading vectorizer ...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    vocab = getattr(vectorizer, 'vocabulary_', None) or vectorizer.vocabulary
    print(f"  Vocabulary size: {len(vocab):,}")
    print(f"  Mode: {'relative frequency' if USE_RELATIVE_FREQ else 'TF-IDF'}")

    n_cores = max(1, joblib.cpu_count() or 1) if N_JOBS == -1 else max(1, N_JOBS)
    print(f"  Using {n_cores} CPU cores for parallel transform")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. First pass: count articles per congress (for progress bar)
    congress_counts = {}
    congresses = sorted(CONGRESS_YEARS.keys())

    for cong in congresses:
        year1, year2 = CONGRESS_YEARS[cong]
        total = 0
        for year in (year1, year2):
            p = NEWSPAPER_DIR / f"newspapers_{year}.parquet"
            if p.exists():
                total += pd.read_parquet(p, columns=["paper"]).shape[0]
        congress_counts[cong] = total

    grand_total = sum(congress_counts.values())
    print(f"\n  Total articles across all congresses: {grand_total:,}")

    # 3. Process one congress at a time (load -> transform -> normalize -> save -> free)
    print("\nTransforming newspaper text ...")
    pipeline_start = time.time()
    total = len(congresses)
    pbar = tqdm(total=grand_total, desc="  Transform", unit="doc")

    summary = []

    for i, cong in enumerate(congresses, 1):
        window_start = time.time()
        year1, year2 = CONGRESS_YEARS[cong]

        # Load only this congress's years
        dfs = []
        for year in (year1, year2):
            p = NEWSPAPER_DIR / f"newspapers_{year}.parquet"
            if not p.exists():
                print(f"  WARNING: {p.name} not found, skipping")
                continue
            df = pd.read_parquet(p)
            df["year"] = year
            df["congress"] = cong
            dfs.append(df)

        if not dfs:
            continue

        articles = pd.concat(dfs, ignore_index=True)
        del dfs
        articles = articles.dropna(subset=["text"])

        # Dev subsample (for quick iteration)
        news_sample = cfg.CONFIG.get("newspaper_sample_frac")
        if news_sample is not None and 0 < news_sample < 1:
            sample_idx = articles.sample(frac=news_sample, random_state=42).index
            np.save(OUT_DIR / f"07_sample_idx_cong_{cong}.npy", sample_idx.values)
            articles = articles.loc[sample_idx].reset_index(drop=True)

        # Combine title + text
        texts = (articles["title"].fillna("") + " " + articles["text"]).values
        n_docs = len(texts)

        # Save metadata before freeing the full dataframe
        meta = articles[["date", "paper", "title", "word_count", "year", "congress"]].reset_index(drop=True)
        n_papers = articles["paper"].nunique()
        del articles
        gc.collect()

        # Transform to counts
        if n_cores == 1 or n_docs < 1000:
            count_matrix = vectorizer.transform(texts)
        else:
            chunks = np.array_split(texts, min(n_cores, n_docs))
            chunks = [c for c in chunks if len(c) > 0]
            del texts
            gc.collect()

            results = Parallel(n_jobs=N_JOBS)(
                delayed(_transform_chunk)(vectorizer, chunk)
                for chunk in chunks
            )
            del chunks
            count_matrix = sp.vstack(results, format="csr")
            del results
            gc.collect()

        pbar.update(n_docs)

        # Normalize to relative frequencies if using Widmer mode
        if USE_RELATIVE_FREQ:
            feature_matrix = normalize_to_relative_freq(count_matrix)
            del count_matrix
        else:
            feature_matrix = count_matrix

        # Save
        sp.save_npz(OUT_DIR / f"07_newspaper_tfidf_cong_{cong}.npz", feature_matrix)
        meta.to_parquet(OUT_DIR / f"07_newspaper_meta_cong_{cong}.parquet")

        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / i
        remaining = avg_per * (total - i)
        pbar.write(f"  [{i}/{total}] Congress {cong}: {n_docs:,} articles from "
                   f"{n_papers} papers, shape {feature_matrix.shape}  |  {elapsed:.1f}s  |  ETA: {remaining:.0f}s")

        summary.append((cong, year1, year1 + 1, n_docs, n_papers))

        del feature_matrix, meta
        gc.collect()

    pbar.close()

    # 4. Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Newspaper articles per congress")
    print("=" * 72)

    for cong, y1, y2, n_articles, n_papers in summary:
        print(f"  Congress {cong} ({y1}-{y2}): "
              f"{n_articles:,} articles from {n_papers} papers")

    print("=" * 72)
    print(f"\nSaved to -> {OUT_DIR}")
