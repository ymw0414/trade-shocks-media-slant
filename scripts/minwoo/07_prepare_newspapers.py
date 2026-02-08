"""
07_prepare_newspapers.py

Apply the pre-trained TF-IDF vectorizer (from step 05) to pre-cleaned
newspaper articles, grouped by congress window.

Assumes input data is already clean: name-standardized and filtered to
the target sample by newspapers/03_apply_crosswalk.py.

Year-to-Congress mapping:
  Congress 100 -> 1987-1988
  Congress 101 -> 1989-1990
  ...
  Congress 108 -> 2003-2004

Inputs:
  - data/intermediate/newspapers/minwoo/newspapers_{year}.parquet
  - data/processed/speeches/minwoo/05_tfidf_vectorizer.joblib

Outputs (per congress):
  - data/processed/newspapers/minwoo/07_newspaper_tfidf_cong_{cong}.npz
  - data/processed/newspapers/minwoo/07_newspaper_meta_cong_{cong}.parquet
"""

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

# Ensure text_analyzer is importable (needed for unpickling the vectorizer)
sys.path.insert(0, str(Path(__file__).parent))
import text_analyzer  # noqa: F401 — registers module for joblib.load

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

NEWSPAPER_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "minwoo"
VECTORIZER_PATH = BASE_DIR / "data" / "processed" / "speeches" / "minwoo" / "05_tfidf_vectorizer.joblib"

OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "minwoo"

# ------------------------------------------------------------------
# Year-to-Congress mapping
# ------------------------------------------------------------------
YEAR_TO_CONGRESS = {}
for cong in range(100, 109):
    year1 = 1787 + (cong * 2)
    year2 = year1 + 1
    YEAR_TO_CONGRESS[year1] = cong
    YEAR_TO_CONGRESS[year2] = cong

N_JOBS = -1  # use all CPU cores; set to 1 to disable parallelism


def _transform_chunk(vectorizer, texts):
    """Transform a text chunk in a worker process."""
    return vectorizer.transform(texts)


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    # 1. Load vectorizer
    print("Loading TF-IDF vectorizer ...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    n_cores = max(1, joblib.cpu_count() or 1) if N_JOBS == -1 else max(1, N_JOBS)
    print(f"  Using {n_cores} CPU cores for parallel transform")

    # 2. Load articles, grouped by congress
    congress_articles = {}

    for year in range(1987, 2005):
        cong = YEAR_TO_CONGRESS[year]
        parquet_path = NEWSPAPER_DIR / f"newspapers_{year}.parquet"

        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path.name} not found, skipping")
            continue

        df = pd.read_parquet(parquet_path)
        df["year"] = year
        df["congress"] = cong

        congress_articles.setdefault(cong, []).append(df)
        print(f"  Loaded {year}: {len(df):,} articles -> Congress {cong}")

    # 3. Transform and save per congress
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTransforming newspaper text ...")
    total = len(congress_articles)
    pipeline_start = time.time()

    # Count total articles for the overall progress bar
    total_articles = sum(
        sum(len(df) for df in dfs)
        for dfs in congress_articles.values()
    )
    pbar = tqdm(total=total_articles, desc="  TF-IDF transform", unit="doc")

    for i, (cong, dfs) in enumerate(sorted(congress_articles.items()), 1):
        window_start = time.time()
        articles = pd.concat(dfs, ignore_index=True)
        articles = articles.dropna(subset=["text"])

        texts = articles["text"].values
        n_docs = len(texts)

        if n_cores == 1 or n_docs < 1000:
            # Single-process path
            tfidf = vectorizer.transform(texts)
        else:
            # Split into chunks, transform in parallel, vstack
            chunks = np.array_split(texts, min(n_cores, n_docs))
            chunks = [c for c in chunks if len(c) > 0]

            results = Parallel(n_jobs=N_JOBS)(
                delayed(_transform_chunk)(vectorizer, chunk)
                for chunk in chunks
            )
            tfidf = sp.vstack(results, format="csr")

        pbar.update(n_docs)

        sp.save_npz(OUT_DIR / f"07_newspaper_tfidf_cong_{cong}.npz", tfidf)

        meta = articles[["date", "paper", "title", "word_count", "year", "congress"]].reset_index(drop=True)
        meta.to_parquet(OUT_DIR / f"07_newspaper_meta_cong_{cong}.parquet")

        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / i
        remaining = avg_per * (total - i)
        n_papers = articles["paper"].nunique()
        pbar.write(f"  [{i}/{total}] Congress {cong}: {len(articles):,} articles from "
                   f"{n_papers} papers, TF-IDF {tfidf.shape}  |  {elapsed:.1f}s  |  ETA: {remaining:.0f}s")

    pbar.close()

    # 4. Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Newspaper articles per congress")
    print("=" * 72)

    for cong in sorted(congress_articles.keys()):
        dfs = congress_articles[cong]
        total_articles = sum(len(df) for df in dfs)
        years = sorted(set(y for df in dfs for y in df["year"].unique()))
        n_papers = len(set(p for df in dfs for p in df["paper"].unique()))
        print(f"  Congress {cong} ({years[0]}-{years[-1]}): "
              f"{total_articles:,} articles from {n_papers} papers")

    print("=" * 72)
    print(f"\nSaved to -> {OUT_DIR}")
