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
CONGRESS_YEARS = {}
for cong in range(100, 109):
    year1 = 1787 + (cong * 2)
    CONGRESS_YEARS[cong] = (year1, year1 + 1)

N_JOBS = 8  # parallel workers per congress; lower = less memory


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

    # 3. Process one congress at a time (load -> transform -> save -> free)
    print("\nTransforming newspaper text ...")
    pipeline_start = time.time()
    total = len(congresses)
    pbar = tqdm(total=grand_total, desc="  TF-IDF transform", unit="doc")

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

        # Combine title + text
        texts = (articles["title"].fillna("") + " " + articles["text"]).values
        n_docs = len(texts)

        # Save metadata before freeing the full dataframe
        meta = articles[["date", "paper", "title", "word_count", "year", "congress"]].reset_index(drop=True)
        n_papers = articles["paper"].nunique()
        del articles
        gc.collect()

        # Transform
        if n_cores == 1 or n_docs < 1000:
            tfidf = vectorizer.transform(texts)
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
            tfidf = sp.vstack(results, format="csr")
            del results
            gc.collect()

        pbar.update(n_docs)

        # Save
        sp.save_npz(OUT_DIR / f"07_newspaper_tfidf_cong_{cong}.npz", tfidf)
        meta.to_parquet(OUT_DIR / f"07_newspaper_meta_cong_{cong}.parquet")

        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / i
        remaining = avg_per * (total - i)
        pbar.write(f"  [{i}/{total}] Congress {cong}: {n_docs:,} articles from "
                   f"{n_papers} papers, TF-IDF {tfidf.shape}  |  {elapsed:.1f}s  |  ETA: {remaining:.0f}s")

        summary.append((cong, year1, year1 + 1, n_docs, n_papers))

        del tfidf, meta
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
