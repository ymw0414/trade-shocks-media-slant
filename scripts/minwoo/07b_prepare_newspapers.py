"""
07b_prepare_newspapers.py

Load the standardized newspaper articles from 07a, map each article
to its corresponding congress window, and transform text using the
uniform TF-IDF vectorizer from step 05.

Expects 07a to have already run (name standardization + filtering).

Year-to-Congress mapping:
  Congress 100 -> 1987-1988  (model: 99+100)
  Congress 101 -> 1989-1990  (model: 100+101)
  ...
  Congress 108 -> 2003-2004  (model: 107+108)

Outputs:
  - One TF-IDF matrix per congress (sparse .npz)
  - One metadata file per congress (paper, date, title)
"""

import os
import time
import joblib
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

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
    year1 = 1787 + (cong * 2)  # first year of congress
    year2 = year1 + 1
    YEAR_TO_CONGRESS[year1] = cong
    YEAR_TO_CONGRESS[year2] = cong

# ------------------------------------------------------------------
# 1. Load vectorizer
# ------------------------------------------------------------------
print("Loading TF-IDF vectorizer ...")
vectorizer = joblib.load(VECTORIZER_PATH)
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

# ------------------------------------------------------------------
# 2. Load standardized newspapers, grouped by congress
# ------------------------------------------------------------------
congress_articles = {}  # congress -> list of DataFrames

for year in range(1987, 2005):
    cong = YEAR_TO_CONGRESS[year]
    parquet_path = NEWSPAPER_DIR / f"newspapers_{year}.parquet"

    if not parquet_path.exists():
        print(f"  WARNING: {parquet_path.name} not found, skipping")
        continue

    df = pd.read_parquet(parquet_path)
    df["year"] = year
    df["congress"] = cong

    if cong not in congress_articles:
        congress_articles[cong] = []
    congress_articles[cong].append(df)

    print(f"  Loaded {year}: {len(df):,} articles -> Congress {cong}")

# ------------------------------------------------------------------
# 3. Transform and save per congress
# ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nTransforming newspaper text ...")
total = len(congress_articles)
pipeline_start = time.time()

for i, (cong, dfs) in enumerate(sorted(congress_articles.items()), 1):
    window_start = time.time()
    articles = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing text
    articles = articles.dropna(subset=["text"])

    # Transform using the uniform vectorizer
    tfidf = vectorizer.transform(articles["text"])

    # Save sparse matrix
    sp.save_npz(OUT_DIR / f"07_newspaper_tfidf_cong_{cong}.npz", tfidf)

    # Save metadata (same row order as matrix)
    meta = articles[["date", "paper", "title", "word_count", "year", "congress"]].reset_index(drop=True)
    meta.to_parquet(OUT_DIR / f"07_newspaper_meta_cong_{cong}.parquet")

    elapsed = time.time() - window_start
    total_elapsed = time.time() - pipeline_start
    avg_per = total_elapsed / i
    remaining = avg_per * (total - i)
    n_papers = articles["paper"].nunique()
    print(f"  [{i}/{total}] Congress {cong}: {len(articles):,} articles from "
          f"{n_papers} papers, TF-IDF {tfidf.shape}  |  {elapsed:.1f}s  |  ETA: {remaining:.0f}s")

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
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
