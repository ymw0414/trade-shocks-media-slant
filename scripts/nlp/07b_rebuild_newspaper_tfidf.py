"""
07b_rebuild_newspaper_tfidf.py

Targeted rebuild of newspaper TF-IDF files that have inconsistent
dimensions or are corrupted. Only processes specified congresses.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import text_analyzer  # noqa: F401

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
NEWSPAPER_DIR = BASE_DIR / "data" / "intermediate" / "newspapers"
VECTORIZER_PATH = BASE_DIR / "data" / "processed" / "speeches" / "05_tfidf_vectorizer.joblib"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers"

# Rebuild congresses with wrong dimensions (need to match current vectorizer)
CONGRESSES_TO_REBUILD = [108]
N_JOBS = 8

CONGRESS_YEARS = {}
for cong in range(100, 109):
    year1 = 1787 + (cong * 2)
    CONGRESS_YEARS[cong] = (year1, year1 + 1)


def _transform_chunk(vectorizer, texts):
    return vectorizer.transform(texts)


if __name__ == "__main__":
    print("Loading TF-IDF vectorizer ...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    expected_cols = len(vectorizer.vocabulary_)
    print(f"  Vocabulary size: {expected_cols:,}")

    pipeline_start = time.time()
    total = len(CONGRESSES_TO_REBUILD)

    for i, cong in enumerate(CONGRESSES_TO_REBUILD, 1):
        window_start = time.time()
        year1, year2 = CONGRESS_YEARS[cong]

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

        texts = (articles["title"].fillna("") + " " + articles["text"]).values
        n_docs = len(texts)

        meta = articles[["date", "paper", "title", "word_count", "year", "congress"]].reset_index(drop=True)
        n_papers = articles["paper"].nunique()
        del articles
        gc.collect()

        # Transform
        print(f"  [{i}/{total}] Congress {cong}: transforming {n_docs:,} articles ...")
        if n_docs < 1000:
            tfidf = vectorizer.transform(texts)
        else:
            chunks = np.array_split(texts, min(N_JOBS, n_docs))
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

        assert tfidf.shape[1] == expected_cols, \
            f"Got {tfidf.shape[1]} cols, expected {expected_cols}"

        sp.save_npz(OUT_DIR / f"07_newspaper_tfidf_cong_{cong}.npz", tfidf)
        meta.to_parquet(OUT_DIR / f"07_newspaper_meta_cong_{cong}.parquet")

        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / i
        remaining = avg_per * (total - i)
        print(f"    {tfidf.shape}  |  {n_papers} papers  |  "
              f"{elapsed:.0f}s  |  ETA: {remaining:.0f}s")

        del tfidf, meta
        gc.collect()

    print(f"\nDone in {time.time() - pipeline_start:.0f}s")
    print(f"Saved to -> {OUT_DIR}")
