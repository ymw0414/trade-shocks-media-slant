# -------------------------------------------------------------------------
# FILE: 01a_unsupervised_clustering.py
# DESCRIPTION:
#   - Unsupervised clustering of newspaper articles (pre-1990 only)
#   - Representation: TF-IDF (1–2 grams)
#   - Algorithm: KMeans
#   - Purpose: discover section-like clusters (econ, crime, sports, etc.)
# -------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# --------------------------------------------------
# Config
# --------------------------------------------------
TRAIN_END_YEAR = 1989
K = 10
N_INIT = 20
TOP_TERMS = 20

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
CLEAN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"
OUT_DIR   = BASE_DIR / "data" / "processed" / "newspapers" / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLUSTER = OUT_DIR / f"clusters_pre1990_K{K}.parquet"
OUT_TERMS   = OUT_DIR / f"cluster_top_terms_pre1990_K{K}.txt"

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("=" * 70)
    print("01a_unsupervised_clustering.py")
    print(f"Training years : <= {TRAIN_END_YEAR}")
    print(f"K              : {K}")
    print("=" * 70)

    # --------------------------------------------------
    # Load data (ALL congress files)
    # --------------------------------------------------
    files = sorted(CLEAN_DIR.glob("newspapers_congress_*_clean.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    df["year"] = pd.to_datetime(df["date"]).dt.year
    df = df[df["year"] <= TRAIN_END_YEAR].reset_index(drop=True)

    if "clean_text" not in df.columns:
        raise ValueError("clean_text column missing")

    texts = df["clean_text"].fillna("")

    print(f"Documents used: {len(df):,}")

    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9
    )

    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    # --------------------------------------------------
    # KMeans
    # --------------------------------------------------
    km = KMeans(
        n_clusters=K,
        n_init=N_INIT,
        random_state=0
    )

    print("Running KMeans...")
    df["cluster"] = km.fit_predict(X)

    # --------------------------------------------------
    # Save cluster assignments
    # --------------------------------------------------
    df[["date", "year", "cluster"]].to_parquet(OUT_CLUSTER)

    # --------------------------------------------------
    # Top terms per cluster
    # --------------------------------------------------
    print("Extracting top terms per cluster...")
    centers = km.cluster_centers_

    with open(OUT_TERMS, "w", encoding="utf-8") as f:
        for k in range(K):
            top_idx = centers[k].argsort()[::-1][:TOP_TERMS]
            top_terms = terms[top_idx]

            f.write(f"\n=== Cluster {k} ===\n")
            f.write(", ".join(top_terms) + "\n")

    print("-" * 70)
    print(f"Saved clusters : {OUT_CLUSTER}")
    print(f"Saved terms    : {OUT_TERMS}")
    print("=" * 70)

if __name__ == "__main__":
    main()
