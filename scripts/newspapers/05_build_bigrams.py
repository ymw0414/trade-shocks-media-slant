"""
05_build_bigrams.py

JOB:
- Build newspaper-side contiguous bigram count matrix
- Input: clean_text from 04_clean_text.py
- Output: sparse X matrix, vocabulary, metadata
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--congress", type=int, required=True)
args = parser.parse_args()
CONGRESS = args.congress

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

IN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"

IN_FILE = IN_DIR / f"newspapers_congress_{CONGRESS}_clean.parquet"

OUT_X = OUT_DIR / f"X_newspapers_congress_{CONGRESS}.npz"
OUT_VOCAB = OUT_DIR / f"vocab_newspapers_congress_{CONGRESS}.csv"
OUT_META = OUT_DIR / f"meta_newspapers_congress_{CONGRESS}.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("05_build_bigrams.py")
    print("Task: Build newspaper bigram count matrix")
    print(f"Congress: {CONGRESS}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Input ] {IN_FILE}")
    print(f"[Output] {OUT_X}")
    print(f"[Output] {OUT_VOCAB}")
    print(f"[Output] {OUT_META}")
    print("-" * 70)

    print("Loading cleaned newspaper data...")
    df = pd.read_parquet(IN_FILE)
    print(f"Documents loaded: {len(df):,}")

    if "clean_text" not in df.columns:
        raise ValueError("Missing clean_text column")

    texts = df["clean_text"].fillna("").tolist()

    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        lowercase=False,
        token_pattern=r"[a-z]{2,}",
        min_df=10
    )

    print("Fitting bigram vocabulary...")
    vectorizer.fit(texts)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out()):,}")

    print("Transforming documents into bigram counts...")
    chunk_size = 50_000
    X_chunks = []

    for i in tqdm(range(0, len(texts), chunk_size), desc="Bigram transform"):
        chunk = texts[i:i + chunk_size]
        X_chunks.append(vectorizer.transform(chunk))

    X = sp.vstack(X_chunks)

    print("Saving outputs...")
    sp.save_npz(OUT_X, X)

    vocab = pd.DataFrame({
        "term": vectorizer.get_feature_names_out()
    })
    vocab.to_csv(OUT_VOCAB, index=False)

    meta = df.drop(columns=["text", "clean_text"], errors="ignore")
    meta.to_parquet(OUT_META)

    print("-" * 70)
    print(f"Done. Final matrix shape: {X.shape}")
    print("=" * 70)

if __name__ == "__main__":
    main()
