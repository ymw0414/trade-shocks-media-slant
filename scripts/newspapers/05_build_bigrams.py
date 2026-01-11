"""
05_build_bigrams.py

JOB:
- Build newspaper-side contiguous bigram count matrices
- Input:
    - clean_text from 04a
    - clean_title from 04b
- Output:
    - X_body (bigram counts)
    - X_title (bigram counts, same vocab)
    - vocabulary
    - metadata
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

CLEAN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"

BODY_FILE  = CLEAN_DIR / f"newspapers_congress_{CONGRESS}_clean_body.parquet"
TITLE_FILE = CLEAN_DIR / f"newspapers_congress_{CONGRESS}_clean_title.parquet"

OUT_X_BODY  = OUT_DIR / f"X_body_congress_{CONGRESS}.npz"
OUT_X_TITLE = OUT_DIR / f"X_title_congress_{CONGRESS}.npz"
OUT_VOCAB   = OUT_DIR / f"vocab_congress_{CONGRESS}.csv"
OUT_META    = OUT_DIR / f"meta_congress_{CONGRESS}.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("05_build_bigrams.py")
    print("Task: Build body + title bigram matrices (separately)")
    print(f"Congress: {CONGRESS}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading clean body and title data...")
    body_df  = pd.read_parquet(BODY_FILE).reset_index(drop=True)
    title_df = pd.read_parquet(TITLE_FILE).reset_index(drop=True)

    if len(body_df) != len(title_df):
        raise ValueError("Body and title row counts do not match")

    texts_body  = body_df["clean_text"].fillna("").tolist()
    texts_title = title_df["clean_title"].fillna("").tolist()

    # --------------------------------------------------
    # Vectorizer (shared vocabulary)
    # --------------------------------------------------

    vectorizer = CountVectorizer(
        ngram_range=(2, 2),
        lowercase=False,
        token_pattern=r"[a-z]{2,}",
        min_df=10
    )

    print("Fitting bigram vocabulary on BODY text only...")
    vectorizer.fit(texts_body)
    vocab_terms = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(vocab_terms):,}")

    # --------------------------------------------------
    # Transform BODY
    # --------------------------------------------------

    print("Transforming BODY text into bigram counts...")
    chunk_size = 50_000
    X_body_chunks = []

    for i in tqdm(range(0, len(texts_body), chunk_size), desc="BODY bigrams"):
        chunk = texts_body[i:i + chunk_size]
        X_body_chunks.append(vectorizer.transform(chunk))

    X_body = sp.vstack(X_body_chunks)

    # --------------------------------------------------
    # Transform TITLE (same vocab)
    # --------------------------------------------------

    print("Transforming TITLE text into bigram counts...")
    X_title_chunks = []

    for i in tqdm(range(0, len(texts_title), chunk_size), desc="TITLE bigrams"):
        chunk = texts_title[i:i + chunk_size]
        X_title_chunks.append(vectorizer.transform(chunk))

    X_title = sp.vstack(X_title_chunks)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    print("Saving outputs...")
    sp.save_npz(OUT_X_BODY, X_body)
    sp.save_npz(OUT_X_TITLE, X_title)

    pd.DataFrame({"term": vocab_terms}).to_csv(OUT_VOCAB, index=False)

    meta = body_df.drop(columns=["clean_text"], errors="ignore")
    meta.to_parquet(OUT_META)

    print("-" * 70)
    print(f"X_body shape : {X_body.shape}")
    print(f"X_title shape: {X_title.shape}")
    print("Done.")
    print("=" * 70)

if __name__ == "__main__":
    main()
