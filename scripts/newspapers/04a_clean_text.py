# -------------------------------------------------------------------------
# FILE: 04a_clean_text.py
# DESCRIPTION:
#   - Newspaper-side preprocessing following Gentzkow & Shapiro (2010) and Widmer et al. (2020).
#   - This script cleans raw newspaper text by:
#     1. Lowercasing and tokenizing (words with 2+ alphabetic characters).
#     2. Removing standard English and U.S. state-related stopwords.
#     3. Stemming tokens using the Porter Stemmer.
#     4. Filtering out routine announcements (obituaries, weddings, etc.) based on title/body patterns.
#     5. Applying a length cap to ensure data quality.
# MODIFICATIONS:
#   - Organized code structure for better readability in PyCharm.
#   - Translated all internal comments into English.
# -------------------------------------------------------------------------

import os
import re
import time
import math
import argparse
import pandas as pd
from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

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

IN_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "biennial"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"

IN_FILE = IN_DIR / f"newspapers_congress_{CONGRESS}.parquet"
OUT_FILE = OUT_DIR / f"newspapers_congress_{CONGRESS}_clean.parquet"

# --------------------------------------------------
# Stopwords and Stemmer
# --------------------------------------------------
stemmer = PorterStemmer()
BASE_STOPWORDS = set(stopwords.words("english"))

STATE_WORDS = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa",
    "kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new","york","jersey","mexico","north","south","carolina","dakota",
    "ohio","oklahoma","oregon","pennsylvania","rhode","island",
    "tennessee","texas","utah","vermont","virginia","washington",
    "west","wisconsin","wyoming"
}

STOPWORDS = BASE_STOPWORDS | STATE_WORDS

# --------------------------------------------------
# Routine Announcement Filters
# --------------------------------------------------
TITLE_DROP_PATTERNS = re.compile(
    r"(?:obituary|death notice|births?|wedding|notice of sale under power)",
    flags=re.I,
)

BODY_DROP_PATTERNS = re.compile(
    r"(?:funeral service|hospital notes|wedding announcement)",
    flags=re.I,
)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------
TOKEN_RE = re.compile(r"[a-z]{2,}")

# --------------------------------------------------
# Cleaning Function
# --------------------------------------------------
def clean_text(text):
    """
    Lowercases, tokenizes, removes stopwords, and stems the input text.
    """
    if not isinstance(text, str):
        return ""
    tokens = TOKEN_RE.findall(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
def main():
    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing congress {CONGRESS}")
    print(f"Input:  {IN_FILE}")
    print(f"Output: {OUT_FILE}")

    # Load data
    df = pd.read_parquet(IN_FILE)
    print(f"Loaded rows: {len(df):,}")

    # Filter based on title patterns
    if "title" in df.columns:
        df = df[~df["title"].str.contains(TITLE_DROP_PATTERNS, na=False)]
        print(f"After title filter: {len(df):,}")

    # Ensure text column exists
    if "text" not in df.columns:
        raise ValueError("Missing 'text' column")

    # Filter based on body patterns
    df = df[~df["text"].str.contains(BODY_DROP_PATTERNS, na=False)]
    print(f"After body filter: {len(df):,}")

    print("Cleaning text...")

    texts = df["text"].tolist()
    n = len(texts)
    chunk_size = 100_000
    n_chunks = math.ceil(n / chunk_size)

    clean_texts = []
    start_time = time.time()

    # Process cleaning in chunks for memory efficiency
    for c in range(n_chunks):
        chunk = texts[c*chunk_size : (c+1)*chunk_size]
        clean_chunk = [clean_text(t) for t in chunk]
        clean_texts.extend(clean_chunk)

        done = min((c+1) * chunk_size, n)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (n - done) / rate if rate > 0 else float("inf")

        print(
            f"  chunk {c+1}/{n_chunks} | "
            f"{done:,}/{n:,} rows | "
            f"elapsed {elapsed/60:.1f} min | "
            f"ETA {remaining/60:.1f} min"
        )

    df["clean_text"] = clean_texts

    # --------------------------------------------------
    # Length Cap (CRITICAL FIX)
    # --------------------------------------------------
    MAX_LEN = 1000  # conservative, safe for 80-word articles
    before = len(df)
    df = df[df["clean_text"].str.len() <= MAX_LEN]
    after = len(df)

    print(f"Length cap applied (<= {MAX_LEN} chars): removed {before - after:,} rows")

    # Save final results
    df.to_parquet(OUT_FILE)
    print(f"Saved: {OUT_FILE} ({len(df):,} rows)")

if __name__ == "__main__":
    main()