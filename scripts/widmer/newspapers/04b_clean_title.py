# -------------------------------------------------------------------------
# FILE: 04b_clean_title.py
# DESCRIPTION:
#   - Newspaper-side title preprocessing following Gentzkow & Shapiro (2010).
#   - This script cleans the 'title' column independently using the same
#     logic as the body text cleaning (lowercase, stem, stopword removal).
#   - Use this to create a clean baseline for title-weighted analysis.
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
# Output directory remains the same as 04a
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"

IN_FILE = IN_DIR / f"newspapers_congress_{CONGRESS}.parquet"
# Output file specifically for clean title
OUT_FILE = OUT_DIR / f"newspapers_congress_{CONGRESS}_clean_title.parquet"

# --------------------------------------------------
# Stopwords and Stemmer
# --------------------------------------------------
stemmer = PorterStemmer()
BASE_STOPWORDS = set(stopwords.words("english"))
STATE_WORDS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new", "york", "jersey", "mexico", "north", "south", "carolina", "dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode", "island",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west", "wisconsin", "wyoming"
}
STOPWORDS = BASE_STOPWORDS | STATE_WORDS

# --------------------------------------------------
# Routine Announcement Filters
# --------------------------------------------------
TITLE_DROP_PATTERNS = re.compile(
    r"(?:obituary|death notice|births?|wedding|notice of sale under power)",
    flags=re.I,
)

# --------------------------------------------------
# Tokenization
# --------------------------------------------------
TOKEN_RE = re.compile(r"[a-z]{2,}")


# --------------------------------------------------
# Cleaning Function
# --------------------------------------------------
def clean_title_logic(text):
    """
    Standard cleaning logic applied to titles.
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing titles for congress {CONGRESS}")
    df = pd.read_parquet(IN_FILE)
    print(f"Loaded rows: {len(df):,}")

    # 1. Filter routine announcements from titles
    if "title" in df.columns:
        df = df[~df["title"].str.contains(TITLE_DROP_PATTERNS, na=False)]
        print(f"After title filter: {len(df):,}")
    else:
        print("Error: 'title' column not found.")
        return

    # 2. Process Cleaning for Titles
    print("Cleaning Titles...")
    titles = df["title"].fillna("").tolist()

    n = len(titles)
    chunk_size = 100_000
    n_chunks = math.ceil(n / chunk_size)

    clean_titles = []
    start_time = time.time()

    for c in range(n_chunks):
        t_chunk = titles[c * chunk_size: (c + 1) * chunk_size]

        # Clean each title in the chunk
        chunk_results = [clean_title_logic(t) for t in t_chunk]
        clean_titles.extend(chunk_results)

        done = min((c + 1) * chunk_size, n)
        elapsed = time.time() - start_time
        print(f"  chunk {c + 1}/{n_chunks} | {done:,}/{n:,} | {elapsed / 60:.1f} min")

    # Store results in a new column
    df["clean_title"] = clean_titles

    # Select only necessary columns to keep the file size small
    # You can merge this later with 04a output using an index or ID
    output_df = df[["clean_title"]]

    # 3. Save results
    output_df.to_parquet(OUT_FILE)
    print(f"Saved Clean Titles: {OUT_FILE} ({len(output_df):,} rows)")



if __name__ == "__main__":
    main()