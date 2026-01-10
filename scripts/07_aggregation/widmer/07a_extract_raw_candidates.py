"""
FILE: 07a_extract_raw_candidates.py
DESCRIPTION:
    - Scans raw congress files for the 1992-1995 window.
    - Identifies paper names that meet the 100+ articles/year stability threshold.
    - Outputs the RAW list of names for manual/hard-coded review in 07b.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
if "SHIFTING_SLANT_DIR" not in os.environ:
    BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant")
else:
    BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"
OUTPUT_FILE = BASE_DIR / "data" / "temp" / "candidate_papers_1992_1995.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REQUIRED_YEARS = {1992, 1993, 1994, 1995}
MIN_ARTICLES = 100

def main():
    files = sorted(list(SLANT_DIR.glob("news_slant_congress_*.parquet")))
    paper_year_counts = {}

    print(">>> [07a] Scanning raw data for 1992-1995 stability...")
    for f in tqdm(files, desc="Scanning Parquet Files"):
        try:
            # Load only necessary columns for efficiency
            df = pd.read_parquet(f, columns=['paper', 'date'])
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

            # Filter for the target stability window
            target_period = df[df['year'].isin(REQUIRED_YEARS)]
            if target_period.empty:
                continue

            # Article count per paper per year
            counts = target_period.groupby(['paper', 'year']).size()

            for (p, y), count in counts.items():
                paper_year_counts[(p, y)] = paper_year_counts.get((p, y), 0) + count
        except Exception as e:
            print(f"Error scanning {f.name}: {e}")

    # Identify papers that exist in all 4 years with 100+ articles each
    all_papers = set(p for p, y in paper_year_counts.keys())
    stable_list = []

    for p in sorted(all_papers):
        # Check if 100+ articles exist for every year in REQUIRED_YEARS
        if all(paper_year_counts.get((p, y), 0) >= MIN_ARTICLES for y in REQUIRED_YEARS):
            stable_list.append(p)

    # Save the RAW list
    pd.DataFrame({'paper': stable_list}).to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"Extraction Complete. Found {len(stable_list)} raw stable papers.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 50)

if __name__ == "__main__":
    main()