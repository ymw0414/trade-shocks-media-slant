import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

# --------------------------------------------------
# Paths
# --------------------------------------------------
if "SHIFTING_SLANT_DIR" not in os.environ:
    BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant")
else:
    BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REQUIRED_YEARS = {1992, 1993, 1994, 1995}
MIN_ARTICLES = 100

def simplify_name(name):
    """
    Groups similar names by removing common suffixes and punctuation.
    e.g., "The Times-Picayune" and "Times-Picayune, The" -> "timespicayune"
    """
    if pd.isna(name): return "unknown"
    name = str(name).lower()
    # Remove 'the', punctuation, and spaces
    name = re.sub(r'\bthe\b', '', name)
    name = re.sub(r'[^a-z]', '', name)
    return name.strip()

def main():
    files = sorted(list(SLANT_DIR.glob("news_slant_congress_*.parquet")))

    # We will store: { simplified_name: { original_name: count } }
    # and: { original_name: { year: count } }
    raw_counts = {}

    print(">>> Scanning files to identify stable papers...")
    for f in tqdm(files):
        # We only need 'paper' and 'date' for this check
        df = pd.read_parquet(f, columns=['paper', 'date'])
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

        # Filter for our stability period
        target_df = df[df['year'].isin(REQUIRED_YEARS)].copy()

        if target_df.empty:
            continue

        # Group by original name and year
        counts = target_df.groupby(['paper', 'year']).size()

        for (paper, year), count in counts.items():
            if paper not in raw_counts:
                raw_counts[paper] = {}
            raw_counts[paper][year] = raw_counts[paper].get(year, 0) + count

    # Filter for stability: Must have 100+ articles in ALL required years
    stable_papers = []
    for paper, year_dict in raw_counts.items():
        is_stable = all(year_dict.get(y, 0) >= MIN_ARTICLES for y in REQUIRED_YEARS)
        if is_stable:
            stable_papers.append({
                'paper': paper,
                'avg_articles': sum(year_dict.values()) / len(year_dict)
            })

    # Sort by volume and display
    stable_df = pd.DataFrame(stable_papers).sort_values('paper')

    print(f"\n" + "="*50)
    print(f"IDENTIFIED {len(stable_df)} STABLE PAPERS (1992-1995)")
    print("="*50)

    # Print the full list for you to see
    pd.set_option('display.max_rows', None)
    print(stable_df['paper'].to_string(index=False))

    # Save this list to a CSV for reference
    out_path = BASE_DIR / "data" / "geo" / "identified_targets_list.csv"
    stable_df.to_csv(out_path, index=False)
    print(f"\n>>> List saved to: {out_path}")

if __name__ == "__main__":
    main()