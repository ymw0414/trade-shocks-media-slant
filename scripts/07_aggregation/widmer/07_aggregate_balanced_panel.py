"""
07_aggregate_balanced_panel_strict_counts.py

[GOAL]
1. Identify 'Stable Papers' with Robust Volume:
   - Must exist in ALL years (1992, 1993, 1994, 1995, 1996).
   - Must have >= 100 articles in EACH of these years.
2. Aggregate Article-level Slant -> Newspaper-Year Level.
   - For valid papers, keep data for ALL years (1986-2008).
   - Filter Condition: Even for valid papers, drop specific years if article_count < 100.
   - Calculate Mean Slant & Weighted Mean Slant.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"
OUT_DIR = BASE_DIR / "data" / "analysis"

# Make output directory
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REQUIRED_YEARS = {1992, 1993, 1994, 1995, 1996}
MIN_ARTICLES_PER_YEAR = 100  # Threshold for validity

def get_slant_files():
    """Find all projected slant files from Step 06."""
    return sorted(list(SLANT_DIR.glob("news_slant_congress_*.parquet")))

def main():
    files = get_slant_files()
    if not files:
        print("Error: No slant files found. Did you run 06_project_widmer.py?")
        return

    print(f"Found {len(files)} slant files.")
    print(f"Criteria: Must have >= {MIN_ARTICLES_PER_YEAR} articles in EACH of {sorted(REQUIRED_YEARS)}")

    # --------------------------------------------------
    # PHASE 1: Identify "Robust Survivors" (Count Check)
    # --------------------------------------------------
    # We map: (Paper, Year) -> Article Count
    paper_year_counts = defaultdict(int)

    for f in tqdm(files, desc="Phase 1: Scanning Counts"):
        try:
            # Only load necessary columns
            df = pd.read_parquet(f, columns=['paper', 'date'])

            # Ensure Year exists
            if 'year' not in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['year'] = df['date'].dt.year

            df = df.dropna(subset=['paper', 'year'])

            # Count articles per paper-year
            counts = df.groupby(['paper', 'year']).size()

            # Accumulate counts (in case years are split across files)
            for (paper, year), count in counts.items():
                paper_year_counts[(paper, int(year))] += count

        except Exception as e:
            print(f"Warning: Could not read {f.name}: {e}")

    # Identify papers that meet the strict criteria
    # 1. Get list of all unique papers encountered
    all_papers = set(p for p, y in paper_year_counts.keys())

    valid_papers = []

    for paper in all_papers:
        # Check if it passes the test for ALL required years
        is_stable = True
        for req_year in REQUIRED_YEARS:
            count = paper_year_counts.get((paper, req_year), 0)
            if count < MIN_ARTICLES_PER_YEAR:
                is_stable = False
                break

        if is_stable:
            valid_papers.append(paper)

    valid_papers_set = set(valid_papers)

    print("-" * 40)
    print(f"Total Papers Scanned: {len(all_papers)}")
    print(f"Papers Passing Strict Check (>=100 arts in 92-96): {len(valid_papers)}")
    print("-" * 40)

    if len(valid_papers) == 0:
        print("STOP: No papers met the criteria.")
        return

    # --------------------------------------------------
    # PHASE 2: Aggregate Data (Filter Low Volume Years)
    # --------------------------------------------------
    print("Starting Phase 2: Aggregating & Filtering Panel Data...")

    yearly_aggregates = []

    for f in tqdm(files, desc="Phase 2: Aggregating"):
        try:
            df = pd.read_parquet(f)

            # 1. Keep only Valid Papers
            df = df[df['paper'].isin(valid_papers_set)].copy()

            if df.empty:
                continue

            # 2. Prepare Year
            if 'year' not in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year

            # 3. Calculate Weighted Slant components
            df['weighted_sum'] = df['slant'] * df['used_terms']

            # 4. Group by Paper-Year
            grouped = df.groupby(['paper', 'year'])

            agg_df = grouped.agg(
                article_count=('slant', 'count'),
                slant_mean=('slant', 'mean'),
                sum_weighted=('weighted_sum', 'sum'),
                sum_weights=('used_terms', 'sum')
            ).reset_index()

            # 5. [CRITICAL STEP] Drop years with insufficient articles
            # This applies to 1986-1991, 1997-2008 (and 92-96 is guaranteed safe)
            agg_df = agg_df[agg_df['article_count'] >= MIN_ARTICLES_PER_YEAR].copy()

            if agg_df.empty:
                continue

            # Finish weighted average calc
            agg_df['slant_weighted'] = agg_df['sum_weighted'] / agg_df['sum_weights']

            # Clean up
            agg_df = agg_df.drop(columns=['sum_weighted', 'sum_weights'])

            yearly_aggregates.append(agg_df)

        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # --------------------------------------------------
    # PHASE 3: Final Merge & Save
    # --------------------------------------------------
    if yearly_aggregates:
        final_df = pd.concat(yearly_aggregates, ignore_index=True)
        final_df = final_df.sort_values(by=['paper', 'year'])

        # Save
        out_csv = OUT_DIR / "newspaper_panel_1986_2008_robust.csv"
        out_pq = OUT_DIR / "newspaper_panel_1986_2008_robust.parquet"

        final_df.to_csv(out_csv, index=False)
        final_df.to_parquet(out_pq)

        print("-" * 40)
        print("AGGREGATION COMPLETE")
        print(f"Saved: {out_csv}")
        print(f"Total Rows: {len(final_df):,}")
        print(f"Unique Papers: {final_df['paper'].nunique()}")
        print("-" * 40)
    else:
        print("Error: No data left after filtering.")

if __name__ == "__main__":
    main()