"""
FILE: 07c_aggregate_slant_panel.py
DESCRIPTION:
    - Loads the final target list.
    - Aggregates slant data from 1986 to 2004.
    - FIX: Explicit type casting and verification step for Akron 1986 data.
    - Output: 'newspaper_panel_1986_2004.csv'
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))
TARGET_LIST_FILE = BASE_DIR / "data" / "temp" / "final_target_papers.csv"
SLANT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant"
OUTPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004.csv"

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MAX_YEAR = 2004

NAME_FIX_MAP = {
    "Richmond Times-Dispatch (VA)": "Richmond Times-Dispatch",
    "St. Louis Post-Dispatch (MO)": "St. Louis Post-Dispatch",
    "SACRAMENTO BEE": "Sacramento Bee, The (CA)",
    "THE SAN FRANCISCO CHRONICLE": "San Francisco Chronicle (CA)",
    "THE SEATTLE TIMES": "Seattle Times, The (WA)",
    "Star Tribune: Newspaper of the Twin Cities": "Star Tribune (Minneapolis, MN)"
}

def main():
    if not TARGET_LIST_FILE.exists():
        print(f"Error: {TARGET_LIST_FILE} not found. Run 07b first.")
        return

    # Load Clean List
    target_df = pd.read_csv(TARGET_LIST_FILE)
    target_set = set(target_df['paper'].unique())
    print(f">>> [07c] Target Papers: {len(target_set)}")

    files = sorted(list(SLANT_DIR.glob("news_slant_congress_*.parquet")))
    all_chunks = []

    print(">>> Starting Aggregation...")

    for f in tqdm(files, desc="Processing Files"):
        try:
            # Load Data
            df = pd.read_parquet(f, columns=['paper', 'date', 'slant', 'used_terms'])

            # Extract Year
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
            df = df[df['year'] <= MAX_YEAR]

            if df.empty: continue

            # --- 1. Apply Name Mapping First ---
            df['paper_clean'] = df['paper'].replace(NAME_FIX_MAP).str.strip()

            # --- 2. Filter Targets ---
            df = df[df['paper_clean'].isin(target_set)]
            if df.empty: continue

            # --- 3. Explicit Type Casting (Critical Fix) ---
            df['slant'] = df['slant'].astype(float)
            df['used_terms'] = df['used_terms'].astype(float)

            # --- 4. Verify Calculation for Akron 1986 (Debug Check) ---
            # If this file contains the Akron data, print it to prove it's not 0.
            check_mask = (df['paper_clean'] == "Akron Beacon Journal (OH)") & (df['year'] == 1986)
            if check_mask.any():
                debug_val = (df.loc[check_mask, 'slant'] * df.loc[check_mask, 'used_terms']).sum()
                print(f"\n[VERIFY] Found Akron 1986 in {f.name}. Partial Weighted Sum: {debug_val}")

            # --- 5. Calculate Weighted Slant ---
            df['weighted_slant_sum'] = df['slant'] * df['used_terms']

            # --- 6. Groupby & Aggregate ---
            chunk_agg = df.groupby(['paper_clean', 'year']).agg(
                n_articles=('slant', 'count'),
                sum_weighted_slant=('weighted_slant_sum', 'sum'),
                sum_weights=('used_terms', 'sum')
            ).reset_index()

            chunk_agg.rename(columns={'paper_clean': 'paper'}, inplace=True)
            all_chunks.append(chunk_agg)

        except Exception as e:
            print(f"Error in {f.name}: {e}")

    if not all_chunks:
        print("No data found.")
        return

    # Combine All Chunks
    print(">>> Combining all chunks...")
    total_df = pd.concat(all_chunks)
    final_df = total_df.groupby(['paper', 'year']).sum().reset_index()

    # Final Division
    final_df = final_df[final_df['sum_weights'] > 0] # Avoid div by zero
    final_df['slant_weighted'] = final_df['sum_weighted_slant'] / final_df['sum_weights']

    # Sort
    final_df = final_df.sort_values(['paper', 'year'])

    # Save
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"Aggregation Complete.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 50)

if __name__ == "__main__":
    main()