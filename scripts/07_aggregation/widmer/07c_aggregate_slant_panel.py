"""
FILE: 07c_aggregate_slant_panel.py
DESCRIPTION:
    - Loads the final target newspaper list (final_target_papers.csv).
    - Iterates over article-level slant files produced by 06_project_widmer.py
      (news_slant_congress_*.parquet).
    - Keeps ONLY articles classified as trade-related
      (is_trade == 1, as defined in 06_project_widmer.py using stemmed
       trade-vocabulary matching).
    - Restricts the sample to articles published between 1986 and 2004 (inclusive).
    - Harmonizes newspaper names using NAME_FIX_MAP.
    - Keeps only newspapers in the final target list.
    - Aggregates article-level slant to a newspaper–year panel using
      used_terms-weighted averages.
    - Output: data/analysis/newspaper_panel_1986_2004.csv
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
MIN_YEAR = 1986
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

    # Load target list
    target_df = pd.read_csv(TARGET_LIST_FILE)
    target_set = set(target_df["paper"].unique())
    print(f">>> [07c] Target Papers: {len(target_set)}")

    files = sorted(SLANT_DIR.glob("news_slant_congress_*.parquet"))
    all_chunks = []

    print(">>> Starting aggregation...")

    for f in tqdm(files, desc="Processing files"):
        try:
            # --------------------------------------------------
            # CHANGE 1: load is_trade
            # --------------------------------------------------
            df = pd.read_parquet(
                f,
                columns=["paper", "date", "slant", "used_terms", "is_trade"]
            )

            # Extract and filter year (STRICT)
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
            df = df[df["year"].between(MIN_YEAR, MAX_YEAR)]
            if df.empty:
                continue

            # --------------------------------------------------
            # CHANGE 2: trade filter
            # --------------------------------------------------
            df = df[df["is_trade"] == 1]
            if df.empty:
                continue

            # Apply name fixes
            df["paper_clean"] = df["paper"].replace(NAME_FIX_MAP).str.strip()

            # Keep only target papers
            df = df[df["paper_clean"].isin(target_set)]
            if df.empty:
                continue

            # Explicit type casting
            df["slant"] = df["slant"].astype(float)
            df["used_terms"] = df["used_terms"].astype(float)

            # Debug check: Akron 1986
            check_mask = (
                (df["paper_clean"] == "Akron Beacon Journal (OH)") &
                (df["year"] == 1986)
            )
            if check_mask.any():
                debug_val = (df.loc[check_mask, "slant"] *
                             df.loc[check_mask, "used_terms"]).sum()
                print(f"\n[VERIFY] Akron 1986 in {f.name}: {debug_val}")

            # Weighted slant
            df["weighted_slant_sum"] = df["slant"] * df["used_terms"]

            # Aggregate within file
            chunk_agg = (
                df.groupby(["paper_clean", "year"])
                  .agg(
                      n_articles=("slant", "count"),
                      sum_weighted_slant=("weighted_slant_sum", "sum"),
                      sum_weights=("used_terms", "sum")
                  )
                  .reset_index()
            )

            chunk_agg.rename(columns={"paper_clean": "paper"}, inplace=True)
            all_chunks.append(chunk_agg)

        except Exception as e:
            print(f"Error in {f.name}: {e}")

    if not all_chunks:
        print("No data found.")
        return

    # Combine all chunks
    print(">>> Combining all chunks...")
    total_df = pd.concat(all_chunks, ignore_index=True)

    final_df = (
        total_df
        .groupby(["paper", "year"], as_index=False)
        .sum()
    )

    # Final weighted slant
    final_df = final_df[final_df["sum_weights"] > 0]
    final_df["slant_weighted"] = (
        final_df["sum_weighted_slant"] / final_df["sum_weights"]
    )

    # Sort and save
    final_df = final_df.sort_values(["paper", "year"])

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print("Aggregation complete.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 50)

if __name__ == "__main__":
    main()
