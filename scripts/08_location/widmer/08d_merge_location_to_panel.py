"""
FILE: 08d_merge_location_to_panel.py
DESCRIPTION:
    1. Loads the Slant Panel (from 07c).
    2. Loads the County Map (from 08c).
    3. Merges them on 'paper'.
    4. Saves 'newspaper_panel_with_geo.csv'.
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))

# Input 1: Slant Panel (Yearly Data)
PANEL_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004.csv"

# Input 2: County Map (Paper -> FIPS)
GEO_MAP_FILE = BASE_DIR / "data" / "geo" / "newspaper_county_map.csv"

# Output: Merged Panel
OUTPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_with_geo.csv"


def main():
    if not PANEL_FILE.exists():
        print(f"Error: {PANEL_FILE} not found. Run 07c first.")
        return
    if not GEO_MAP_FILE.exists():
        print(f"Error: {GEO_MAP_FILE} not found. Run 08c first.")
        return

    print(">>> [08d] Merging Slant Panel with Geographic Info...")

    # 1. Load Data
    df_panel = pd.read_csv(PANEL_FILE)
    df_geo = pd.read_csv(GEO_MAP_FILE)

    print(f"    - Slant Panel Rows: {len(df_panel):,} (Paper-Years)")
    print(f"    - Geo Map Rows:     {len(df_geo):,} (Unique Papers)")

    # 2. Merge (Left Join on 'paper')
    # We want to keep all years in the panel, attaching the static geo info to each year.
    df_merged = pd.merge(df_panel, df_geo, on="paper", how="left")

    # 3. Validation
    # Check if any paper in the panel failed to get a FIPS code
    missing_geo = df_merged[df_merged['fips'].isnull()]

    if not missing_geo.empty:
        print("\n" + "!" * 50)
        print("WARNING: Some papers in the panel do not have location info!")
        print(missing_geo['paper'].unique())
        print("!" * 50 + "\n")
    else:
        print("\n>>> Verification Success: All panel observations have FIPS codes.")

    # 4. Format FIPS Code (Ensure 5 digits, e.g., 1015 -> "01015")
    # FIPS columns often get read as int, losing the leading zero.
    if 'fips' in df_merged.columns:
        df_merged['fips'] = df_merged['fips'].astype(str).str.replace('.0', '', regex=False)
        df_merged['fips'] = df_merged['fips'].str.zfill(5)

    # 5. Save
    df_merged.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"Merged Panel Saved: {OUTPUT_FILE}")
    print(f"Columns: {list(df_merged.columns)}")
    print(f"Total Rows: {len(df_merged)}")
    print("-" * 50)


if __name__ == "__main__":
    main()