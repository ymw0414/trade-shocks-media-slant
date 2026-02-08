"""
07a_standardize_papers.py

Load raw newspaper articles (1987-2004) from yearly parquets,
standardize paper names to a canonical form, filter to the 132
target local newspapers, and save clean intermediate files.

The data source changed naming conventions over time:
  - Added state abbreviation suffixes: "Houston Chronicle" -> "Houston Chronicle (TX)"
  - Reformatted "The X" to "X, The (ST)" and vice versa
  - ALL CAPS variants: "THE SAN FRANCISCO CHRONICLE"
  - Newspaper mergers: "Atlanta Journal-Constitution, The (GA)"

This step makes name standardization explicit and auditable,
separate from the TF-IDF transformation in 07b.

Inputs:
  - data/intermediate/newspapers/yearly/newspapers_{year}.parquet
  - data/temp/final_target_papers.csv  (132 target papers)

Outputs:
  - data/intermediate/newspapers/minwoo/newspapers_{year}.parquet
    (only target papers, standardized names)
"""

import os
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

INPUT_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly"
TARGET_PAPERS_PATH = BASE_DIR / "data" / "temp" / "final_target_papers.csv"

OUT_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "minwoo"

# ------------------------------------------------------------------
# Name variant mapping: raw name -> canonical target name.
# The data source changed naming conventions over time.
# ------------------------------------------------------------------
NAME_FIX_MAP = {
    # --- ALL CAPS variants ---
    "CHICAGO SUN-TIMES": "Chicago Sun-Times",
    "DALLAS MORNING NEWS": "The Dallas Morning News",
    "SACRAMENTO BEE": "Sacramento Bee, The (CA)",
    "THE SACRAMENTO BEE": "Sacramento Bee, The (CA)",
    "THE SAN FRANCISCO CHRONICLE": "San Francisco Chronicle (CA)",
    "THE SEATTLE TIMES": "Seattle Times, The (WA)",
    "THE MILWAUKEE JOURNAL": "The Milwaukee Journal",

    # --- Added state abbreviation suffix ---
    "Austin American-Statesman (TX)": "Austin American-Statesman",
    "Chicago Sun-Times (IL)": "Chicago Sun-Times",
    "Fort Worth Star-Telegram (TX)": "Fort Worth Star-Telegram",
    "Greensboro News & Record (NC)": "Greensboro News & Record",
    "Houston Chronicle (TX)": "Houston Chronicle",
    "Richmond Times-Dispatch (VA)": "Richmond Times-Dispatch",
    "San Antonio Express-News (TX)": "San Antonio Express-News",
    "Seattle Post-Intelligencer (WA)": "Seattle Post-Intelligencer",
    "St. Louis Post-Dispatch (MO)": "St. Louis Post-Dispatch",
    "St. Petersburg Times (FL)": "St. Petersburg Times",
    "Tulsa World (OK)": "Tulsa World",

    # --- "The X" reformatted to "X, The (ST)" ---
    "Arizona Daily Star, The (Tucson, AZ)": "The Arizona Daily Star",
    "Buffalo News, The (NY)": "The Buffalo News",
    "Cincinnati Post, The (OH)": "The Cincinnati Post",
    "Commercial Appeal, The (Memphis, TN)": "The Commercial Appeal",
    "Commercial Appeal, The (Memphis,TN)": "The Commercial Appeal",
    "Daily Oklahoman, The (Oklahoma City, OK)": "The Daily Oklahoman",
    "Dallas Morning News, The (TX)": "The Dallas Morning News",
    "Denver Post, The (CO)": "The Denver Post",
    "Hartford Courant, The (CT)": "The Hartford Courant",
    "Kansas City Star, The (MO)": "The Kansas City Star",
    "Kentucky Post, The (Covington, KY)": "The Kentucky Post",
    "News & Observer, The (Raleigh, NC)": "The News & Observer",
    "News Tribune, The (Tacoma, WA)": "The News Tribune",
    "Orange County Register, The (Santa Ana, CA)": "The Orange County Register",
    "Palm Beach Post, The (FL)": "The Palm Beach Post",
    "Tampa Tribune, The (FL)": "The Tampa Tribune",

    # --- "X, The (ST)" reformatted to "The X" ---
    "The Knoxville News-Sentinel": "Knoxville News-Sentinel, The (TN)",
    "The Miami Herald": "Miami Herald, The (FL)",
    "The Sacramento Bee": "Sacramento Bee, The (CA)",
    "The San Diego Union-Tribune": "San Diego Union-Tribune, The (CA)",
    "The Philadelphia Inquirer": "Philadelphia Inquirer, The (PA)",
    "Philadelphia Inquirer": "Philadelphia Inquirer, The (PA)",
    "Boston Herald": "Boston Herald (MA)",

    # --- Minor spelling / punctuation differences ---
    "News-Sentinel, The (Ft. Wayne, IN)": "News-Sentinel, The (Fort Wayne, IN)",
    "Times Leader, The (Wilkes-Barre, PA)": "Times Leader, The (Wilkes Barre, PA)",
    "Post-Tribune (Gary, IN)": "Post-Tribune (IN)",
    "York Daily Record": "York Daily Record (PA)",

    # --- Star Tribune subtitle change ---
    "Star Tribune: Newspaper of the Twin Cities": "Star Tribune (Minneapolis, MN)",
    "Star Tribune: Newspaper of the Twin Cities (Minneapolis, MN)": "Star Tribune (Minneapolis, MN)",

    # --- Nuevo Herald reformatted ---
    "El Nuevo Herald": "Nuevo Herald, El (Miami, FL)",

    # --- Newspaper mergers (map post-merger name to one target entry) ---
    "Atlanta Journal-Constitution, The (GA)": "The Atlanta Journal and The Atlanta Constitution",
    "Milwaukee Journal Sentinel": "The Milwaukee Journal",
    "Milwaukee Journal Sentinel (WI)": "The Milwaukee Journal",
}


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    # Load target paper list
    print("Loading target paper list ...")
    target_papers = set(pd.read_csv(TARGET_PAPERS_PATH)["paper"].tolist())
    print(f"  Target papers: {len(target_papers)}")

    # Build lookup set: target names + all raw variants
    paper_lookup = set(target_papers)
    for raw_name, clean_name in NAME_FIX_MAP.items():
        if clean_name in target_papers:
            paper_lookup.add(raw_name)
    print(f"  Lookup set (incl. variants): {len(paper_lookup)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each year
    print()
    total_in = 0
    total_out = 0

    for year in range(1987, 2005):
        parquet_path = INPUT_DIR / f"newspapers_{year}.parquet"

        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path.name} not found, skipping")
            continue

        df = pd.read_parquet(parquet_path)
        n_before = len(df)

        # Filter to target papers (matching both original and variant names)
        df = df[df["paper"].isin(paper_lookup)].copy()

        # Standardize variant names to canonical target names
        df["paper"] = df["paper"].replace(NAME_FIX_MAP)

        n_after = len(df)
        n_papers = df["paper"].nunique()
        total_in += n_before
        total_out += n_after

        # Save
        out_path = OUT_DIR / f"newspapers_{year}.parquet"
        df.to_parquet(out_path)

        pct = n_after / n_before * 100 if n_before > 0 else 0
        print(f"  {year}: {n_after:>9,} / {n_before:>9,} articles "
              f"({pct:5.1f}%) from {n_papers:>3} papers -> {out_path.name}")

    # Summary
    print()
    print("=" * 72)
    pct_total = total_out / total_in * 100 if total_in > 0 else 0
    print(f"  Total: {total_out:,} / {total_in:,} articles kept ({pct_total:.1f}%)")
    print(f"  Saved to -> {OUT_DIR}")
    print("=" * 72)
