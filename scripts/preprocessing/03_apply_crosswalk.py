"""
03_apply_crosswalk.py

Apply the crosswalk from 02 to the raw yearly parquets: standardize
paper names to canonical forms and filter to the target analysis sample.

Inputs:
  - data/intermediate/newspapers/yearly/newspapers_{year}.parquet
  - data/temp/02_paper_crosswalk.csv
  - data/temp/02_target_papers.csv

Outputs:
  - data/intermediate/newspapers/newspapers_{year}.parquet
    (only target papers, canonical names)
"""

import os
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

INPUT_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly"
CROSSWALK_PATH = BASE_DIR / "data" / "temp" / "02_paper_crosswalk.csv"
TARGET_PAPERS_PATH = BASE_DIR / "data" / "temp" / "02_target_papers.csv"

OUT_DIR = BASE_DIR / "data" / "intermediate" / "newspapers"


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    # Load crosswalk and target papers
    print("Loading crosswalk and target papers ...")
    crosswalk = pd.read_csv(CROSSWALK_PATH)
    target_papers = set(pd.read_csv(TARGET_PAPERS_PATH)["paper"].tolist())
    print(f"  Target papers: {len(target_papers)}")

    # Build lookup: only keep crosswalk entries whose canonical name is a target
    xwalk = crosswalk[crosswalk["canonical_name"].isin(target_papers)]
    raw_to_canon = dict(zip(xwalk["raw_name"], xwalk["canonical_name"]))
    lookup_set = set(raw_to_canon.keys())
    print(f"  Lookup set (incl. variants): {len(lookup_set)}")

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

        # Filter to target papers (matching raw variants)
        df = df[df["paper"].isin(lookup_set)].copy()

        # Standardize to canonical names
        df["paper"] = df["paper"].map(raw_to_canon)

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
