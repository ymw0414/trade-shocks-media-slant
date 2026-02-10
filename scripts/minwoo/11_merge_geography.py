"""
11_merge_geography.py

Merge geographic identifiers (county FIPS, commuting zone) onto the
newspaper-year panel from step 10.

Sources:
  - data/geo/newspaper_county_map.csv (Widmer's 133-paper mapping)
  - Manual additions for 9 papers with name mismatches or missing entries
  - data/raw/econ/crosswalk/cw_cty_czone/cw_cty_czone.dta (FIPS -> CZ)

Inputs:
  - data/processed/newspapers/minwoo/10_newspaper_year_panel.parquet

Outputs:
  - data/processed/newspapers/minwoo/11_newspaper_year_panel_geo.parquet
"""

import os
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

PANEL_PATH = BASE_DIR / "data" / "processed" / "newspapers" / "minwoo" / "10_newspaper_year_panel.parquet"
GEO_PATH = BASE_DIR / "data" / "geo" / "newspaper_county_map.csv"
CZ_PATH = BASE_DIR / "data" / "raw" / "econ" / "crosswalk" / "cw_cty_czone" / "cw_cty_czone.dta"

OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "minwoo"

# ------------------------------------------------------------------
# Name aliases: panel name -> geo map name
# (Papers that exist in both but with different formatting)
# ------------------------------------------------------------------
NAME_ALIASES = {
    "SACRAMENTO BEE": "Sacramento Bee, The (CA)",
    "THE SAN FRANCISCO CHRONICLE": "San Francisco Chronicle (CA)",
    "THE SEATTLE TIMES": "Seattle Times, The (WA)",
    "Star Tribune: Newspaper of the Twin Cities": "Star Tribune (Minneapolis, MN)",
}

# ------------------------------------------------------------------
# Manual entries: papers missing from Widmer's map entirely
# (city, state, fips verified for 1990s headquarters)
# ------------------------------------------------------------------
MANUAL_ENTRIES = [
    {"paper": "CHRISTIAN SCIENCE MONITOR", "city": "Boston", "state": "MA", "fips": 25025, "county_name": "Suffolk County"},
    {"paper": "Daily Pennsylvanian, The: University of Pennsylvania (Philadelphia, PA)", "city": "Philadelphia", "state": "PA", "fips": 42101, "county_name": "Philadelphia County"},
    {"paper": "Miami New Times (FL)", "city": "Miami", "state": "FL", "fips": 12086, "county_name": "Miami-Dade County"},
    {"paper": "The Washington Times", "city": "Washington", "state": "DC", "fips": 11001, "county_name": "District of Columbia"},
    {"paper": "USA TODAY", "city": "McLean", "state": "VA", "fips": 51059, "county_name": "Fairfax County"},
]


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    print("Merging geography onto newspaper-year panel ...\n")

    # 1. Load panel
    panel = pd.read_parquet(PANEL_PATH)
    n_papers = panel["paper"].nunique()
    print(f"  Panel: {len(panel):,} rows, {n_papers} papers")

    # 2. Load Widmer's county map
    geo = pd.read_csv(GEO_PATH)
    print(f"  Widmer geo map: {geo['paper'].nunique()} papers")

    # 3. Apply name aliases (rename panel names to match geo names)
    panel["paper_geo"] = panel["paper"].replace(NAME_ALIASES)

    # 4. Add manual entries to geo map
    manual_df = pd.DataFrame(MANUAL_ENTRIES)
    manual_df["source_category"] = "4_manual_minwoo"
    manual_df["match_type"] = "Manual"
    geo = pd.concat([geo, manual_df], ignore_index=True)
    print(f"  After manual additions: {geo['paper'].nunique()} papers")

    # 5. Merge geo onto panel
    geo_cols = geo[["paper", "city", "state", "fips", "county_name"]].copy()
    panel = panel.merge(geo_cols, left_on="paper_geo", right_on="paper",
                        how="left", suffixes=("", "_geo"))
    panel.drop(columns=["paper_geo", "paper_geo"], errors="ignore", inplace=True)

    # Check coverage
    n_matched = panel["fips"].notna().sum()
    n_total = len(panel)
    n_missing = panel[panel["fips"].isna()]["paper"].unique()
    print(f"\n  FIPS matched: {n_matched:,} / {n_total:,} rows")
    if len(n_missing) > 0:
        print(f"  WARNING: {len(n_missing)} papers without FIPS:")
        for p in n_missing:
            print(f"    - {p}")

    # 6. Load FIPS -> CZ crosswalk and merge
    cz = pd.read_stata(CZ_PATH)
    cz = cz.rename(columns={"cty_fips": "fips", "czone": "cz"})
    cz["fips"] = cz["fips"].astype("Int64")
    cz["cz"] = cz["cz"].astype("Int64")

    # Fix known FIPS code changes (Miami-Dade: 12086 -> old Dade 12025)
    FIPS_REMAP = {12086: 12025}
    panel["fips_cz"] = panel["fips"].replace(FIPS_REMAP)
    panel["fips_cz"] = panel["fips_cz"].astype("Int64")
    panel = panel.merge(cz, left_on="fips_cz", right_on="fips",
                        how="left", suffixes=("", "_cz_xw"))
    panel.drop(columns=["fips_cz", "fips_cz_xw"], errors="ignore", inplace=True)

    n_cz = panel["cz"].notna().sum()
    print(f"  CZ matched: {n_cz:,} / {n_total:,} rows")

    # Clean up redundant column
    if "paper_geo" in panel.columns:
        panel.drop(columns=["paper_geo"], inplace=True)

    # 7. Save
    out_path = OUT_DIR / "11_newspaper_year_panel_geo.parquet"
    panel.to_parquet(out_path)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Panel with geography")
    print("=" * 72)

    n_papers = panel["paper"].nunique()
    n_counties = panel["fips"].nunique()
    n_czs = panel["cz"].nunique()

    print(f"\n  Panel: {len(panel):,} rows ({n_papers} papers x {panel['year'].nunique()} years)")
    print(f"  Counties (FIPS): {n_counties}")
    print(f"  Commuting zones: {n_czs}")

    # Papers per CZ
    papers_per_cz = panel.groupby("cz")["paper"].nunique()
    print(f"\n  Papers per CZ: mean={papers_per_cz.mean():.1f}, "
          f"median={papers_per_cz.median():.0f}, "
          f"max={papers_per_cz.max()}")

    # Sample
    print(f"\n  Sample rows:")
    sample = panel[["paper", "city", "state", "fips", "county_name", "cz", "year", "net_slant_norm"]].drop_duplicates("paper").head(10)
    print(sample.to_string(index=False))

    print(f"\n  Saved to -> {out_path}")
    print("=" * 72)
