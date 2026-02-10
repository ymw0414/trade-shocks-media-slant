"""
13_merge_panel.py

Merge the newspaper-year slant panel (step 11) with CZ-level NAFTA
exposure variables (step 12) to create the final regression-ready dataset.

Inputs:
  - data/processed/newspapers/minwoo/11_newspaper_year_panel_geo.parquet
  - data/processed/econ/minwoo/12_nafta_vars_cz.parquet
  - data/processed/econ/minwoo/12_nafta_vars_county.parquet  (for education/income controls)
  - data/processed/econ/minwoo/12c_china_shock_cz.parquet    (ADH China shock)
  - data/raw/econ/crosswalk/cw_cty_czone/cw_cty_czone.dta

Outputs:
  - data/processed/panel/minwoo/13_regression_panel.parquet
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# Inputs
NEWSPAPER_PANEL = (BASE_DIR / "data" / "processed" / "newspapers" / "minwoo"
                   / "11_newspaper_year_panel_geo.parquet")
CZ_NAFTA = (BASE_DIR / "data" / "processed" / "econ" / "minwoo"
            / "12_nafta_vars_cz.parquet")
COUNTY_NAFTA = (BASE_DIR / "data" / "processed" / "econ" / "minwoo"
                / "12_nafta_vars_county.parquet")
CHINA_SHOCK = (BASE_DIR / "data" / "processed" / "econ" / "minwoo"
               / "12c_china_shock_cz.parquet")
CZ_XW_PATH = (BASE_DIR / "data" / "raw" / "econ" / "crosswalk"
              / "cw_cty_czone" / "cw_cty_czone.dta")

# Output
OUT_DIR = BASE_DIR / "data" / "processed" / "panel" / "minwoo"

NAFTA_YEAR = 1994


def aggregate_county_controls_to_cz():
    """Population-weighted CZ aggregation of education/income controls.

    Step 12's CZ collapse only keeps vulnerability, manushare, pop, emp.
    Here we aggregate the remaining county-level controls to CZ.
    """
    print("  Aggregating county controls to CZ ...")

    cty = pd.read_parquet(COUNTY_NAFTA)

    # Time-invariant controls: one row per county
    controls = cty.drop_duplicates("county")[
        ["county", "pop1990_total",
         "less_highschool1990", "highschool1990",
         "college1990", "bachelor_higher1990", "income1989"]
    ].copy()

    # County -> CZ crosswalk
    cz_xw = pd.read_stata(CZ_XW_PATH, convert_dates=False)
    cz_xw = cz_xw.rename(columns={"cty_fips": "county", "czone": "cz"})
    cz_xw["county"] = cz_xw["county"].astype(int)
    cz_xw["cz"] = cz_xw["cz"].astype(int)
    cz_xw = cz_xw.drop_duplicates("county")

    merged = controls.merge(cz_xw[["county", "cz"]], on="county", how="inner")
    print(f"    {merged['county'].nunique()} counties matched to {merged['cz'].nunique()} CZs")

    # Population-weighted mean
    control_cols = ["less_highschool1990", "highschool1990",
                    "college1990", "bachelor_higher1990", "income1989"]

    def pop_weighted_mean(g):
        w = g["pop1990_total"].fillna(0)
        w_sum = w.sum()
        if w_sum == 0:
            return pd.Series({c: np.nan for c in control_cols})
        return pd.Series({c: np.average(g[c].fillna(0), weights=w)
                          for c in control_cols})

    cz_controls = merged.groupby("cz").apply(
        pop_weighted_mean, include_groups=False
    ).reset_index()

    print(f"    CZ controls: {len(cz_controls)} CZs")
    return cz_controls


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load newspaper panel
    print("Loading newspaper panel (step 11) ...")
    panel = pd.read_parquet(NEWSPAPER_PANEL)
    panel["cz"] = panel["cz"].astype("Int64")
    print(f"  {len(panel):,} rows, {panel['paper'].nunique()} papers, "
          f"{panel['cz'].nunique()} CZs, years {panel['year'].min()}-{panel['year'].max()}")

    # Drop rows without CZ assignment
    n_no_cz = panel["cz"].isna().sum()
    if n_no_cz > 0:
        print(f"  Dropping {n_no_cz} rows without CZ assignment")
        panel = panel[panel["cz"].notna()].copy()

    panel["cz"] = panel["cz"].astype(int)

    # 2. Load CZ NAFTA vars (time-invariant + emp)
    print("Loading CZ NAFTA vars (step 12) ...")
    cz_nafta = pd.read_parquet(CZ_NAFTA)
    cz_nafta = cz_nafta.rename(columns={"czone": "cz"})

    # Split time-invariant and time-varying
    cz_ti = cz_nafta.drop_duplicates("cz")[
        ["cz", "vulnerability1990_scaled", "manushare1990", "pop1990_total"]
    ]
    cz_tv = cz_nafta[["cz", "year", "emp"]]

    # 3. Aggregate county controls to CZ
    cz_controls = aggregate_county_controls_to_cz()

    # 4. Merge everything onto newspaper panel
    print("Merging ...")
    n0 = len(panel)

    # Time-invariant CZ vars
    panel = panel.merge(cz_ti, on="cz", how="left")
    # Time-varying CZ vars (emp)
    panel = panel.merge(cz_tv, on=["cz", "year"], how="left")
    # Education/income controls
    panel = panel.merge(cz_controls, on="cz", how="left")
    # ADH China shock (CZ-level, cross-sectional)
    china = pd.read_parquet(CHINA_SHOCK)
    panel = panel.merge(china, on="cz", how="left")

    assert len(panel) == n0, f"Row count changed: {n0} -> {len(panel)}"

    n_matched = panel["vulnerability1990_scaled"].notna().sum()
    print(f"  {n_matched}/{n0} rows matched with vulnerability data "
          f"({panel.loc[panel['vulnerability1990_scaled'].notna(), 'cz'].nunique()} CZs)")
    n_china = panel["china_shock"].notna().sum()
    print(f"  {n_china}/{n0} rows matched with China shock data "
          f"({panel.loc[panel['china_shock'].notna(), 'cz'].nunique()} CZs)")

    # 5. Add NAFTA timing variables
    panel["post_nafta"] = (panel["year"] >= NAFTA_YEAR).astype(int)
    panel["vuln_x_post"] = panel["vulnerability1990_scaled"] * panel["post_nafta"]

    # 6. Sort and save
    panel = panel.sort_values(["paper", "year"]).reset_index(drop=True)

    out_path = OUT_DIR / "13_regression_panel.parquet"
    panel.to_parquet(out_path)

    # Summary
    print(f"\n{'=' * 60}")
    print("REGRESSION PANEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Shape: {panel.shape[0]:,} rows x {panel.shape[1]} cols")
    print(f"  Papers: {panel['paper'].nunique()}")
    print(f"  CZs:    {panel['cz'].nunique()}")
    print(f"  Years:  {panel['year'].min()}-{panel['year'].max()}")
    print(f"  Pre-NAFTA:  {(panel['post_nafta'] == 0).sum():,} obs")
    print(f"  Post-NAFTA: {(panel['post_nafta'] == 1).sum():,} obs")

    has_vuln = panel[panel["vulnerability1990_scaled"].notna()]
    print(f"\n  vulnerability1990_scaled:")
    print(f"    mean={has_vuln['vulnerability1990_scaled'].mean():.4f}, "
          f"std={has_vuln['vulnerability1990_scaled'].std():.4f}")
    print(f"    min={has_vuln['vulnerability1990_scaled'].min():.4f}, "
          f"max={has_vuln['vulnerability1990_scaled'].max():.4f}")

    print(f"\n  Columns: {panel.columns.tolist()}")
    print(f"\n  Saved -> {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
