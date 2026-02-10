"""
15_did_regression.py

Difference-in-differences regressions of NAFTA tariff vulnerability
on newspaper slant, following Choi et al. (2024).

Specifications (all cluster SE at CZ level):
  (1) paper FE + year FE + division×year FE
  (2) + manushare1990 × year
  (3) + china_shock × year + manushare1990 × year

Inputs:
  - data/processed/panel/minwoo/13_regression_panel.parquet

Outputs:
  - output/tables/did_results.tex
  - output/tables/did_results.csv
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = (BASE_DIR / "data" / "processed" / "panel" / "minwoo"
              / "13_regression_panel.parquet")
TAB_DIR = BASE_DIR / "output" / "tables"

NAFTA_YEAR = 1994
END_YEAR = 2000  # Exclude China shock period

# State FIPS -> Census Division mapping
STATE_TO_DIVISION = {
    9: 1, 23: 1, 25: 1, 33: 1, 44: 1, 50: 1,
    34: 2, 36: 2, 42: 2,
    17: 3, 18: 3, 26: 3, 39: 3, 55: 3,
    19: 4, 20: 4, 27: 4, 29: 4, 31: 4, 38: 4, 46: 4,
    10: 5, 11: 5, 12: 5, 13: 5, 24: 5, 37: 5, 45: 5, 51: 5, 54: 5,
    1: 6, 21: 6, 28: 6, 47: 6,
    5: 7, 22: 7, 40: 7, 48: 7,
    4: 8, 8: 8, 16: 8, 30: 8, 32: 8, 35: 8, 49: 8, 56: 8,
    2: 9, 6: 9, 15: 9, 41: 9, 53: 9,
}


def load_panel():
    """Load and prepare regression panel."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()

    if END_YEAR is not None:
        df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes

    # Post × vulnerability
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    years = sorted(df["year"].unique())
    base_yr = years[0]

    # Manushare × year (omit first year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

    # China shock × year (omit first year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    return df, years


def run_did(df, depvar, years, label):
    """Run DiD specifications for one outcome."""
    print(f"\n{'='*60}")
    print(f"  {label} ({depvar})")
    print(f"{'='*60}")

    base_yr = years[0]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_rhs = " + ".join(manu_vars)
    china_rhs = " + ".join(china_vars)

    specs = [
        ("spec1", "vuln_x_post",
         "Paper + Year + Division×Year FE"),
        ("spec2", f"vuln_x_post + {manu_rhs}",
         "+ Manushare×Year"),
        ("spec3", f"vuln_x_post + {china_rhs} + {manu_rhs}",
         "+ China shock×Year + Manushare×Year"),
    ]

    results = {}
    for spec_name, rhs, desc in specs:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy().loc["vuln_x_post"]
        results[spec_name] = {
            "coef": t["Estimate"], "se": t["Std. Error"],
            "pval": t["Pr(>|t|)"], "N": m._N, "R2": m._r2,
        }
        print(f"\n  {desc}")
        print(f"    vuln×post = {t['Estimate']:.6f} (SE={t['Std. Error']:.6f}), "
              f"p={t['Pr(>|t|)']:.4f}")
        print(f"    N={m._N}, R2={m._r2:.4f}")

    return results


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel ...")
    df, years = load_panel()
    print(f"  {len(df):,} obs, {df['paper_id'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs, years {df['year'].min()}-{df['year'].max()}")
    print(f"  Post-NAFTA: {(df['post']==1).sum():,} obs, "
          f"Pre-NAFTA: {(df['post']==0).sum():,} obs")

    outcomes = [
        ("right_norm",               "Right Intensity"),
        ("left_norm",                "Left Intensity"),
        ("net_slant_norm",           "Net Slant"),
        ("politicization_norm",      "Politicization"),
        ("right_norm_econ",          "Right Intensity (Econ)"),
        ("left_norm_econ",           "Left Intensity (Econ)"),
        ("net_slant_norm_econ",      "Net Slant (Econ)"),
        ("politicization_norm_econ", "Politicization (Econ)"),
        ("econ_share",               "Econ Article Share"),
    ]

    all_results = []
    for depvar, label in outcomes:
        res = run_did(df, depvar, years, label)
        for spec, vals in res.items():
            all_results.append({"depvar": depvar, "label": label,
                                "spec": spec, **vals})

    # Save CSV
    res_df = pd.DataFrame(all_results)
    csv_path = TAB_DIR / "did_results.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Results saved: {csv_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY: vuln×post coefficients")
    print(f"{'='*90}")
    print(f"{'Outcome':<28} {'Spec 1':>18} {'Spec 2 (+manu)':>18} {'Spec 3 (+china,manu)':>22}")
    print(f"{'':<28} {'coef (se)':>18} {'coef (se)':>18} {'coef (se)':>22}")
    print(f"{'-'*86}")
    for depvar, label in outcomes:
        s1 = res_df[(res_df["depvar"]==depvar) & (res_df["spec"]=="spec1")].iloc[0]
        s2 = res_df[(res_df["depvar"]==depvar) & (res_df["spec"]=="spec2")].iloc[0]
        s3 = res_df[(res_df["depvar"]==depvar) & (res_df["spec"]=="spec3")].iloc[0]

        def fmt(row):
            stars = ""
            if row["pval"] < 0.01:
                stars = "***"
            elif row["pval"] < 0.05:
                stars = "**"
            elif row["pval"] < 0.1:
                stars = "*"
            return f"{row['coef']:.4f}{stars} ({row['se']:.4f})"

        print(f"{label:<28} {fmt(s1):>18} {fmt(s2):>18} {fmt(s3):>22}")

    print(f"\nN = {res_df['N'].iloc[0]}, CZ clusters = {df['cz'].nunique()}")
    print("* p<0.1, ** p<0.05, *** p<0.01")
    print("Done.")


if __name__ == "__main__":
    main()
