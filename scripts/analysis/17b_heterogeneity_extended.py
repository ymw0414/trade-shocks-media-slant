"""
17b_heterogeneity_extended.py

Additional heterogeneity analyses:
  1. Pre-NAFTA slant (initially R-leaning vs D-leaning papers)
  2. Income (income1989)
  3. Newspaper size (mean pre-NAFTA article count)

Each uses the same approach as the education heterogeneity:
  - Median split: vuln_x_post + vuln_x_post × HighX
  - Continuous: vuln_x_post + vuln_x_post × X_std

All specifications include paper FE + year FE + division×year FE +
china_shock×year + manushare×year, clustered at CZ.
"""

import os
import sys
import numpy as np
import pandas as pd
import pyfixest as pf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
TAB_DIR = cfg.TAB_DIR

NAFTA_YEAR = 1994

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

OUTCOMES = [
    # (depvar, label, panel)
    ("ext_R",           "Share R-leaning",              "A"),
    ("ext_D",           "Share D-leaning",              "A"),
    ("int_right_norm",  r"Republican slant ($\tilde{R}$)", "B"),
    ("int_left_norm",   r"Democratic slant ($\tilde{L}$)", "B"),
    ("net_slant_norm",  r"Net slant ($\tilde{S}$)",     "C"),
]


def load_panel():
    """Load and prepare regression panel."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= 2004].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    years = sorted(df["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    return df, years


def build_controls_rhs(years):
    """Build the control RHS string."""
    base_yr = years[0]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    return " + ".join(china_vars + manu_vars)


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def run_heterogeneity(df, years, het_var, het_label, high_label, continuous_label):
    """
    Run median-split + continuous heterogeneity for a given variable.

    het_var: column name of the continuous heterogeneity variable
    high_label: label for above-median indicator
    continuous_label: label for standardized continuous variable
    """
    controls = build_controls_rhs(years)

    # Compute median and indicator
    paper_level = df.groupby("paper")[het_var].first()
    median_val = paper_level.median()
    n_lo = (paper_level <= median_val).sum()
    n_hi = (paper_level > median_val).sum()
    sd_val = paper_level.std()

    print(f"\n{'='*70}")
    print(f"  Heterogeneity: {het_label}")
    print(f"{'='*70}")
    print(f"  Median = {median_val:.4f}, SD = {sd_val:.4f}")
    print(f"  Low (<=median): {n_lo} papers, High (>median): {n_hi} papers")

    # Create variables
    df["_high"] = (df[het_var] > median_val).astype(int)
    df["_vuln_x_post_x_high"] = df["vuln_x_post"] * df["_high"]
    df["_het_std"] = (df[het_var] - df[het_var].mean()) / df[het_var].std()
    df["_vuln_x_post_x_std"] = df["vuln_x_post"] * df["_het_std"]

    results = []
    print(f"\n  {'Outcome':<28} {'Base':>16} {'Interaction':>16} {'Continuous':>16}")
    print(f"  {'':28s} {'coef (se)':>16} {'coef (se)':>16} {'coef (se)':>16}")
    print("  " + "-" * 76)

    for depvar, label, panel in OUTCOMES:
        row = {"depvar": depvar, "label": label, "panel": panel}

        # Median split
        fml_split = (f"{depvar} ~ vuln_x_post + _vuln_x_post_x_high + {controls} "
                     f"| paper_id + year + division^year")
        m1 = pf.feols(fml_split, data=df, vcov={"CRV1": "cz"})
        t1 = m1.tidy()
        row["base_coef"] = t1.loc["vuln_x_post", "Estimate"]
        row["base_se"] = t1.loc["vuln_x_post", "Std. Error"]
        row["base_p"] = t1.loc["vuln_x_post", "Pr(>|t|)"]
        row["inter_coef"] = t1.loc["_vuln_x_post_x_high", "Estimate"]
        row["inter_se"] = t1.loc["_vuln_x_post_x_high", "Std. Error"]
        row["inter_p"] = t1.loc["_vuln_x_post_x_high", "Pr(>|t|)"]

        # Continuous (standardized)
        fml_cont = (f"{depvar} ~ vuln_x_post + _vuln_x_post_x_std + {controls} "
                    f"| paper_id + year + division^year")
        m2 = pf.feols(fml_cont, data=df, vcov={"CRV1": "cz"})
        t2 = m2.tidy()
        row["cont_coef"] = t2.loc["_vuln_x_post_x_std", "Estimate"]
        row["cont_se"] = t2.loc["_vuln_x_post_x_std", "Std. Error"]
        row["cont_p"] = t2.loc["_vuln_x_post_x_std", "Pr(>|t|)"]
        row["N"] = m2._N

        # Print
        b = f"{row['base_coef']:.4f}{stars(row['base_p'])} ({row['base_se']:.4f})"
        i = f"{row['inter_coef']:.4f}{stars(row['inter_p'])} ({row['inter_se']:.4f})"
        c = f"{row['cont_coef']:.4f}{stars(row['cont_p'])} ({row['cont_se']:.4f})"
        print(f"  {label:<28} {b:>16} {i:>16} {c:>16}")

        results.append(row)

    print(f"\n  N = {results[0]['N']}")
    print(f"  * p<0.1, ** p<0.05, *** p<0.01")
    print(f"  Base = effect for low group (<=median)")
    print(f"  Interaction = differential for high group")
    print(f"  Continuous = per-SD effect")

    # Clean up temp columns
    df.drop(columns=["_high", "_vuln_x_post_x_high", "_het_std", "_vuln_x_post_x_std"], inplace=True)

    return pd.DataFrame(results), median_val, sd_val, n_lo, n_hi


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading panel ...")
    df, years = load_panel()
    print(f"  {len(df):,} obs, {df['paper_id'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs")

    # ---- 1. Pre-NAFTA slant ----
    pre = df[df["year"] < NAFTA_YEAR].groupby("paper")["net_slant_norm"].mean()
    df["pre_slant"] = df["paper"].map(pre)
    res_slant, med_sl, sd_sl, n_lo_sl, n_hi_sl = run_heterogeneity(
        df, years, "pre_slant",
        "Pre-NAFTA Slant (mean net_slant_norm, 1987-1993)",
        "HighSlant (>median)", "PreSlant_std")

    # ---- 2. Income ----
    res_income, med_inc, sd_inc, n_lo_inc, n_hi_inc = run_heterogeneity(
        df, years, "income1989",
        "Per-Capita Income (1989)",
        "HighIncome (>median)", "Income_std")

    # ---- 3. Newspaper size ----
    pre_size = df[df["year"] < NAFTA_YEAR].groupby("paper")["n_articles"].mean()
    df["pre_size"] = df["paper"].map(pre_size)
    res_size, med_sz, sd_sz, n_lo_sz, n_hi_sz = run_heterogeneity(
        df, years, "pre_size",
        "Newspaper Size (mean pre-NAFTA articles/year)",
        "Large (>median)", "Size_std")

    # Save all results
    all_res = pd.concat([
        res_slant.assign(het="pre_slant"),
        res_income.assign(het="income"),
        res_size.assign(het="size"),
    ], ignore_index=True)
    csv_path = TAB_DIR / "heterogeneity_extended.csv"
    all_res.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nResults saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
