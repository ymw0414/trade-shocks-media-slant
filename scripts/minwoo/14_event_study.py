"""
14_event_study.py

Event study of NAFTA tariff vulnerability on newspaper slant,
following the Choi et al. (2024) specification adapted for newspaper-level data.

Specifications:
  Spec 1: paper FE + year FE,                    cluster(CZ)
  Spec 2: paper FE + year FE + division×year FE,  cluster(CZ)

Inputs:
  - data/processed/panel/minwoo/13_regression_panel.parquet

Outputs:
  - output/figures/event_study_{depvar}.png   (one per outcome)
  - output/tables/event_study_coefficients.csv
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = (BASE_DIR / "data" / "processed" / "panel" / "minwoo"
              / "13_regression_panel.parquet")
FIG_DIR = BASE_DIR / "output" / "figures"
TAB_DIR = BASE_DIR / "output" / "tables"

BASE_YEAR = 1993  # Last pre-NAFTA year

# State FIPS -> Census Division mapping
STATE_TO_DIVISION = {
    # New England
    9: 1, 23: 1, 25: 1, 33: 1, 44: 1, 50: 1,
    # Middle Atlantic
    34: 2, 36: 2, 42: 2,
    # East North Central
    17: 3, 18: 3, 26: 3, 39: 3, 55: 3,
    # West North Central
    19: 4, 20: 4, 27: 4, 29: 4, 31: 4, 38: 4, 46: 4,
    # South Atlantic
    10: 5, 11: 5, 12: 5, 13: 5, 24: 5, 37: 5, 45: 5, 51: 5, 54: 5,
    # East South Central
    1: 6, 21: 6, 28: 6, 47: 6,
    # West South Central
    5: 7, 22: 7, 40: 7, 48: 7,
    # Mountain
    4: 8, 8: 8, 16: 8, 30: 8, 32: 8, 35: 8, 49: 8, 56: 8,
    # Pacific
    2: 9, 6: 9, 15: 9, 41: 9, 53: 9,
}


def load_panel():
    """Load regression panel and create event-study interaction variables."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()

    # State FIPS and Census Division
    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)

    # Paper numeric ID for fixed effects
    df["paper_id"] = df["paper"].astype("category").cat.codes

    # Create year-specific vulnerability interactions (omit base year)
    years = sorted(df["year"].unique())
    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = ((df["year"] == yr).astype(float)
                           * df["vulnerability1990_scaled"])

    return df, years


def run_event_study(df, depvar, years, label):
    """Run two event-study specifications and collect coefficients.

    Spec 1: paper FE + year FE, cluster(CZ)
    Spec 2: paper FE + year FE + division×year FE, cluster(CZ)
    """
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    vul_rhs = " + ".join(vul_vars)

    print(f"\n  [{label}] {depvar}")

    # Spec 1: two-way FE
    fml1 = f"{depvar} ~ {vul_rhs} | paper_id + year"
    m1 = pf.feols(fml1, data=df, vcov={"CRV1": "cz"})
    print(f"    Spec 1 (paper + year FE):        N={m1._N}, R2={m1._r2:.4f}")

    # Spec 2: + division × year FE
    fml2 = f"{depvar} ~ {vul_rhs} | paper_id + year + division^year"
    m2 = pf.feols(fml2, data=df, vcov={"CRV1": "cz"})
    print(f"    Spec 2 (+ division×year FE):     N={m2._N}, R2={m2._r2:.4f}")

    # Collect coefficients
    rows = []
    for yr in years:
        if yr == BASE_YEAR:
            rows.append({"year": yr, "coef1": 0, "se1": 0,
                         "ci_lo1": 0, "ci_hi1": 0,
                         "coef2": 0, "se2": 0,
                         "ci_lo2": 0, "ci_hi2": 0})
            continue

        vname = f"vul_{yr}"

        t1 = m1.tidy()
        r1 = t1.loc[vname]
        coef1, se1 = r1["Estimate"], r1["Std. Error"]
        ci_lo1, ci_hi1 = r1["2.5%"], r1["97.5%"]

        t2 = m2.tidy()
        r2 = t2.loc[vname]
        coef2, se2 = r2["Estimate"], r2["Std. Error"]
        ci_lo2, ci_hi2 = r2["2.5%"], r2["97.5%"]

        rows.append({"year": yr, "coef1": coef1, "se1": se1,
                     "ci_lo1": ci_lo1, "ci_hi1": ci_hi1,
                     "coef2": coef2, "se2": se2,
                     "ci_lo2": ci_lo2, "ci_hi2": ci_hi2})

    return pd.DataFrame(rows)


def plot_event_study(coefs, depvar_label, out_path):
    """Plot event-study coefficients with 95% CI for two specifications."""
    fig, ax = plt.subplots(figsize=(10, 6))

    yrs = coefs["year"].values
    offset = 0.15

    # Spec 1: two-way FE
    ax.errorbar(yrs - offset, coefs["coef1"],
                yerr=[coefs["coef1"] - coefs["ci_lo1"],
                      coefs["ci_hi1"] - coefs["coef1"]],
                fmt="o", color="#2171b5", markersize=5, capsize=3,
                linewidth=1.2, label="Paper + Year FE")

    # Spec 2: + division × year FE
    ax.errorbar(yrs + offset, coefs["coef2"],
                yerr=[coefs["coef2"] - coefs["ci_lo2"],
                      coefs["ci_hi2"] - coefs["coef2"]],
                fmt="s", color="#cb181d", markersize=5, capsize=3,
                linewidth=1.2, label="+ Division \u00d7 Year FE")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7, label="NAFTA (1994)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title(f"Event Study: {depvar_label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks(yrs)
    ax.set_xticklabels(yrs, rotation=45, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved: {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel ...")
    df, years = load_panel()
    print(f"  {len(df):,} obs, {df['paper_id'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs, {df['division'].nunique()} divisions")

    outcomes = [
        ("net_slant_norm",           "Net Slant (Normalized)"),
        ("politicization_norm",      "Politicization (Normalized)"),
        ("net_slant_norm_econ",      "Net Slant \u2014 Econ Articles (Normalized)"),
        ("politicization_norm_econ", "Politicization \u2014 Econ Articles (Normalized)"),
        ("econ_share",               "Economic Article Share"),
    ]

    all_coefs = []
    for depvar, label in outcomes:
        coefs = run_event_study(df, depvar, years, label)
        coefs["depvar"] = depvar

        out_name = depvar.replace("_norm", "").replace("_econ", "_econ")
        plot_event_study(coefs, label, FIG_DIR / f"event_study_{out_name}.png")
        all_coefs.append(coefs)

    all_df = pd.concat(all_coefs, ignore_index=True)
    tab_path = TAB_DIR / "event_study_coefficients.csv"
    all_df.to_csv(tab_path, index=False, float_format="%.6f")
    print(f"\n  Coefficients saved: {tab_path}")
    print("Done.")


if __name__ == "__main__":
    main()
