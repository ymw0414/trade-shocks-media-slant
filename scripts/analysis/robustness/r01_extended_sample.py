"""
r01_extended_sample.py

Robustness check: extend the sample from 2000 to 2004.
The baseline cuts at 2000 to avoid confounding with the China shock (PNTR 2000).
Since spec 3 already controls for china_shock × year, extending to 2004
tests whether results hold with 4 additional post-treatment years.

Outputs:
  - output/robustness/extended_sample_did.csv
  - output/robustness/extended_sample_event_study.csv
  - output/robustness/event_study_extended_{depvar}.png
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = (BASE_DIR / "data" / "processed" / "panel"
              / "14_regression_panel.parquet")
OUT_DIR = BASE_DIR / "output" / "robustness"

BASE_YEAR = 1993
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


def load_panel(end_year):
    """Load regression panel with specified end year."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()

    if end_year is not None:
        df = df[df["year"] <= end_year].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    years = sorted(df["year"].unique())
    base_yr = years[0]

    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]

    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

    return df, years


def run_did(df, depvar, years):
    """Run 3-spec DiD for one outcome."""
    base_yr = years[0]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]

    specs = [
        ("spec1", "vuln_x_post"),
        ("spec2", f"vuln_x_post + {' + '.join(manu_vars)}"),
        ("spec3", f"vuln_x_post + {' + '.join(china_vars)} + {' + '.join(manu_vars)}"),
    ]

    results = {}
    for spec_name, rhs in specs:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy().loc["vuln_x_post"]
        results[spec_name] = {
            "coef": t["Estimate"], "se": t["Std. Error"],
            "pval": t["Pr(>|t|)"], "N": m._N,
        }
    return results


def run_event_study(df, depvar, years):
    """Run controlled event study for one outcome."""
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    base_yr = years[0]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]

    rhs = " + ".join(vul_vars + china_vars + manu_vars)
    fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
    m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
    t = m.tidy()

    rows = []
    for yr in years:
        if yr == BASE_YEAR:
            rows.append({"year": yr, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
            continue
        r = t.loc[f"vul_{yr}"]
        rows.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                     "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
    return pd.DataFrame(rows)


def plot_comparison(coefs_2000, coefs_2004, depvar_label, out_path):
    """Plot event study comparing 2000 vs 2004 cutoffs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    offset = 0.12

    yrs_2000 = coefs_2000["year"].values
    ax.errorbar(yrs_2000 - offset, coefs_2000["coef"],
                yerr=[coefs_2000["coef"] - coefs_2000["ci_lo"],
                      coefs_2000["ci_hi"] - coefs_2000["coef"]],
                fmt="o", color="#2d2d2d", markersize=5, capsize=3,
                linewidth=1.2, label="Baseline (1987-2000)")

    yrs_2004 = coefs_2004["year"].values
    ax.errorbar(yrs_2004 + offset, coefs_2004["coef"],
                yerr=[coefs_2004["coef"] - coefs_2004["ci_lo"],
                      coefs_2004["ci_hi"] - coefs_2004["coef"]],
                fmt="s", color="#7a7a7a", markersize=5, capsize=3,
                linewidth=1.2, label="Extended (1987-2004)")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title(f"Event Study: {depvar_label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks(yrs_2004)
    ax.set_xticklabels(yrs_2004, rotation=45, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = [
        ("net_slant_norm",           "Net Slant"),
        ("right_norm",               "Right Intensity"),
        ("left_norm",                "Left Intensity"),
        ("politicization_norm",      "Politicization"),
        ("econ_share",               "Econ Article Share"),
    ]

    # --- DiD comparison ---
    print("=" * 70)
    print("DiD: Baseline (2000) vs Extended (2004)")
    print("=" * 70)

    all_did = []
    for end_year in [2000, 2004]:
        df, years = load_panel(end_year)
        print(f"\n  END_YEAR={end_year}: {len(df):,} obs, "
              f"{df['paper_id'].nunique()} papers, {df['cz'].nunique()} CZs")

        for depvar, label in outcomes:
            res = run_did(df, depvar, years)
            for spec, vals in res.items():
                all_did.append({"end_year": end_year, "depvar": depvar,
                                "label": label, "spec": spec, **vals})

    did_df = pd.DataFrame(all_did)
    did_path = OUT_DIR / "extended_sample_did.csv"
    did_df.to_csv(did_path, index=False, float_format="%.6f")

    # Print comparison
    print(f"\n{'='*90}")
    print(f"{'Outcome':<22} {'2000 Spec3':>16} {'2004 Spec3':>16} {'Change':>10}")
    print(f"{'-'*64}")
    for depvar, label in outcomes:
        r2000 = did_df[(did_df["depvar"]==depvar) & (did_df["spec"]=="spec3")
                       & (did_df["end_year"]==2000)].iloc[0]
        r2004 = did_df[(did_df["depvar"]==depvar) & (did_df["spec"]=="spec3")
                       & (did_df["end_year"]==2004)].iloc[0]
        s2000 = "***" if r2000.pval<0.01 else "**" if r2000.pval<0.05 else "*" if r2000.pval<0.1 else ""
        s2004 = "***" if r2004.pval<0.01 else "**" if r2004.pval<0.05 else "*" if r2004.pval<0.1 else ""
        pct = (r2004.coef - r2000.coef) / abs(r2000.coef) * 100 if r2000.coef != 0 else 0
        print(f"{label:<22} {r2000.coef:7.4f}{s2000:<3} ({r2000.se:.4f})  "
              f"{r2004.coef:7.4f}{s2004:<3} ({r2004.se:.4f})  {pct:+6.1f}%")
    print(f"\nN: 2000={did_df[did_df.end_year==2000].N.iloc[0]}, "
          f"2004={did_df[did_df.end_year==2004].N.iloc[0]}")

    # --- Event study comparison ---
    print(f"\n{'='*70}")
    print("Event Study: Baseline vs Extended (controlled spec)")
    print("=" * 70)

    es_rows = []
    es_results = {}

    for end_year in [2000, 2004]:
        df, years = load_panel(end_year)
        es_results[end_year] = {}
        for depvar, label in outcomes:
            coefs = run_event_study(df, depvar, years)
            coefs["end_year"] = end_year
            coefs["depvar"] = depvar
            es_rows.append(coefs)
            es_results[end_year][depvar] = coefs
            print(f"  {end_year} {label}: done")

    es_df = pd.concat(es_rows, ignore_index=True)
    es_path = OUT_DIR / "extended_sample_event_study.csv"
    es_df.to_csv(es_path, index=False, float_format="%.6f")

    # Plot overlaid comparisons
    for depvar, label in outcomes:
        out_name = depvar.replace("_norm", "")
        plot_comparison(
            es_results[2000][depvar],
            es_results[2004][depvar],
            f"{label}: 2000 vs 2004",
            OUT_DIR / f"event_study_extended_{out_name}.png",
        )

    print(f"\n  DiD results: {did_path}")
    print(f"  Event study results: {es_path}")
    print("Done.")


if __name__ == "__main__":
    main()
