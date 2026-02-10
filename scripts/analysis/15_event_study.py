"""
14_event_study.py

Event study of NAFTA tariff vulnerability on newspaper slant,
following the Choi et al. (2024) specification adapted for newspaper-level data.

Two specifications (both with paper FE + year FE + division×year FE, cluster CZ):
  Spec 1 (baseline): vulnerability × year only
  Spec 2 (controls): + china_shock × year + manushare × year

Inputs:
  - data/processed/panel/minwoo/13_regression_panel.parquet

Outputs:
  - output/figures/event_study_{depvar}.png   (one per outcome)
  - output/figures/event_study_intensity.png  (right vs left combined)
  - output/figures/event_study_intensity_econ.png
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

BASE_YEAR = 1993  # Last pre-NAFTA year (following Choi et al. 2024)
END_YEAR = 2000   # Exclude China shock period (PNTR 2000, WTO 2001)

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
    """Load regression panel and create event-study interaction variables."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()

    if END_YEAR is not None:
        df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes

    years = sorted(df["year"].unique())
    base_yr = years[0]

    # Vulnerability × year (omit base year)
    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]

    # China shock × year (omit base year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    # Manushare × year (omit base year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

    return df, years


def run_event_study(df, depvar, years, label):
    """Run event-study: baseline and with controls."""
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    vul_rhs = " + ".join(vul_vars)

    base_yr = years[0]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    control_rhs = " + ".join(china_vars + manu_vars)

    print(f"\n  [{label}] {depvar}")

    results = {}
    for spec_name, rhs in [("baseline", vul_rhs),
                            ("controls", f"{vul_rhs} + {control_rhs}")]:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        print(f"    {spec_name}: N={m._N}, R2={m._r2:.4f}")

        rows = []
        t = m.tidy()
        for yr in years:
            if yr == BASE_YEAR:
                rows.append({"year": yr, "coef": 0, "se": 0,
                             "ci_lo": 0, "ci_hi": 0})
                continue
            r = t.loc[f"vul_{yr}"]
            rows.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                         "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
        results[spec_name] = pd.DataFrame(rows)

    return results


def plot_event_study_dual(coefs_base, coefs_ctrl, depvar_label, out_path):
    """Plot event-study with baseline and controlled specifications."""
    fig, ax = plt.subplots(figsize=(10, 6))

    yrs = coefs_base["year"].values
    offset = 0.12

    # Baseline
    ax.errorbar(yrs - offset, coefs_base["coef"],
                yerr=[coefs_base["coef"] - coefs_base["ci_lo"],
                      coefs_base["ci_hi"] - coefs_base["coef"]],
                fmt="o", color="#cb181d", markersize=5, capsize=3,
                linewidth=1.2, label="Baseline")

    # With controls
    ax.errorbar(yrs + offset, coefs_ctrl["coef"],
                yerr=[coefs_ctrl["coef"] - coefs_ctrl["ci_lo"],
                      coefs_ctrl["ci_hi"] - coefs_ctrl["coef"]],
                fmt="s", color="#2171b5", markersize=5, capsize=3,
                linewidth=1.2, label="+ China shock, manushare")

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


def plot_combined_intensity(coefs_right, coefs_left, title, out_path):
    """Plot right and left intensity on the same axes (controlled spec)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    yrs = coefs_right["year"].values
    offset = 0.15

    ax.errorbar(yrs - offset, coefs_right["coef"],
                yerr=[coefs_right["coef"] - coefs_right["ci_lo"],
                      coefs_right["ci_hi"] - coefs_right["coef"]],
                fmt="o", color="#cb181d", markersize=5, capsize=3,
                linewidth=1.2, label="Right Intensity")

    ax.errorbar(yrs + offset, coefs_left["coef"],
                yerr=[coefs_left["coef"] - coefs_left["ci_lo"],
                      coefs_left["ci_hi"] - coefs_left["coef"]],
                fmt="s", color="#2171b5", markersize=5, capsize=3,
                linewidth=1.2, label="Left Intensity")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7, label="NAFTA (1994)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title(f"Event Study: {title}", fontsize=14, fontweight="bold")
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
        ("right_norm",               "Right Intensity (Normalized)"),
        ("left_norm",                "Left Intensity (Normalized)"),
        ("net_slant_norm",           "Net Slant (Normalized)"),
        ("politicization_norm",      "Politicization (Normalized)"),
        ("right_norm_econ",          "Right Intensity \u2014 Econ Articles (Normalized)"),
        ("left_norm_econ",           "Left Intensity \u2014 Econ Articles (Normalized)"),
        ("net_slant_norm_econ",      "Net Slant \u2014 Econ Articles (Normalized)"),
        ("politicization_norm_econ", "Politicization \u2014 Econ Articles (Normalized)"),
        ("econ_share",               "Economic Article Share"),
    ]

    all_results = {}  # depvar -> {baseline: df, controls: df}
    all_rows = []
    for depvar, label in outcomes:
        res = run_event_study(df, depvar, years, label)
        all_results[depvar] = res
        for spec_name, coefs in res.items():
            c = coefs.copy()
            c["depvar"] = depvar
            c["spec"] = spec_name
            all_rows.append(c)

    # --- Individual plots with baseline vs controls ---
    individual = [
        ("net_slant_norm",           "Net Slant (Normalized)"),
        ("politicization_norm",      "Politicization (Normalized)"),
        ("net_slant_norm_econ",      "Net Slant \u2014 Econ Articles (Normalized)"),
        ("politicization_norm_econ", "Politicization \u2014 Econ Articles (Normalized)"),
        ("econ_share",               "Economic Article Share"),
    ]
    for depvar, label in individual:
        out_name = depvar.replace("_norm", "").replace("_econ", "_econ")
        plot_event_study_dual(
            all_results[depvar]["baseline"],
            all_results[depvar]["controls"],
            label,
            FIG_DIR / f"event_study_{out_name}.png",
        )

    # --- Combined right vs left plots (controlled spec) ---
    plot_combined_intensity(
        all_results["right_norm"]["controls"],
        all_results["left_norm"]["controls"],
        "Right vs. Left Intensity (Normalized, with controls)",
        FIG_DIR / "event_study_intensity.png",
    )
    plot_combined_intensity(
        all_results["right_norm_econ"]["controls"],
        all_results["left_norm_econ"]["controls"],
        "Right vs. Left Intensity \u2014 Econ Articles (Normalized, with controls)",
        FIG_DIR / "event_study_intensity_econ.png",
    )

    # --- Save coefficients ---
    all_df = pd.concat(all_rows, ignore_index=True)
    tab_path = TAB_DIR / "event_study_coefficients.csv"
    all_df.to_csv(tab_path, index=False, float_format="%.6f")
    print(f"\n  Coefficients saved: {tab_path}")
    print("Done.")


if __name__ == "__main__":
    main()
