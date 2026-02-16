"""
Generate event study figures for each experiment run.
For each run, produces 4 plots:
  1. Net Slant (raw) event study
  2. Net Slant (normalized) event study
  3. Right Intensity (normalized) event study
  4. Left Intensity (normalized) event study

Reads each experiment's regression panel, runs pyfixest event studies,
saves figures to reports/figures/{exp_name}_*.png,
and exports event study coefficients for the report.
"""

import os
import json
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
RUNS = BASE / "data" / "processed" / "runs"
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_YEAR = 1993

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
    ("net_slant",      "Net Slant (Raw)"),
    ("net_slant_norm", "Net Slant (Normalized)"),
    ("right_norm",     "Right Intensity (Normalized)"),
    ("left_norm",      "Left Intensity (Normalized)"),
]


def get_panel_path(run_name):
    if run_name == "main":
        return BASE / "data" / "processed" / "panel" / "14_regression_panel.parquet"
    else:
        return RUNS / run_name / "panel" / "14_regression_panel.parquet"


def load_and_prep(panel_path):
    df = pd.read_parquet(panel_path)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= 2004].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes

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


def run_one_es(df, depvar, years):
    """Run event study (controls spec only) and return DataFrame of coefficients."""
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    base_yr = years[0]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]

    rhs = " + ".join(vul_vars + china_vars + manu_vars)
    fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
    m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})

    rows = []
    t = m.tidy()
    for yr in years:
        if yr == BASE_YEAR:
            rows.append({"year": yr, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
            continue
        vname = f"vul_{yr}"
        if vname in t.index:
            r = t.loc[vname]
            rows.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                         "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
        else:
            rows.append({"year": yr, "coef": np.nan, "se": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan})
    return pd.DataFrame(rows)


def plot_es(coefs, title, out_path, color="#2d2d2d"):
    """Single event-study plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    yrs = coefs["year"].values

    ax.errorbar(yrs, coefs["coef"],
                yerr=[coefs["coef"] - coefs["ci_lo"],
                      coefs["ci_hi"] - coefs["coef"]],
                fmt="o", color=color, markersize=5, capsize=3,
                linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Coefficient", fontsize=11)
    ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_4panel(all_coefs, run_name, desc):
    """Generate 2x2 panel figure with all 4 event studies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#2d2d2d", "#7a7a7a", "#2d2d2d", "#7a7a7a"]
    titles = ["Net Slant (Raw)", "Net Slant (Normalized)",
              "Right Intensity (Norm)", "Left Intensity (Norm)"]

    for idx, (depvar, label) in enumerate(OUTCOMES):
        ax = axes[idx // 2][idx % 2]
        coefs = all_coefs[depvar]
        yrs = coefs["year"].values
        c = colors[idx]

        ax.errorbar(yrs, coefs["coef"],
                    yerr=[coefs["coef"] - coefs["ci_lo"],
                          coefs["ci_hi"] - coefs["coef"]],
                    fmt="o", color=c, markersize=4, capsize=2.5,
                    linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.5)
        ax.set_title(titles[idx], fontsize=11, fontweight="bold")
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Coefficient", fontsize=9)
        ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
        ax.tick_params(labelsize=8)
        ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{run_name}: {desc}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / f"{run_name}_4panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# Experiment descriptions
EXP_DESC = {
    "main": "Baseline (Widmer)",
    "exp_win2": "2-congress windows",
    "exp_win3": "3-congress windows",
    "exp_core": "Partisan core only",
    "exp_cv": "CV lambda",
    "exp_core_win2": "Core + 2-window",
    "exp_core_win3": "Core + 3-window",
    "exp_cv_win2": "CV + 2-window",
    "exp_cv_win3": "CV + 3-window",
    "exp_core_cv_win3": "Core + CV + 3-window",
    "exp_win2_sign": "2-win + sign filter",
    "exp_win2_shared": "2-win + shared vocab",
    "exp_win2_sign_shared": "2-win + sign + shared",
    "exp_clean_vocab": "Shared vocab + sign",
    "exp_proba": "Prob direct norm",
}


if __name__ == "__main__":
    exp_names = sorted([d.name for d in RUNS.iterdir() if d.is_dir()])
    if "main" in exp_names:
        exp_names.remove("main")
        exp_names.insert(0, "main")

    print(f"Generating event study figures for {len(exp_names)} runs...")

    for name in exp_names:
        panel_path = get_panel_path(name)
        if not panel_path.exists():
            print(f"  {name}: SKIP (no panel)")
            continue

        print(f"  {name}...", end="", flush=True)
        df, years = load_and_prep(panel_path)

        all_coefs = {}
        for depvar, label in OUTCOMES:
            if depvar not in df.columns:
                print(f" [{depvar} missing]", end="")
                continue
            coefs = run_one_es(df, depvar, years)
            all_coefs[depvar] = coefs

            # Individual plot
            plot_es(coefs, f"{name}: {label}",
                    FIG_DIR / f"{name}_{depvar}.png",
                    color="#bf6b63" if "right" in depvar or "raw" in label.lower() else "#5d8aa8")

        # 4-panel combined figure
        if len(all_coefs) == 4:
            plot_4panel(all_coefs, name, EXP_DESC.get(name, name))

        # Save coefficients CSV
        rows = []
        for depvar, coefs in all_coefs.items():
            c = coefs.copy()
            c["depvar"] = depvar
            rows.append(c)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(
                FIG_DIR / f"{name}_es_coefs.csv", index=False, float_format="%.6f")

        print(" OK")

    print(f"\nFigures saved to {FIG_DIR}")
