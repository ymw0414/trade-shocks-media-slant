"""
plot_slant_vs_voteshare.py

External validation of the newspaper slant measure:
binned scatterplot of newspaper average slant vs. local Republican
presidential vote share.

Approach:
  1. County-level direct match (newspaper's own county FIPS)
  2. Multiple election years (1988, 1992, 1996, 2000) averaged
  3. Full sample period newspaper slant

Inputs:
  - data/processed/runs/exp_shvocab_cv/panel/14_regression_panel.parquet
  - replication/.../pres_elections_release.dta (county-level election data)

Outputs:
  - output/figures/slant_vs_voteshare.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
from scipy import stats
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

PANEL_PATH = BASE_DIR / "data" / "processed" / "runs" / "exp_shvocab_cv" / "panel" / "14_regression_panel.parquet"
ELECTION_PATH = (BASE_DIR / "replication" / "Replication Project" / "Replication Project" /
                 "data" / "raw_data" / "election" / "dataverse_files" /
                 "County_Level_US_Elections_Data" / "pres_elections_release.dta")
FIG_DIR = BASE_DIR / "output" / "figures"

ELECTION_YEARS_PRE = [1988, 1992]  # Pre-NAFTA only (exogenous)
ELECTION_YEARS_ALL = [1988, 1992, 1996, 2000]


def load_election_data(years):
    """Load county-level presidential election data for given years."""
    df = pd.read_stata(ELECTION_PATH)
    df = df[df["election_year"].isin(years)].copy()

    df["fips"] = pd.to_numeric(df["fips"], errors="coerce")
    df = df.dropna(subset=["fips"])
    df["fips"] = df["fips"].astype(int)

    df["rep_votes"] = pd.to_numeric(df["republican_raw_votes"], errors="coerce")
    df["dem_votes"] = pd.to_numeric(df["democratic_raw_votes"], errors="coerce")
    df["total_2p"] = df["rep_votes"] + df["dem_votes"]
    df = df[df["total_2p"] > 0].copy()
    df["rep_share"] = df["rep_votes"] / df["total_2p"]

    # Vote-weighted average across election years per county
    avg = (df.groupby("fips")
           .agg(rep_votes=("rep_votes", "sum"),
                total_2p=("total_2p", "sum"))
           .reset_index())
    avg["rep_share"] = avg["rep_votes"] / avg["total_2p"]

    print(f"  Election data ({years}): {len(avg):,} counties")
    return avg[["fips", "rep_share"]]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load regression panel
    print("Loading regression panel ...")
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel[
        panel["cz"].notna()
        & panel["vulnerability1990_scaled"].notna()
        & panel["manushare1990"].notna()
    ].copy()
    panel["fips"] = panel["fips"].astype(int)
    print(f"  Panel: {len(panel):,} obs, {panel['paper'].nunique()} papers, "
          f"{panel['cz'].nunique()} CZs")

    # 2. Newspaper-level average slant (full sample)
    paper_avg = (panel.groupby(["paper", "fips"])
                 .agg(avg_slant=("net_slant_norm", "mean"),
                      avg_share_R=("ext_R", "mean"))
                 .reset_index())

    # 3. Compare pre-NAFTA only vs all elections
    print("\n" + "=" * 60)
    for label, years in [("Pre-NAFTA (1988+1992)", ELECTION_YEARS_PRE),
                         ("All (1988-2000)", ELECTION_YEARS_ALL)]:
        election = load_election_data(years)
        merged = paper_avg.merge(election, on="fips", how="inner")
        print(f"\n  --- {label}, county-level, N={len(merged)} ---")
        for yvar, ylabel in [("avg_slant", "Net Slant"), ("avg_share_R", "Share R")]:
            r, p = stats.pearsonr(merged["rep_share"], merged[yvar])
            print(f"    {ylabel:>12} vs RepShare: r = {r:+.3f} (p = {p:.3f})")

    # 4. Use pre-NAFTA elections for the figure (exogenous)
    print("\n" + "=" * 60)
    print("Figure: Pre-NAFTA elections (1988 + 1992), county-level match")
    election_pre = load_election_data(ELECTION_YEARS_PRE)
    merged = paper_avg.merge(election_pre, on="fips", how="inner")

    xvar = "rep_share"
    # Pick y variable with stronger correlation
    r_slant, _ = stats.pearsonr(merged[xvar], merged["avg_slant"])
    r_shareR, _ = stats.pearsonr(merged[xvar], merged["avg_share_R"])
    if abs(r_slant) > abs(r_shareR):
        yvar, ylabel = "avg_slant", "Average Normalized Net Slant"
    else:
        yvar, ylabel = "avg_share_R", "Average Share of R-Leaning Articles"

    r_pearson, p_pearson = stats.pearsonr(merged[xvar], merged[yvar])
    slope, intercept, r_val, p_val, se = stats.linregress(merged[xvar], merged[yvar])
    print(f"  {ylabel}: r = {r_pearson:+.3f} (p = {p_pearson:.4f}), slope = {slope:.4f}")

    # 5. Plot
    x = merged[xvar].values
    y = merged[yvar].values

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Scatter: solid dots with white edge for definition
    ax.scatter(x, y, s=40, color="#666666", alpha=0.55,
               edgecolors="white", linewidth=0.4, zorder=3)

    # Regression line + 95% CI band
    x_grid = np.linspace(x.min() - 0.02, x.max() + 0.02, 200)
    y_hat = intercept + slope * x_grid
    # SE of prediction at each x
    n = len(x)
    x_mean = x.mean()
    ss_x = np.sum((x - x_mean) ** 2)
    resid_var = np.sum((y - (intercept + slope * x)) ** 2) / (n - 2)
    se_pred = np.sqrt(resid_var * (1.0 / n + (x_grid - x_mean) ** 2 / ss_x))
    t_crit = 1.96

    C_R = "#bf6b63"
    ax.fill_between(x_grid, y_hat - t_crit * se_pred, y_hat + t_crit * se_pred,
                     color=C_R, alpha=0.12, zorder=1)
    ax.plot(x_grid, y_hat, color=C_R, linewidth=1.8, zorder=4)

    # Annotation (bottom-right for cleaner look)
    ax.text(0.97, 0.05,
            f"$\\beta$ = {slope:.3f} (SE = {se:.3f})\n"
            f"$r$ = {r_pearson:.2f},  $N$ = {len(merged)}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.95))

    ax.set_xlabel("Republican Vote Share (1988--1992)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(axis="both", alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)
    out_path = FIG_DIR / "slant_vs_voteshare.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
