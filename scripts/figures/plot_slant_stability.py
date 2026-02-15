"""
plot_slant_stability.py

Validation figure: within-newspaper stability of the slant measure.
Shows that newspaper slant is a persistent characteristic, not random noise.

Two panels:
  (a) Lag-1 autocorrelation scatter: slant_t vs slant_{t-1}
  (b) Split-half scatter: odd-year avg vs even-year avg

Inputs:
  - data/processed/runs/exp_shvocab_cv/panel/14_regression_panel.parquet

Outputs:
  - output/figures/slant_stability.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

PANEL_PATH = BASE_DIR / "data" / "processed" / "runs" / "exp_shvocab_cv" / "panel" / "14_regression_panel.parquet"
FIG_DIR = BASE_DIR / "output" / "figures"


def compute_icc(df, group_col, value_col):
    """Compute ICC(1) — fraction of variance between groups."""
    groups = df.groupby(group_col)[value_col]
    k_groups = groups.ngroups
    n_total = len(df)
    grand_mean = df[value_col].mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for _, g in groups)

    ms_between = ss_between / (k_groups - 1)
    ms_within = ss_within / (n_total - k_groups)

    n_avg = n_total / k_groups
    icc = (ms_between - ms_within) / (ms_between + (n_avg - 1) * ms_within)
    return icc


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel ...")
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel[
        panel["cz"].notna()
        & panel["vulnerability1990_scaled"].notna()
        & panel["manushare1990"].notna()
    ].copy()
    print(f"  {len(panel):,} obs, {panel['paper'].nunique()} papers, "
          f"years {panel['year'].min()}-{panel['year'].max()}")

    # --- ICC ---
    icc_slant = compute_icc(panel, "paper", "net_slant_norm")
    icc_shareR = compute_icc(panel, "paper", "ext_R")
    print(f"\n  ICC(1) net_slant_norm: {icc_slant:.3f}")
    print(f"  ICC(1) ext_R:          {icc_shareR:.3f}")

    # --- Lag-1 autocorrelation (observation-level) ---
    panel_sorted = panel.sort_values(["paper", "year"])
    panel_sorted["slant_lag1"] = panel_sorted.groupby("paper")["net_slant_norm"].shift(1)
    panel_sorted["shareR_lag1"] = panel_sorted.groupby("paper")["ext_R"].shift(1)
    auto = panel_sorted.dropna(subset=["slant_lag1"])
    r_auto_slant, p_auto_slant = stats.pearsonr(auto["net_slant_norm"], auto["slant_lag1"])
    r_auto_shareR, p_auto_shareR = stats.pearsonr(auto["ext_R"], auto["shareR_lag1"])
    print(f"\n  Lag-1 autocorrelation (slant):  {r_auto_slant:.3f} (p={p_auto_slant:.4f})")
    print(f"  Lag-1 autocorrelation (shareR): {r_auto_shareR:.3f} (p={p_auto_shareR:.4f})")

    # --- Split-half: odd years vs even years ---
    odd = panel[panel["year"] % 2 == 1].groupby("paper").agg(
        slant_odd=("net_slant_norm", "mean"),
        shareR_odd=("ext_R", "mean"),
        n_odd=("year", "count"),
    ).reset_index()
    even = panel[panel["year"] % 2 == 0].groupby("paper").agg(
        slant_even=("net_slant_norm", "mean"),
        shareR_even=("ext_R", "mean"),
        n_even=("year", "count"),
    ).reset_index()
    split = odd.merge(even, on="paper")
    split = split[(split["n_odd"] >= 3) & (split["n_even"] >= 3)].copy()
    print(f"\n  Split-half papers (odd/even, >=3yr each): {len(split)}")

    r_split_slant, p_split_slant = stats.pearsonr(split["slant_odd"], split["slant_even"])
    r_split_shareR, p_split_shareR = stats.pearsonr(split["shareR_odd"], split["shareR_even"])
    print(f"  Odd vs Even slant:  r = {r_split_slant:.3f} (p = {p_split_slant:.4f})")
    print(f"  Odd vs Even shareR: r = {r_split_shareR:.3f} (p = {p_split_shareR:.4f})")

    # --- Figure: 2-panel ---
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Diverging colormap: blue (D-leaning) → red (R-leaning)
    cmap_div = plt.cm.RdBu_r
    midpoint = 0.5

    # Panel (a): Lag-1 autocorrelation scatter — share R
    ax = axes[0]
    x_a = auto["shareR_lag1"].values
    y_a = auto["ext_R"].values
    c_a = (x_a + y_a) / 2
    norm_a = TwoSlopeNorm(vmin=c_a.min(), vcenter=midpoint, vmax=c_a.max())
    ax.scatter(x_a, y_a, s=18, c=c_a, cmap=cmap_div, norm=norm_a, alpha=0.5,
               edgecolors="white", linewidth=0.3, zorder=3)

    slope_a, intercept_a, _, _, _ = stats.linregress(x_a, y_a)
    x_grid_a = np.linspace(x_a.min() - 0.01, x_a.max() + 0.01, 200)
    y_hat_a = intercept_a + slope_a * x_grid_a
    ax.plot(x_grid_a, y_hat_a, color="#c0392b", linewidth=1.8, zorder=4)

    # 45-degree line
    lims_a = [min(x_a.min(), y_a.min()) - 0.01, max(x_a.max(), y_a.max()) + 0.01]
    ax.plot(lims_a, lims_a, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)

    ax.text(0.97, 0.05,
            f"$r$ = {r_auto_shareR:.2f},  $N$ = {len(auto):,}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.95))

    ax.set_xlabel("Share R-Leaning in Year $t-1$", fontsize=11)
    ax.set_ylabel("Share R-Leaning in Year $t$", fontsize=11)
    ax.set_title("(a) Year-to-Year Persistence", fontsize=12, pad=8)
    ax.tick_params(labelsize=9)
    ax.grid(axis="both", alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel (b): Split-half scatter — odd vs even years
    ax = axes[1]
    x_b = split["shareR_odd"].values
    y_b = split["shareR_even"].values
    c_b = (x_b + y_b) / 2
    norm_b = TwoSlopeNorm(vmin=c_b.min(), vcenter=midpoint, vmax=c_b.max())
    ax.scatter(x_b, y_b, s=40, c=c_b, cmap=cmap_div, norm=norm_b, alpha=0.7,
               edgecolors="white", linewidth=0.4, zorder=3)

    slope_b, intercept_b, _, _, _ = stats.linregress(x_b, y_b)
    x_grid_b = np.linspace(x_b.min() - 0.005, x_b.max() + 0.005, 200)
    y_hat_b = intercept_b + slope_b * x_grid_b
    n_b = len(x_b)
    x_mean_b = x_b.mean()
    ss_x_b = np.sum((x_b - x_mean_b) ** 2)
    resid_var_b = np.sum((y_b - (intercept_b + slope_b * x_b)) ** 2) / (n_b - 2)
    se_pred_b = np.sqrt(resid_var_b * (1.0 / n_b + (x_grid_b - x_mean_b) ** 2 / ss_x_b))
    ax.fill_between(x_grid_b, y_hat_b - 1.96 * se_pred_b, y_hat_b + 1.96 * se_pred_b,
                     color="#c0392b", alpha=0.10, zorder=1)
    ax.plot(x_grid_b, y_hat_b, color="#c0392b", linewidth=1.8, zorder=4)

    lims_b = [min(x_b.min(), y_b.min()) - 0.01, max(x_b.max(), y_b.max()) + 0.01]
    ax.plot(lims_b, lims_b, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)

    ax.text(0.97, 0.05,
            f"$r$ = {r_split_shareR:.2f},  $N$ = {len(split)}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.95))

    ax.set_xlabel("Odd-Year Average Share R-Leaning", fontsize=11)
    ax.set_ylabel("Even-Year Average Share R-Leaning", fontsize=11)
    ax.set_title("(b) Split-Half Reliability", fontsize=12, pad=8)
    ax.tick_params(labelsize=9)
    ax.grid(axis="both", alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    out_path = FIG_DIR / "slant_stability.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("STABILITY SUMMARY")
    print(f"{'='*60}")
    print(f"  ICC(1) net_slant_norm:            {icc_slant:.3f}")
    print(f"  ICC(1) ext_R:                     {icc_shareR:.3f}")
    print(f"  Lag-1 autocorr (slant):           {r_auto_slant:.3f}")
    print(f"  Lag-1 autocorr (shareR):          {r_auto_shareR:.3f}")
    print(f"  Split-half r (slant, odd/even):   {r_split_slant:.3f}")
    print(f"  Split-half r (shareR, odd/even):  {r_split_shareR:.3f}")


if __name__ == "__main__":
    main()
