"""
plot_slant_distribution.py

Histogram of newspaper-level normalized net slant for representative years,
showing the cross-sectional distribution.

Inputs:
  - data/processed/runs/exp_shvocab_cv/panel/14_regression_panel.parquet

Outputs:
  - output/figures/slant_distribution.pdf
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
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

PANEL_PATH = BASE_DIR / "data" / "processed" / "runs" / "exp_shvocab_cv" / "panel" / "14_regression_panel.parquet"
FIG_DIR = BASE_DIR / "output" / "figures"

REP_YEARS = [1992, 1994, 1996, 1998]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH)
    panel = panel[
        panel["cz"].notna()
        & panel["vulnerability1990_scaled"].notna()
        & panel["manushare1990"].notna()
    ].copy()

    bins = np.arange(-0.8, 0.85, 0.05)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)

    for ax, yr in zip(axes.flat, REP_YEARS):
        sub = panel[panel["year"] == yr]
        x = sub["net_slant_norm"].values
        n = len(x)
        n_pos = (x > 0).sum()
        n_neg = (x < 0).sum()

        ax.hist(x[x >= 0], bins=bins, color="#ddb5b1", edgecolor="#bf6b63",
                linewidth=0.5, zorder=2, label="R-leaning")
        ax.hist(x[x < 0], bins=bins, color="#adc4d4", edgecolor="#5d8aa8",
                linewidth=0.5, zorder=2, label="D-leaning")

        ax.axvline(0, color="black", linewidth=0.7, linestyle="-", zorder=3)
        ax.axvline(x.mean(), color="#333333", linewidth=1.0, linestyle="--", zorder=3)

        pre_post = "Pre-NAFTA" if yr < 1994 else "Post-NAFTA"
        ax.set_title(f"{yr} ({pre_post})", fontsize=11, pad=5)

        ax.text(0.97, 0.95,
                f"$N$ = {n}\nMean = {x.mean():.3f}\nR: {n_pos}, D: {n_neg}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.95))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)

        print(f"  {yr}: N={n}, mean={x.mean():.3f}, R={n_pos}, D={n_neg}")

    # Shared labels
    axes[0, 0].legend(fontsize=8, framealpha=0.9, loc="upper left")
    fig.supxlabel("Normalized Net Slant ($\\tilde{S}$)", fontsize=11, y=0.02)
    fig.supylabel("Number of Newspapers", fontsize=11, x=0.02)

    fig.tight_layout(rect=[0.03, 0.04, 1, 1], h_pad=1.5, w_pad=1.0)
    out_path = FIG_DIR / "slant_distribution.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
