"""
plot_article_distribution.py

Within-newspaper distribution of article-level normalized net slant
for representative newspapers in 1994, illustrating that newspaper-level
averages mask substantial within-newspaper variation.

Inputs:
  - data/processed/newspapers/07_newspaper_meta_cong_103.parquet
  - data/processed/runs/exp_shvocab_cv/newspapers/09_article_slant_norm_cong_103.parquet

Outputs:
  - output/figures/article_slant_distribution.pdf
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

META_PATH = BASE_DIR / "data" / "processed" / "newspapers" / "07_newspaper_meta_cong_103.parquet"
SLANT_PATH = BASE_DIR / "data" / "processed" / "runs" / "exp_shvocab_cv" / "newspapers" / "09_article_slant_norm_cong_103.parquet"
FIG_DIR = BASE_DIR / "output" / "figures"

# Representative newspapers (well-known, spanning the slant spectrum)
PAPERS = [
    "Oregonian, The (Portland, OR)",     # D-leaning (mean ~ -0.10)
    "Houston Chronicle",                 # near center (mean ~ 0.02)
    "The Washington Times",              # R-leaning (mean ~ 0.21)
]

YEAR = 1994


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(META_PATH, columns=["paper", "year"])
    slant = pd.read_parquet(SLANT_PATH, columns=["net_slant_norm"])

    df = meta.copy()
    df["net_slant_norm"] = slant["net_slant_norm"].values
    df = df[df["year"] == YEAR]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    bins = np.arange(-4, 4.1, 0.15)

    for ax, paper in zip(axes, PAPERS):
        sub = df[df["paper"] == paper]
        x = sub["net_slant_norm"].values
        n = len(x)
        mu = x.mean()

        # Histogram
        ax.hist(x[x >= 0], bins=bins, color="#ddb5b1", edgecolor="#bf6b63",
                linewidth=0.4, zorder=2, alpha=0.85, label="R-coded")
        ax.hist(x[x < 0], bins=bins, color="#adc4d4", edgecolor="#5d8aa8",
                linewidth=0.4, zorder=2, alpha=0.85, label="D-coded")

        ax.axvline(0, color="black", linewidth=0.6, linestyle="-", zorder=3)
        ax.axvline(mu, color="#333333", linewidth=1.2, linestyle="--", zorder=3,
                   label=f"Mean = {mu:.3f}")

        # Clean name for title
        name = paper.replace("Oregonian, The (Portland, OR)", "The Oregonian (Portland)")
        name = name.replace("The Washington Times", "Washington Times")
        if name == paper:
            name = paper.split(" (")[0]  # remove state abbreviation
        ax.set_title(name, fontsize=10, pad=5)

        ax.text(0.97, 0.95,
                f"$N$ = {n:,}\nMean = {mu:.3f}\nSD = {x.std():.3f}",
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.95))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.15, linewidth=0.5)
        ax.set_xlim(-4, 4)

        print(f"  {name}: N={n:,}, mean={mu:.3f}, sd={x.std():.3f}")

    axes[0].legend(fontsize=7, framealpha=0.9, loc="upper left")
    fig.supxlabel("Normalized Net Slant ($\\tilde{S}$)", fontsize=10, y=0.01)
    fig.supylabel("Number of Articles", fontsize=10, x=0.02)

    fig.tight_layout(rect=[0.03, 0.05, 1, 1], w_pad=1.5)
    out_path = FIG_DIR / "article_slant_distribution.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
