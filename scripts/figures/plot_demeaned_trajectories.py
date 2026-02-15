"""
plot_demeaned_trajectories.py

Identifying variation in the event study: group-mean divergence
between high- and low-vulnerability newspapers after NAFTA.

Panel (a): Paper-FE-demeaned group means (shows common trend + divergence)
Panel (b): Year-by-year gap (High - Low) with 95% CI (isolates the divergence)

Inputs:
  - data/processed/runs/{RUN}/panel/14_regression_panel.parquet

Outputs:
  - output/figures/demeaned_trajectories.pdf/.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
sys.path.insert(0, str(BASE_DIR / "scripts" / "nlp"))
import pipeline_config as cfg

PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = BASE_DIR / "output" / "figures"

NAFTA_YEAR = 1994

# Colors
C_HIGH = "#b30000"
C_LOW  = "#4575b4"
C_GAP  = "#333333"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()

    # Median vulnerability split
    vuln_med = df.groupby("paper")["vulnerability1990_scaled"].first()
    med = vuln_med.median()
    high_papers = set(vuln_med[vuln_med >= med].index)
    low_papers = set(vuln_med[vuln_med < med].index)
    df["vuln_group"] = df["paper"].apply(
        lambda p: "High" if p in high_papers else "Low"
    )
    n_high = len(high_papers)
    n_low = len(low_papers)

    outcomes = [
        ("net_slant_norm", "Net Slant (normalized)", "Share R-Leaning"),
        ("ext_R",          "Share R-Leaning",         "Share R-Leaning"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    for col_idx, (yvar, ylabel, _) in enumerate(outcomes):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # --- Top: paper-FE-demeaned group means ---
        paper_means = df.groupby("paper")[yvar].transform("mean")
        df[f"dm_{yvar}"] = df[yvar] - paper_means
        dm_col = f"dm_{yvar}"

        grp_stats = df.groupby(["year", "vuln_group"])[dm_col].agg(
            ["mean", "sem"]
        ).reset_index()
        grp_stats.columns = ["year", "vuln_group", "mean", "sem"]

        for grp, color, label in [
            ("High", C_HIGH, f"High vulnerability ($N = {n_high}$)"),
            ("Low",  C_LOW,  f"Low vulnerability ($N = {n_low}$)"),
        ]:
            g = grp_stats[grp_stats["vuln_group"] == grp].sort_values("year")
            ax_top.plot(g["year"], g["mean"], color=color, linewidth=1.8,
                        label=label, zorder=3)
            ax_top.fill_between(g["year"],
                                g["mean"] - 1.96 * g["sem"],
                                g["mean"] + 1.96 * g["sem"],
                                color=color, alpha=0.12, zorder=2)

        ax_top.axvline(NAFTA_YEAR - 0.5, color="gray", linewidth=0.7,
                       linestyle="--", zorder=1)
        ax_top.axhline(0, color="gray", linewidth=0.4, zorder=0)
        ax_top.set_ylabel(ylabel)
        panel_letter = "(a)" if col_idx == 0 else "(b)"
        ax_top.set_title(f"{panel_letter} Group means (paper FE removed)",
                         fontsize=10, loc="left")

        if col_idx == 0:
            ax_top.legend(fontsize=7.5, loc="upper left", frameon=True,
                          framealpha=0.9, edgecolor="#cccccc")

        # --- Bottom: year-by-year gap (High - Low) ---
        years = sorted(df["year"].unique())
        gaps = []
        for yr in years:
            h = df[(df["year"] == yr) & (df["vuln_group"] == "High")][dm_col]
            l = df[(df["year"] == yr) & (df["vuln_group"] == "Low")][dm_col]
            diff = h.mean() - l.mean()
            # SE of difference in means
            se_diff = np.sqrt(h.var() / len(h) + l.var() / len(l))
            gaps.append({"year": yr, "gap": diff, "se": se_diff})

        gap_df = pd.DataFrame(gaps)

        ax_bot.plot(gap_df["year"], gap_df["gap"], color=C_GAP,
                    linewidth=2.0, marker="o", markersize=4, zorder=3)
        ax_bot.fill_between(gap_df["year"],
                            gap_df["gap"] - 1.96 * gap_df["se"],
                            gap_df["gap"] + 1.96 * gap_df["se"],
                            color=C_GAP, alpha=0.12, zorder=2)
        ax_bot.axvline(NAFTA_YEAR - 0.5, color="gray", linewidth=0.7,
                       linestyle="--", zorder=1)
        ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle="-",
                       zorder=0)

        ax_bot.set_xlabel("Year")
        ax_bot.set_ylabel(f"Gap (High $-$ Low)")
        panel_letter2 = "(c)" if col_idx == 0 else "(d)"
        ax_bot.set_title(f"{panel_letter2} High $-$ Low gap",
                         fontsize=10, loc="left")

        for ax in [ax_top, ax_bot]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(width=0.6)

    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"demeaned_trajectories.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white")
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
