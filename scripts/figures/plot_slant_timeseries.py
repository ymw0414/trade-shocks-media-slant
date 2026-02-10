"""
plot_slant_timeseries.py

Figure 4: Media Slant Over Time — High vs. Low Trade Exposure.
2x2 panel showing yearly means for newspapers above/below median vulnerability.

Inputs:
  - data/processed/panel/14_regression_panel.parquet

Outputs:
  - output/figures/slant_timeseries.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel" / "14_regression_panel.parquet"
FIG_DIR = BASE_DIR / "output" / "figures"

NAFTA_YEAR = 1993  # last pre-NAFTA year


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()

    # Split at median vulnerability
    med_vul = df.groupby("paper")["vulnerability1990_scaled"].first().median()
    paper_vul = df.groupby("paper")["vulnerability1990_scaled"].first()
    high_papers = paper_vul[paper_vul >= med_vul].index
    df["exposure"] = np.where(df["paper"].isin(high_papers), "High exposure", "Low exposure")

    panels = [
        ("net_slant_norm",           r"Net Slant ($\tilde{S}$)"),
        ("politicization_norm",      r"Politicization ($\tilde{P}$)"),
        ("net_slant_norm_econ",      r"Net Slant, Econ ($\tilde{S}^{\mathrm{econ}}$)"),
        ("econ_share",               "Economic Article Share"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for ax, (var, title) in zip(axes.flat, panels):
        for label, color, marker in [("High exposure", "#cb181d", "s"),
                                      ("Low exposure", "#2171b5", "o")]:
            sub = df[df["exposure"] == label]
            means = sub.groupby("year")[var].mean()
            ax.plot(means.index, means.values, marker=marker, markersize=4,
                    linewidth=1.2, color=color, label=label)

        ax.axvline(NAFTA_YEAR + 0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=9)

        yrs = sorted(df["year"].unique())
        ax.set_xticks([yr for yr in yrs if yr % 3 == 0 or yr == yrs[0]])
        ax.set_xticklabels([str(yr) for yr in yrs if yr % 3 == 0 or yr == yrs[0]],
                           fontsize=9)
        ax.set_xlim(yrs[0] - 0.5, yrs[-1] + 0.5)
        ax.grid(axis="y", alpha=0.2)

    # Legend on first panel only
    axes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    # Add NAFTA label
    axes[0, 0].text(NAFTA_YEAR + 0.7, axes[0, 0].get_ylim()[1] * 0.95,
                    "NAFTA", fontsize=8, color="gray", va="top")

    # Shared x-label
    fig.text(0.5, 0.01, "Year", ha="center", fontsize=11)

    fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.0)
    fig.subplots_adjust(bottom=0.07)

    out_path = FIG_DIR / "slant_timeseries.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
