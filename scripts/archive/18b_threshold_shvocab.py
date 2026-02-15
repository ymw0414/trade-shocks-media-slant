"""
18b_threshold_shvocab.py

Probability-threshold robustness for exp_shvocab_cv.
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
RUN = BASE / "data" / "processed" / "runs" / "exp_shvocab_cv"
META_DIR = BASE / "data" / "processed" / "newspapers"
FIG_DIR = RUN / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
BASE_YEAR = 1993

STATE_TO_DIVISION = {
    9:1,23:1,25:1,33:1,44:1,50:1, 34:2,36:2,42:2,
    17:3,18:3,26:3,39:3,55:3, 19:4,20:4,27:4,29:4,31:4,38:4,46:4,
    10:5,11:5,12:5,13:5,24:5,37:5,45:5,51:5,54:5,
    1:6,21:6,28:6,47:6, 5:7,22:7,40:7,48:7,
    4:8,8:8,16:8,30:8,32:8,35:8,49:8,56:8, 2:9,6:9,15:9,41:9,53:9,
}


def main():
    # 1. Aggregate by threshold
    print("Aggregating articles by probability threshold...")
    rows = []
    for cong in range(100, 109):
        slant = pd.read_parquet(
            RUN / "newspapers" / f"08_article_slant_cong_{cong}.parquet",
            columns=["prob_R"])
        meta = pd.read_parquet(
            META_DIR / f"04_newspaper_labeled_cong_{cong}.parquet",
            columns=["paper", "year"])
        df = pd.concat([meta, slant], axis=1)
        for (paper, year), g in df.groupby(["paper", "year"]):
            n = len(g)
            prob = g["prob_R"].values
            row = {"paper": paper, "year": year, "n_articles": n}
            for t in THRESHOLDS:
                n_R = int((prob > t).sum())
                n_D = int((prob < (1 - t)).sum())
                row[f"share_R_{t}"] = n_R / n
                row[f"share_D_{t}"] = n_D / n
                row[f"r_ratio_{t}"] = n_R / (n_R + n_D) if (n_R + n_D) > 0 else np.nan
            rows.append(row)
        print(f"  Congress {cong} done ({len(df):,} articles)")
    agg = pd.DataFrame(rows)

    # 2. Build panel
    panel = pd.read_parquet(RUN / "panel" / "14_regression_panel.parquet")
    panel = panel[panel["cz"].notna() & panel["vulnerability1990_scaled"].notna()].copy()
    panel = panel[panel["year"] <= 2004].copy()
    merged = panel.merge(agg, on=["paper", "year"], how="left")
    merged["state_fips"] = (merged["fips"] // 1000).astype(int)
    merged["division"] = merged["state_fips"].map(STATE_TO_DIVISION)
    merged["paper_id"] = merged["paper"].astype("category").cat.codes
    years = sorted(merged["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == BASE_YEAR:
            continue
        merged[f"vul_{yr}"] = (merged["year"] == yr).astype(float) * merged["vulnerability1990_scaled"]
    for yr in years:
        if yr == base_yr:
            continue
        merged[f"china_{yr}"] = (merged["year"] == yr).astype(float) * merged["china_shock"].fillna(0)
        merged[f"manu_{yr}"] = (merged["year"] == yr).astype(float) * merged["manushare1990"].fillna(0)

    print(f"\nRegression panel: {len(merged):,} obs")

    # 3. Run event studies
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    rhs = " + ".join(vul_vars + china_vars + manu_vars)

    all_results = {}
    for t in THRESHOLDS:
        for depvar in [f"share_R_{t}", f"share_D_{t}", f"r_ratio_{t}"]:
            if merged[depvar].isna().all():
                print(f"  {depvar}: all NaN, skipping")
                continue
            fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
            m = pf.feols(fml, data=merged, vcov={"CRV1": "cz"})
            tidy = m.tidy()
            coefs = []
            for yr in years:
                if yr == BASE_YEAR:
                    coefs.append({"year": yr, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
                    continue
                vname = f"vul_{yr}"
                if vname in tidy.index:
                    r = tidy.loc[vname]
                    coefs.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                                  "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
                else:
                    coefs.append({"year": yr, "coef": np.nan, "se": np.nan,
                                  "ci_lo": np.nan, "ci_hi": np.nan})
            all_results[(t, depvar)] = pd.DataFrame(coefs)
            print(f"  {depvar}: done (N={m._N})")

    # 4. Plot share_R overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_r = {0.5: "#cb181d", 0.6: "#e6550d", 0.7: "#fd8d3c", 0.8: "#fdae6b"}
    offsets = {0.5: -0.3, 0.6: -0.1, 0.7: 0.1, 0.8: 0.3}
    for t in THRESHOLDS:
        c = all_results[(t, f"share_R_{t}")]
        yrs = c["year"].values
        ax.errorbar(yrs + offsets[t], c["coef"],
                    yerr=[c["coef"] - c["ci_lo"], c["ci_hi"] - c["coef"]],
                    fmt="o", color=colors_r[t], markersize=3.5, capsize=2, linewidth=0.8,
                    label=f"P(R)>{t}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title("Share R-Leaning by Probability Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_share_R.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'threshold_share_R.png'}")

    # 5. Plot r_ratio overlay
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_b = {0.5: "#2171b5", 0.6: "#4292c6", 0.7: "#6baed6", 0.8: "#9ecae1"}
    for t in THRESHOLDS:
        c = all_results[(t, f"r_ratio_{t}")]
        yrs = c["year"].values
        ax.errorbar(yrs + offsets[t], c["coef"],
                    yerr=[c["coef"] - c["ci_lo"], c["ci_hi"] - c["coef"]],
                    fmt="o", color=colors_b[t], markersize=3.5, capsize=2, linewidth=0.8,
                    label=f"R/(R+D), P>{t}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title("R Ratio [R/(R+D)] by Probability Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_r_ratio.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'threshold_r_ratio.png'}")

    # 6. Plot 0.7 R vs D
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, (depvar, title, color) in enumerate([
        ("share_R_0.7", "Share Confident-R (P>0.7)", "#cb181d"),
        ("share_D_0.7", "Share Confident-D (P<0.3)", "#2171b5"),
    ]):
        ax = axes[idx]
        c = all_results[(0.7, depvar)]
        yrs = c["year"].values
        ax.errorbar(yrs, c["coef"],
                    yerr=[c["coef"] - c["ci_lo"], c["ci_hi"] - c["coef"]],
                    fmt="o", color=color, markersize=5, capsize=3, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Coefficient", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
        ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "threshold_07_R_vs_D.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'threshold_07_R_vs_D.png'}")

    # 7. Grid plot 4x3
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    for row_idx, t in enumerate(THRESHOLDS):
        for col_idx, (stem, title_stem, color) in enumerate([
            ("share_R", "Share Confident-R", "#cb181d"),
            ("share_D", "Share Confident-D", "#2171b5"),
            ("r_ratio", "R Ratio", "#2ca02c"),
        ]):
            ax = axes[row_idx, col_idx]
            depvar = f"{stem}_{t}"
            c = all_results[(t, depvar)]
            yrs = c["year"].values
            ax.errorbar(yrs, c["coef"],
                        yerr=[c["coef"] - c["ci_lo"], c["ci_hi"] - c["coef"]],
                        fmt="o", color=color, markersize=4, capsize=2, linewidth=1)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
            if row_idx == 3:
                ax.set_xlabel("Year", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"tau={t}", fontsize=11, fontweight="bold")
            if row_idx == 0:
                ax.set_title(title_stem, fontsize=12, fontweight="bold")
            ax.set_xticks([yr for yr in yrs if yr % 2 == 0])
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Event Study by Probability Threshold", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "threshold_grid_4x3.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {FIG_DIR / 'threshold_grid_4x3.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("Post-NAFTA average coefficients (1994-2004):")
    print("=" * 70)
    for t in THRESHOLDS:
        for stem in ["share_R", "share_D", "r_ratio"]:
            depvar = f"{stem}_{t}"
            c = all_results[(t, depvar)]
            post = c[c["year"] > BASE_YEAR]
            print(f"  {depvar:>20s}: mean_post = {post['coef'].mean():.4f}")
        print()
    print("Done.")


if __name__ == "__main__":
    main()
