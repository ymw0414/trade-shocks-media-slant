"""
20_topic_separation.py

Demand-side vs supply-side mechanism test:
Split articles into economy-related vs non-economy, aggregate each group
into a newspaper-year panel, and run the event study separately.

If the slant shift appears ONLY in economy articles → mechanical/supply-side
  (newspapers just cover more trade news, which happens to use R-coded language)
If the slant shift appears in NON-economy articles too → demand-side
  (newspapers shift their overall partisan framing to match reader preferences)

Uses econ_score from step 04 (cosine similarity to economy anchor vector).
Threshold: econ_score > 0 → "economy article" (~20% of news articles).

Inputs:
  - data/processed/newspapers/04_newspaper_labeled_cong_{cong}.parquet
  - data/processed/runs/{run}/newspapers/09_article_slant_norm_cong_{cong}.parquet
  - data/processed/runs/{run}/panel/14_regression_panel.parquet

Outputs:
  - output/figures/topic_separation_ext_R.png
  - output/figures/topic_separation_ext_D.png
  - output/figures/topic_separation_net_slant.png
  - output/figures/topic_separation_ext_net.png
  - output/tables/topic_separation_coefficients.csv
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

LABEL_DIR = cfg.NEWSPAPER_LABELS
NEWS_DIR = cfg.NEWS_DIR
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

BASE_YEAR = 1993
END_YEAR = 2004
ECON_THRESHOLD = 0.0  # econ_score > 0 means any economic content

SLANT_COLS_RAW = ["right_intensity", "left_intensity", "net_slant", "politicization"]
SLANT_COLS_NORM = ["right_norm", "left_norm", "net_slant_norm", "politicization_norm"]
SLANT_COLS = SLANT_COLS_RAW + SLANT_COLS_NORM

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


# ==================================================================
# Aggregation (same logic as step 10, but for a subset of articles)
# ==================================================================
def aggregate_subset(df):
    """Aggregate article-level data to newspaper-year panel."""
    # Unconditional means
    agg = (
        df.groupby(["paper", "year"])
        .agg(
            n_articles=("net_slant", "count"),
            **{col: (col, "mean") for col in SLANT_COLS},
        )
        .reset_index()
    )

    # Extensive margins
    grp = df.groupby(["paper", "year"])["net_slant"]
    agg["ext_nonzero"] = grp.apply(lambda x: (x != 0).mean()).values
    agg["ext_R"] = grp.apply(lambda x: (x > 0).mean()).values
    agg["ext_D"] = grp.apply(lambda x: (x < 0).mean()).values

    # Intensive margins
    agg["int_net_slant_norm"] = df.groupby(["paper", "year"]).apply(
        lambda g: g.loc[g["net_slant"] != 0, "net_slant_norm"].mean()
        if (g["net_slant"] != 0).any() else np.nan
    ).values
    agg["int_R_norm"] = df.groupby(["paper", "year"]).apply(
        lambda g: g.loc[g["net_slant"] > 0, "net_slant_norm"].mean()
        if (g["net_slant"] > 0).any() else np.nan
    ).values
    agg["int_D_norm"] = df.groupby(["paper", "year"]).apply(
        lambda g: g.loc[g["net_slant"] < 0, "net_slant_norm"].mean()
        if (g["net_slant"] < 0).any() else np.nan
    ).values

    return agg


def build_topic_panels():
    """Build separate newspaper-year panels for econ and non-econ articles."""
    congresses = cfg.get_congresses()

    chunks_econ = []
    chunks_nonecon = []

    for cong in congresses:
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        slant_path = NEWS_DIR / f"09_article_slant_norm_cong_{cong}.parquet"

        if not label_path.exists() or not slant_path.exists():
            print(f"  WARNING: Congress {cong} files not found, skipping")
            continue

        meta = pd.read_parquet(label_path)
        slant = pd.read_parquet(slant_path)

        # Apply sample index if exists (for shared vocab runs)
        sample_idx_path = NEWS_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)

        assert len(meta) == len(slant), \
            f"Congress {cong}: meta ({len(meta)}) != slant ({len(slant)})"

        df = pd.concat([meta, slant], axis=1)

        # Filter: news only
        df = df[df["is_news"]].copy()

        # Split by econ_score
        is_econ = df["econ_score"] > ECON_THRESHOLD
        df_econ = df[is_econ]
        df_nonecon = df[~is_econ]

        n_econ = len(df_econ)
        n_nonecon = len(df_nonecon)
        pct_econ = n_econ / (n_econ + n_nonecon) * 100

        print(f"  Congress {cong}: {n_econ + n_nonecon:,} news -> "
              f"{n_econ:,} econ ({pct_econ:.1f}%), {n_nonecon:,} non-econ")

        if n_econ > 0:
            chunks_econ.append(aggregate_subset(df_econ))
        if n_nonecon > 0:
            chunks_nonecon.append(aggregate_subset(df_nonecon))

        del meta, slant, df, df_econ, df_nonecon
        gc.collect()

    panel_econ = pd.concat(chunks_econ, ignore_index=True).sort_values(
        ["paper", "year"]).reset_index(drop=True)
    panel_nonecon = pd.concat(chunks_nonecon, ignore_index=True).sort_values(
        ["paper", "year"]).reset_index(drop=True)

    return panel_econ, panel_nonecon


def merge_with_geography(panel_subset, panel_full):
    """Merge topic-specific panel with geography and NAFTA vars from full panel."""
    # Get unique paper -> geography mapping from full panel
    geo_cols = ["paper", "fips", "cz", "division",
                "vulnerability1990_scaled", "china_shock", "manushare1990",
                "pct_bachelor1990", "income_pc1989"]
    # Keep only columns that exist
    geo_cols = [c for c in geo_cols if c in panel_full.columns]

    geo = panel_full.drop_duplicates("paper")[geo_cols].copy()

    # Merge
    merged = panel_subset.merge(geo, on="paper", how="inner")
    return merged


def prepare_for_regression(df):
    """Add event study interaction variables."""
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    if END_YEAR is not None:
        df = df[df["year"] <= END_YEAR].copy()

    if "division" not in df.columns:
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


def run_event_study(df, depvar, years):
    """Run controlled event study, return coefficients."""
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

    return pd.DataFrame(rows), m


def plot_comparison(coefs_econ, coefs_nonecon, depvar_label, out_path,
                    n_econ=None, n_nonecon=None):
    """Plot econ vs non-econ event studies on the same axes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    yrs = coefs_econ["year"].values
    offset = 0.15

    label_e = f"Economy articles"
    label_n = f"Non-economy articles"
    if n_econ is not None:
        label_e += f" (N={n_econ:,})"
    if n_nonecon is not None:
        label_n += f" (N={n_nonecon:,})"

    # Economy articles
    ax.errorbar(yrs - offset, coefs_econ["coef"],
                yerr=[coefs_econ["coef"] - coefs_econ["ci_lo"],
                      coefs_econ["ci_hi"] - coefs_econ["coef"]],
                fmt="o", color="#c44e52", markersize=5, capsize=2.5,
                linewidth=1.2, label=label_e)

    # Non-economy articles
    ax.errorbar(yrs + offset, coefs_nonecon["coef"],
                yerr=[coefs_nonecon["coef"] - coefs_nonecon["ci_lo"],
                      coefs_nonecon["ci_hi"] - coefs_nonecon["coef"]],
                fmt="s", color="#4c72b0", markersize=5, capsize=2.5,
                linewidth=1.2, label=label_n)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title(depvar_label, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
    ax.set_xticklabels([str(yr) for yr in yrs if yr % 2 == 1 or yr == yrs[0]],
                        fontsize=10)
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"    Saved: {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Build topic-specific panels ---
    print("=" * 72)
    print("TOPIC SEPARATION: Economy vs Non-Economy Articles")
    print("=" * 72)

    print("\nBuilding topic-specific panels ...")
    panel_econ, panel_nonecon = build_topic_panels()
    print(f"\n  Economy panel:     {len(panel_econ):,} newspaper-years, "
          f"{panel_econ['n_articles'].sum():,.0f} articles")
    print(f"  Non-economy panel: {len(panel_nonecon):,} newspaper-years, "
          f"{panel_nonecon['n_articles'].sum():,.0f} articles")

    # --- Step 2: Merge with geography ---
    print("\nMerging with geography and NAFTA variables ...")
    panel_full = pd.read_parquet(PANEL_PATH)
    # Add division from state FIPS
    panel_full["state_fips"] = (panel_full["fips"] // 1000).astype(int)
    panel_full["division"] = panel_full["state_fips"].map(STATE_TO_DIVISION)

    df_econ = merge_with_geography(panel_econ, panel_full)
    df_nonecon = merge_with_geography(panel_nonecon, panel_full)
    print(f"  Economy:     {len(df_econ):,} obs after merge")
    print(f"  Non-economy: {len(df_nonecon):,} obs after merge")

    # Derived: net extensive margin
    df_econ["ext_net"] = df_econ["ext_R"] - df_econ["ext_D"]
    df_nonecon["ext_net"] = df_nonecon["ext_R"] - df_nonecon["ext_D"]

    # --- Step 3: Prepare for regression ---
    df_econ_reg, years_e = prepare_for_regression(df_econ)
    df_nonecon_reg, years_n = prepare_for_regression(df_nonecon)
    years = years_e  # should be same

    n_econ = int(df_econ_reg["n_articles"].sum())
    n_nonecon = int(df_nonecon_reg["n_articles"].sum())

    print(f"\n  Regression sample:")
    print(f"    Economy:     {len(df_econ_reg):,} obs, "
          f"{df_econ_reg['paper_id'].nunique()} papers, "
          f"{df_econ_reg['cz'].nunique()} CZs")
    print(f"    Non-economy: {len(df_nonecon_reg):,} obs, "
          f"{df_nonecon_reg['paper_id'].nunique()} papers, "
          f"{df_nonecon_reg['cz'].nunique()} CZs")

    # --- Step 4: Run event studies ---
    outcomes = [
        ("net_slant_norm", "Net Slant (Normalized)"),
        ("ext_net",        "Net Extensive Margin (Share R - Share D)"),
        ("ext_R",          "Share R-Leaning Articles"),
        ("ext_D",          "Share D-Leaning Articles"),
        ("int_net_slant_norm", "Net Slant (Intensive Margin)"),
        ("int_R_norm",     "R Slant (Intensive Margin)"),
        ("int_D_norm",     "D Slant (Intensive Margin)"),
    ]

    all_rows = []
    for depvar, label in outcomes:
        print(f"\n  Running: {depvar}")

        coefs_e, m_e = run_event_study(df_econ_reg, depvar, years)
        coefs_n, m_n = run_event_study(df_nonecon_reg, depvar, years)

        print(f"    Economy:     N={m_e._N}, R2={m_e._r2:.4f}")
        print(f"    Non-economy: N={m_n._N}, R2={m_n._r2:.4f}")

        # Save coefficients
        for subset, coefs in [("economy", coefs_e), ("non_economy", coefs_n)]:
            c = coefs.copy()
            c["depvar"] = depvar
            c["subset"] = subset
            all_rows.append(c)

        # Plot comparison
        plot_comparison(
            coefs_e, coefs_n, label,
            FIG_DIR / f"topic_separation_{depvar}.png",
            n_econ=n_econ, n_nonecon=n_nonecon,
        )

    # --- Combined extensive margin plot (2x1) ---
    print("\n  Creating combined extensive margin figure ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax_idx, (depvar, title) in enumerate([
        ("ext_R", "Share R-Leaning Articles"),
        ("ext_D", "Share D-Leaning Articles"),
    ]):
        ax = axes[ax_idx]
        coefs_e, _ = run_event_study(df_econ_reg, depvar, years)
        coefs_n, _ = run_event_study(df_nonecon_reg, depvar, years)

        yrs = coefs_e["year"].values
        offset = 0.15

        ax.errorbar(yrs - offset, coefs_e["coef"],
                    yerr=[coefs_e["coef"] - coefs_e["ci_lo"],
                          coefs_e["ci_hi"] - coefs_e["coef"]],
                    fmt="o", color="#c44e52", markersize=5, capsize=2.5,
                    linewidth=1.2, label="Economy articles")
        ax.errorbar(yrs + offset, coefs_n["coef"],
                    yerr=[coefs_n["coef"] - coefs_n["ci_lo"],
                          coefs_n["ci_hi"] - coefs_n["coef"]],
                    fmt="s", color="#4c72b0", markersize=5, capsize=2.5,
                    linewidth=1.2, label="Non-economy articles")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.set_xlabel("Year", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
        ax.set_xticklabels([str(yr) for yr in yrs if yr % 2 == 1 or yr == yrs[0]],
                            fontsize=9)
        ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIG_DIR / "topic_separation_ext_combined.png",
                dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"    Saved: {FIG_DIR / 'topic_separation_ext_combined.png'}")

    # --- Save coefficients ---
    all_df = pd.concat(all_rows, ignore_index=True)
    tab_path = TAB_DIR / "topic_separation_coefficients.csv"
    all_df.to_csv(tab_path, index=False, float_format="%.6f")
    print(f"\n  Coefficients saved: {tab_path}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  Economy articles (econ_score > {ECON_THRESHOLD}):")
    print(f"    Total: {n_econ:,} articles across {len(df_econ_reg):,} newspaper-years")
    print(f"  Non-economy articles (econ_score = {ECON_THRESHOLD}):")
    print(f"    Total: {n_nonecon:,} articles across {len(df_nonecon_reg):,} newspaper-years")

    # Print key DiD-style comparison (post-NAFTA mean coefs)
    post_years = [yr for yr in years if yr > BASE_YEAR]
    for depvar, label in [("ext_R", "Share R"), ("ext_D", "Share D"),
                          ("ext_net", "Ext Net"), ("net_slant_norm", "Net Slant")]:
        subset_df = all_df[all_df["depvar"] == depvar]
        for subset in ["economy", "non_economy"]:
            sub = subset_df[(subset_df["subset"] == subset) &
                            (subset_df["year"].isin(post_years))]
            mean_coef = sub["coef"].mean()
            print(f"    {label:>12s} | {subset:>12s}: mean post-NAFTA coef = {mean_coef:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
