"""
10_aggregate_slant.py

Merge normalized article-level slant scores (step 09) with labels (step 04),
filter, and aggregate into a newspaper-year panel dataset.

Filtering:
  - is_news == True only (exclude death notices, classifieds, etc.)
  - Editorials/op-eds are already excluded by this filter.

Dual-track aggregation:
  Track A (General Slant):
    Mean slant of ALL news articles per newspaper-year.
    Represents the newspaper's overall political brand.

  Track B (Economic Slant):
    Mean slant of top-10% most "economic" articles (by econ_score)
    per newspaper-year. Isolates bias in economic/factual coverage.
    The p90 econ_score threshold is computed per congress (since
    econ_score distributions vary by congress).

  Extensive margin:
    econ_share = fraction of news articles with econ_score > 0
    (article contains at least one economy seed stem).
    Captures whether newspapers respond to shocks by writing MORE
    about economics (extensive) vs changing slant (intensive).

Grouping:
  newspaper × year (not congress) — aligns with yearly economic and
  financial data for downstream regression analysis.

Inputs:
  - data/processed/newspapers/04_newspaper_labeled_cong_{cong}.parquet
  - data/processed/newspapers/09_article_slant_norm_cong_{cong}.parquet

Outputs:
  - data/processed/newspapers/10_newspaper_year_panel.parquet
"""

import gc
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"
OUT_DIR = NEWSPAPER_DIR

ECON_PERCENTILE = 90  # top 10%

SLANT_COLS_RAW = ["right_intensity", "left_intensity", "net_slant", "politicization"]
SLANT_COLS_NORM = ["right_norm", "left_norm", "net_slant_norm", "politicization_norm"]
SLANT_COLS = SLANT_COLS_RAW + SLANT_COLS_NORM


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    congresses = list(range(100, 109))

    print("Building newspaper-year panel ...\n")
    pipeline_start = time.time()

    general_chunks = []
    econ_chunks = []

    for cong in congresses:
        label_path = NEWSPAPER_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        slant_path = NEWSPAPER_DIR / f"09_article_slant_norm_cong_{cong}.parquet"

        if not label_path.exists() or not slant_path.exists():
            print(f"  WARNING: Congress {cong} files not found, skipping")
            continue

        # Load and merge (row-aligned)
        meta = pd.read_parquet(label_path)
        slant = pd.read_parquet(slant_path)

        assert len(meta) == len(slant), \
            f"Congress {cong}: meta ({len(meta)}) != slant ({len(slant)})"

        df = pd.concat([meta, slant], axis=1)
        n_total = len(df)

        # Filter: news only
        df = df[df["is_news"]].copy()
        n_news = len(df)

        # Econ threshold: per-congress p90
        econ_threshold = np.percentile(df["econ_score"], ECON_PERCENTILE)
        econ_mask = df["econ_score"] >= econ_threshold
        n_econ = econ_mask.sum()

        print(f"  Congress {cong}: {n_total:,} total -> {n_news:,} news -> "
              f"{n_econ:,} econ (p{ECON_PERCENTILE} >= {econ_threshold:.4f})")

        # Extensive margin: flag articles with any economy content
        df["is_econ"] = df["econ_score"] > 0

        # Track A: General — all news articles, grouped by paper × year
        general_agg = (
            df.groupby(["paper", "year"])
            .agg(
                n_articles=("net_slant", "count"),
                n_econ_any=("is_econ", "sum"),
                **{f"{col}": (col, "mean") for col in SLANT_COLS},
            )
            .reset_index()
        )
        general_agg["econ_share"] = general_agg["n_econ_any"] / general_agg["n_articles"]
        general_chunks.append(general_agg)

        # Track B: Economic — top 10% by econ_score, grouped by paper × year
        df_econ = df[econ_mask]
        econ_agg = (
            df_econ.groupby(["paper", "year"])
            .agg(
                n_articles_econ=("net_slant", "count"),
                **{f"{col}_econ": (col, "mean") for col in SLANT_COLS},
            )
            .reset_index()
        )
        econ_chunks.append(econ_agg)

        del meta, slant, df, df_econ, general_agg, econ_agg
        gc.collect()

    # Combine all congresses
    print("\nCombining across congresses ...")
    general = pd.concat(general_chunks, ignore_index=True)
    econ = pd.concat(econ_chunks, ignore_index=True)

    # Merge tracks A and B on paper × year
    panel = general.merge(econ, on=["paper", "year"], how="left")

    # Sort
    panel = panel.sort_values(["paper", "year"]).reset_index(drop=True)

    # Save
    out_path = OUT_DIR / "10_newspaper_year_panel.parquet"
    panel.to_parquet(out_path)

    elapsed = time.time() - pipeline_start

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Newspaper-Year Panel")
    print("=" * 72)

    n_papers = panel["paper"].nunique()
    n_years = panel["year"].nunique()
    print(f"\n  Panel dimensions: {len(panel):,} rows  "
          f"({n_papers} papers x {n_years} years)")
    print(f"  Year range: {panel['year'].min()} - {panel['year'].max()}")

    print(f"\n  Track A (General):")
    print(f"    Total news articles:   {panel['n_articles'].sum():,}")
    print(f"    Mean net_slant_norm:   {panel['net_slant_norm'].mean():.4f}")
    print(f"    Std net_slant_norm:    {panel['net_slant_norm'].std():.4f}")

    print(f"\n  Extensive margin (econ_share = articles with econ_score > 0):")
    print(f"    Mean econ_share:       {panel['econ_share'].mean():.4f}")
    print(f"    Std econ_share:        {panel['econ_share'].std():.4f}")

    n_with_econ = panel["n_articles_econ"].notna().sum()
    print(f"\n  Track B (Economic):")
    print(f"    Paper-years with econ: {n_with_econ:,} / {len(panel):,}")
    if n_with_econ > 0:
        print(f"    Total econ articles:   {panel['n_articles_econ'].sum():,.0f}")
        print(f"    Mean net_slant_norm:   {panel['net_slant_norm_econ'].mean():.4f}")
        print(f"    Std net_slant_norm:    {panel['net_slant_norm_econ'].std():.4f}")

    # Per-year summary
    print(f"\n  Per-year averages:")
    yearly = panel.groupby("year").agg(
        papers=("paper", "nunique"),
        articles=("n_articles", "sum"),
        net_slant_norm=("net_slant_norm", "mean"),
        politicization_norm=("politicization_norm", "mean"),
        econ_share=("econ_share", "mean"),
    )
    for year, row in yearly.iterrows():
        print(f"    {year}: {row['papers']:>3} papers  "
              f"{row['articles']:>9,.0f} articles  "
              f"slant_norm={row['net_slant_norm']:>7.3f}  "
              f"politic_norm={row['politicization_norm']:.3f}  "
              f"econ_share={row['econ_share']:.3f}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Saved to -> {out_path}")
    print("=" * 72)
