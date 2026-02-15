"""
03_validate.py -- Validate PRISM slant against TF-IDF and SBERT baselines.

Checks:
  1. Correlation with TF-IDF slant (newspaper-congress level)
  2. Correlation with SBERT slant
  3. Newspaper ranking sanity check
  4. Distribution analysis + histograms
  5. Topic-level analysis (PRISM-unique)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.prism.config import (
    OUTPUT_DIR, FIG_DIR, TAB_DIR, TFIDF_SLANT_DIR, SBERT_DIR,
    SCORE_CONGRESSES,
)


def load_prism():
    """Load PRISM article-level slant."""
    path = OUTPUT_DIR / "article_slant_all.parquet"
    df = pd.read_parquet(path)
    print(f"PRISM articles: {len(df):,}")
    return df


def load_tfidf():
    """Load TF-IDF newspaper-year slant."""
    path = TFIDF_SLANT_DIR / "10_newspaper_year_slant.parquet"
    if not path.exists():
        print(f"  TF-IDF slant not found: {path}")
        return None
    df = pd.read_parquet(path)
    print(f"TF-IDF newspaper-year obs: {len(df):,}")
    return df


def load_sbert():
    """Load SBERT article-level slant."""
    path = SBERT_DIR / "article_slant_all.parquet"
    if not path.exists():
        print(f"  SBERT slant not found: {path}")
        return None
    df = pd.read_parquet(path)
    print(f"SBERT articles: {len(df):,}")
    return df


def aggregate_to_paper_congress(df, slant_col, prefix):
    """Aggregate article-level slant to paper x congress."""
    agg = df.groupby(["paper", "congress"]).agg(
        **{f"{prefix}_slant": (slant_col, "mean"),
           f"{prefix}_n": (slant_col, "count"),
           f"{prefix}_std": (slant_col, "std")}
    ).reset_index()
    return agg


def compare_methods(prism, tfidf, sbert):
    """Compute correlations between methods at newspaper-congress level."""
    print("\n" + "=" * 60)
    print("METHOD COMPARISON (newspaper-congress level)")
    print("=" * 60)

    # Aggregate PRISM
    prism_agg = aggregate_to_paper_congress(
        prism.dropna(subset=["prism_slant"]),
        "prism_slant", "prism"
    )

    results = []

    # vs TF-IDF
    if tfidf is not None:
        # TF-IDF is already at newspaper-year; need to map to congress
        # For simplicity, merge on paper and check overlap
        merged = prism_agg.merge(
            tfidf[["paper", "year", "net_slant"]].rename(columns={"net_slant": "tfidf_slant"}),
            on="paper", how="inner"
        )
        if len(merged) > 0:
            # Average across years within congress for TF-IDF
            tfidf_by_paper = merged.groupby("paper").agg(
                tfidf_slant=("tfidf_slant", "mean"),
                prism_slant=("prism_slant", "mean"),
            )
            r, p = stats.pearsonr(tfidf_by_paper["prism_slant"],
                                   tfidf_by_paper["tfidf_slant"])
            print(f"\n  PRISM vs TF-IDF (newspaper level, N={len(tfidf_by_paper)}):")
            print(f"    Pearson r = {r:.4f} (p = {p:.4f})")
            results.append({"comparison": "PRISM vs TF-IDF", "r": r, "p": p,
                           "N": len(tfidf_by_paper)})

    # vs SBERT
    if sbert is not None:
        sbert_agg = aggregate_to_paper_congress(sbert, "slant_raw", "sbert")
        merged = prism_agg.merge(sbert_agg, on=["paper", "congress"], how="inner")
        if len(merged) > 0:
            r, p = stats.pearsonr(merged["prism_slant"], merged["sbert_slant"])
            print(f"\n  PRISM vs SBERT (paper-congress level, N={len(merged)}):")
            print(f"    Pearson r = {r:.4f} (p = {p:.4f})")
            results.append({"comparison": "PRISM vs SBERT", "r": r, "p": p,
                           "N": len(merged)})

    if results:
        pd.DataFrame(results).to_csv(
            TAB_DIR / "prism_method_correlations.csv", index=False, float_format="%.6f"
        )

    return results


def newspaper_ranking(prism):
    """Sanity check: newspaper slant ranking."""
    print("\n" + "=" * 60)
    print("NEWSPAPER RANKING (PRISM slant)")
    print("=" * 60)

    ranking = prism.groupby("paper")["prism_slant"].agg(["mean", "std", "count"])
    ranking = ranking.sort_values("mean")
    ranking.columns = ["slant_mean", "slant_std", "n_articles"]

    print(f"\n  {'Paper':<35} {'Slant':>10} {'Std':>8} {'N':>8}")
    print("  " + "-" * 65)
    for paper, row in ranking.iterrows():
        print(f"  {paper:<35} {row['slant_mean']:+.6f} {row['slant_std']:.4f} {int(row['n_articles']):>8}")

    ranking.to_csv(TAB_DIR / "prism_newspaper_ranking.csv", float_format="%.6f")
    return ranking


def plot_distributions(prism):
    """Plot PRISM slant distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid = prism["prism_slant"].dropna()

    # Histogram
    axes[0].hist(valid, bins=100, color="#2171b5", alpha=0.7, edgecolor="white", linewidth=0.3)
    axes[0].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[0].set_xlabel("PRISM Slant (sim_R - sim_D)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"PRISM Slant Distribution (N={len(valid):,})")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # By congress
    for cong in sorted(prism["congress"].unique()):
        subset = prism[prism["congress"] == cong]["prism_slant"].dropna()
        axes[1].hist(subset, bins=50, alpha=0.4, label=f"Congress {cong}")
    axes[1].axvline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_xlabel("PRISM Slant")
    axes[1].set_ylabel("Count")
    axes[1].set_title("PRISM Slant by Congress")
    axes[1].legend(fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "prism_slant_distribution.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: prism_slant_distribution.png")


def topic_analysis(prism):
    """Analyze topic-level scores (PRISM-unique feature)."""
    print("\n" + "=" * 60)
    print("TOPIC-LEVEL ANALYSIS")
    print("=" * 60)

    all_topic_scores = []
    for cong in SCORE_CONGRESSES:
        path = OUTPUT_DIR / f"article_topic_scores_cong_{cong}.parquet"
        if path.exists():
            ts = pd.read_parquet(path)
            all_topic_scores.append(ts)

    if not all_topic_scores:
        print("  No topic scores found")
        return

    topic_df = pd.concat(all_topic_scores, ignore_index=True)
    print(f"\n  Total article-topic pairs: {len(topic_df):,}")

    # Average slant per topic
    topic_avg = topic_df.groupby("topic_label").agg(
        mean_slant=("topic_slant", "mean"),
        std_slant=("topic_slant", "std"),
        n_articles=("topic_slant", "count"),
    ).sort_values("mean_slant")

    print(f"\n  Top 10 D-leaning topics (most negative slant):")
    for label, row in topic_avg.head(10).iterrows():
        print(f"    {row['mean_slant']:+.6f} ({row['n_articles']:,} articles) {label}")

    print(f"\n  Top 10 R-leaning topics (most positive slant):")
    for label, row in topic_avg.tail(10).iterrows():
        print(f"    {row['mean_slant']:+.6f} ({row['n_articles']:,} articles) {label}")

    topic_avg.to_csv(TAB_DIR / "prism_topic_slant_summary.csv", float_format="%.6f")

    # Plot topic slant bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(30, len(topic_avg))
    plot_data = topic_avg.iloc[:top_n]
    colors = ["#cb181d" if x > 0 else "#2171b5" for x in plot_data["mean_slant"]]
    ax.barh(range(top_n), plot_data["mean_slant"], color=colors, alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(plot_data.index, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean Topic Slant (sim_R - sim_D)")
    ax.set_title("PRISM: Topic-Level Slant")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prism_topic_slant_bar.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: prism_topic_slant_bar.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRISM Step 3: Validation")
    print("=" * 60)

    prism = load_prism()
    tfidf = load_tfidf()
    sbert = load_sbert()

    # Distribution stats
    valid = prism["prism_slant"].dropna()
    print(f"\nPRISM slant summary:")
    print(f"  N valid: {len(valid):,} / {len(prism):,}")
    print(f"  Mean:   {valid.mean():.6f}")
    print(f"  Std:    {valid.std():.6f}")
    print(f"  Median: {valid.median():.6f}")
    print(f"  Range:  [{valid.min():.4f}, {valid.max():.4f}]")

    # Compare methods
    compare_methods(prism, tfidf, sbert)

    # Newspaper ranking
    newspaper_ranking(prism)

    # Distributions
    plot_distributions(prism)

    # Topic analysis
    topic_analysis(prism)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
