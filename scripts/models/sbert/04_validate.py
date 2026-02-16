"""
04_validate.py — Compare SBERT slant with existing TF-IDF slant.

Diagnostics:
  1. Article-level correlation (SBERT vs TF-IDF net_slant)
  2. Newspaper-congress level correlation
  3. Known-bias newspaper ranking sanity check
  4. Distribution comparison (histograms)
  5. Short-article signal analysis (does SBERT give more signal on short articles?)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sbert.config import OUTPUT_DIR, TFIDF_SLANT_DIR, SCORE_CONGRESSES


def load_tfidf_slant(cong):
    """Load existing TF-IDF article slant for a congress."""
    path = TFIDF_SLANT_DIR / f"08_article_slant_cong_{cong}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main():
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SBERT vs TF-IDF Validation")
    print("=" * 60)

    # ── Load SBERT slant ──────────────────────────────────────
    sbert_path = OUTPUT_DIR / "article_slant_all.parquet"
    if not sbert_path.exists():
        print("ERROR: Run 03_compute_slant.py first")
        return
    sbert = pd.read_parquet(sbert_path)
    print(f"SBERT articles: {len(sbert):,}")

    # ── 1. Newspaper-congress level comparison ────────────────
    print(f"\n{'─'*60}")
    print("1. Newspaper-Congress Level Comparison")

    sbert_agg = sbert.groupby(["paper", "congress"]).agg(
        sbert_slant=("slant_normalized", "mean"),
        n_articles=("slant_normalized", "count"),
    ).reset_index()

    # Load TF-IDF and aggregate
    tfidf_parts = []
    for cong in SCORE_CONGRESSES:
        tfidf_meta_path = TFIDF_SLANT_DIR / f"07_newspaper_meta_cong_{cong}.parquet"
        tfidf_slant = load_tfidf_slant(cong)
        if tfidf_slant is None or not tfidf_meta_path.exists():
            print(f"  Congress {cong}: TF-IDF data not found, skipping")
            continue
        tfidf_meta = pd.read_parquet(tfidf_meta_path)
        combined = pd.concat([tfidf_meta.reset_index(drop=True),
                              tfidf_slant.reset_index(drop=True)], axis=1)
        combined["congress"] = cong
        agg = combined.groupby(["paper", "congress"]).agg(
            tfidf_slant=("net_slant", "mean"),
        ).reset_index()
        tfidf_parts.append(agg)

    if tfidf_parts:
        tfidf_agg = pd.concat(tfidf_parts, ignore_index=True)
        merged = sbert_agg.merge(tfidf_agg, on=["paper", "congress"], how="inner")

        if len(merged) > 2:
            r, p = stats.pearsonr(merged["sbert_slant"], merged["tfidf_slant"])
            rho, p_rho = stats.spearmanr(merged["sbert_slant"], merged["tfidf_slant"])
            print(f"  Matched paper-congress obs: {len(merged):,}")
            print(f"  Pearson r = {r:.4f} (p = {p:.2e})")
            print(f"  Spearman ρ = {rho:.4f} (p = {p_rho:.2e})")

            # Scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(merged["tfidf_slant"], merged["sbert_slant"], alpha=0.5, s=20)
            ax.set_xlabel("TF-IDF Net Slant (newspaper-congress avg)")
            ax.set_ylabel("SBERT Slant (newspaper-congress avg)")
            ax.set_title(f"SBERT vs TF-IDF Slant (r={r:.3f}, ρ={rho:.3f})")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            fig.tight_layout()
            fig.savefig(fig_dir / "sbert_vs_tfidf_scatter.png", dpi=150)
            plt.close(fig)
            print(f"  Saved: sbert_vs_tfidf_scatter.png")
    else:
        print("  No TF-IDF data available for comparison")

    # ── 2. Known-bias newspaper ranking ───────────────────────
    print(f"\n{'─'*60}")
    print("2. Newspaper Slant Ranking (averaged across congresses)")

    paper_rank = sbert.groupby("paper")["slant_normalized"].mean().sort_values()
    print(f"\n  {'Slant':>8}  Newspaper")
    print(f"  {'─'*8}  {'─'*40}")
    for paper, slant in paper_rank.items():
        print(f"  {slant:+8.4f}  {paper}")

    # ── 3. Distribution comparison ────────────────────────────
    print(f"\n{'─'*60}")
    print("3. Slant Distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SBERT
    axes[0].hist(sbert["slant_normalized"], bins=100, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("SBERT Normalized Slant")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"SBERT Slant Distribution (n={len(sbert):,})")
    axes[0].axvline(0, color="red", lw=1)

    # By word count
    sbert["wc_bin"] = pd.cut(sbert["word_count"], bins=[0, 50, 80, 120, 200, 10000],
                              labels=["<50", "50-80", "80-120", "120-200", ">200"])
    wc_stats = sbert.groupby("wc_bin")["slant_normalized"].agg(["mean", "std", "count"])
    axes[1].bar(range(len(wc_stats)), wc_stats["std"], color="steelblue", alpha=0.7)
    axes[1].set_xticks(range(len(wc_stats)))
    axes[1].set_xticklabels(wc_stats.index, rotation=0)
    axes[1].set_xlabel("Article Word Count Bin")
    axes[1].set_ylabel("Std Dev of Slant")
    axes[1].set_title("Signal Strength by Article Length")

    fig.tight_layout()
    fig.savefig(fig_dir / "sbert_slant_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: sbert_slant_distributions.png")

    # ── 4. Summary statistics ─────────────────────────────────
    print(f"\n{'─'*60}")
    print("4. Summary Statistics")
    print(f"  Mean:   {sbert['slant_normalized'].mean():.4f}")
    print(f"  Std:    {sbert['slant_normalized'].std():.4f}")
    print(f"  Median: {sbert['slant_normalized'].median():.4f}")
    print(f"  IQR:    [{sbert['slant_normalized'].quantile(0.25):.4f}, "
          f"{sbert['slant_normalized'].quantile(0.75):.4f}]")

    # Zero-slant analysis
    near_zero = (sbert["slant_normalized"].abs() < 0.01).mean()
    print(f"  Near-zero (|slant| < 0.01): {near_zero:.1%}")

    print(f"\n{'='*60}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
