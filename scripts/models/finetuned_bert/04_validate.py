"""
04_validate.py -- Compare fine-tuned BERT slant with TF-IDF and SBERT.

Diagnostics:
  1. Newspaper-congress level correlation (BERT vs TF-IDF, BERT vs SBERT)
  2. Known-bias newspaper ranking
  3. Distribution comparison
  4. Training performance summary
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.finetuned_bert.config import (
    OUTPUT_DIR, TFIDF_SLANT_DIR, SBERT_DIR, SCORE_CONGRESSES,
)


def load_tfidf_slant(cong):
    path = TFIDF_SLANT_DIR / f"08_article_slant_cong_{cong}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main():
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fine-Tuned BERT Validation")
    print("=" * 60)

    # -- Load BERT slant --
    bert_path = OUTPUT_DIR / "article_slant_all.parquet"
    if not bert_path.exists():
        print("ERROR: Run 03_score_newspapers.py first")
        return
    bert = pd.read_parquet(bert_path)
    print(f"BERT articles: {len(bert):,}")

    # -- 1. Newspaper-congress level comparison --
    print(f"\n{'---'*20}")
    print("1. Newspaper-Congress Level Comparison")

    bert_agg = bert.groupby(["paper", "congress"]).agg(
        bert_slant=("slant", "mean"),
        bert_prob_R=("prob_R", "mean"),
        n_articles=("slant", "count"),
    ).reset_index()

    # vs TF-IDF
    tfidf_parts = []
    for cong in SCORE_CONGRESSES:
        tfidf_meta_path = TFIDF_SLANT_DIR / f"07_newspaper_meta_cong_{cong}.parquet"
        tfidf_slant = load_tfidf_slant(cong)
        if tfidf_slant is None or not tfidf_meta_path.exists():
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
        merged_tfidf = bert_agg.merge(tfidf_agg, on=["paper", "congress"], how="inner")
        if len(merged_tfidf) > 2:
            r, p = stats.pearsonr(merged_tfidf["bert_slant"], merged_tfidf["tfidf_slant"])
            rho, p_rho = stats.spearmanr(merged_tfidf["bert_slant"], merged_tfidf["tfidf_slant"])
            print(f"  BERT vs TF-IDF:")
            print(f"    Matched obs: {len(merged_tfidf):,}")
            print(f"    Pearson r = {r:.4f} (p = {p:.2e})")
            print(f"    Spearman rho = {rho:.4f} (p = {p_rho:.2e})")

    # vs SBERT
    sbert_path = SBERT_DIR / "article_slant_all.parquet"
    if sbert_path.exists():
        sbert = pd.read_parquet(sbert_path)
        sbert_agg = sbert.groupby(["paper", "congress"]).agg(
            sbert_slant=("slant_raw", "mean"),
        ).reset_index()
        merged_sbert = bert_agg.merge(sbert_agg, on=["paper", "congress"], how="inner")
        if len(merged_sbert) > 2:
            r_s, p_s = stats.pearsonr(merged_sbert["bert_slant"], merged_sbert["sbert_slant"])
            rho_s, p_rho_s = stats.spearmanr(merged_sbert["bert_slant"], merged_sbert["sbert_slant"])
            print(f"\n  BERT vs SBERT:")
            print(f"    Matched obs: {len(merged_sbert):,}")
            print(f"    Pearson r = {r_s:.4f} (p = {p_s:.2e})")
            print(f"    Spearman rho = {rho_s:.4f} (p = {p_rho_s:.2e})")

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if tfidf_parts and len(merged_tfidf) > 2:
        axes[0].scatter(merged_tfidf["tfidf_slant"], merged_tfidf["bert_slant"], alpha=0.5, s=20)
        axes[0].set_xlabel("TF-IDF Net Slant")
        axes[0].set_ylabel("BERT Slant (logit_R - logit_D)")
        axes[0].set_title(f"BERT vs TF-IDF (r={r:.3f})")
        axes[0].axhline(0, color="gray", lw=0.5)
        axes[0].axvline(0, color="gray", lw=0.5)

    if sbert_path.exists() and len(merged_sbert) > 2:
        axes[1].scatter(merged_sbert["sbert_slant"], merged_sbert["bert_slant"], alpha=0.5, s=20)
        axes[1].set_xlabel("SBERT Raw Slant")
        axes[1].set_ylabel("BERT Slant (logit_R - logit_D)")
        axes[1].set_title(f"BERT vs SBERT (r={r_s:.3f})")
        axes[1].axhline(0, color="gray", lw=0.5)
        axes[1].axvline(0, color="gray", lw=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "bert_vs_tfidf_sbert_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: bert_vs_tfidf_sbert_scatter.png")

    # -- 2. Newspaper ranking --
    print(f"\n{'---'*20}")
    print("2. Newspaper Slant Ranking")

    paper_rank = bert.groupby("paper")["slant"].mean().sort_values()
    print(f"\n  {'Slant':>8}  {'P(R)':>6}  Newspaper")
    print(f"  {'---':>8}  {'---':>6}  {'---'*13}")
    paper_prob = bert.groupby("paper")["prob_R"].mean()
    for paper, slant in paper_rank.items():
        pr = paper_prob[paper]
        print(f"  {slant:+8.4f}  {pr:.4f}  {paper}")

    # -- 3. Distribution --
    print(f"\n{'---'*20}")
    print("3. Slant Distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(bert["slant"], bins=100, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("BERT Slant (logit_R - logit_D)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"BERT Slant Distribution (n={len(bert):,})")
    axes[0].axvline(0, color="red", lw=1)

    axes[1].hist(bert["prob_R"], bins=100, alpha=0.7, color="darkorange")
    axes[1].set_xlabel("P(Republican)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("P(R) Distribution")
    axes[1].axvline(0.5, color="red", lw=1)

    fig.tight_layout()
    fig.savefig(fig_dir / "bert_slant_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: bert_slant_distributions.png")

    # -- 4. Summary statistics --
    print(f"\n{'---'*20}")
    print("4. Summary Statistics")
    print(f"  Slant (logit diff):")
    print(f"    Mean:   {bert['slant'].mean():.4f}")
    print(f"    Std:    {bert['slant'].std():.4f}")
    print(f"    Median: {bert['slant'].median():.4f}")
    print(f"  P(R):")
    print(f"    Mean:   {bert['prob_R'].mean():.4f}")
    print(f"    Std:    {bert['prob_R'].std():.4f}")

    # Per-congress
    print(f"\n  Per-congress:")
    for cong in SCORE_CONGRESSES:
        sub = bert[bert["congress"] == cong]
        if len(sub) > 0:
            print(f"    Congress {cong}: mean={sub['slant'].mean():+.4f}, "
                  f"std={sub['slant'].std():.4f}, P(R)={sub['prob_R'].mean():.4f}")

    print(f"\n{'='*60}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
