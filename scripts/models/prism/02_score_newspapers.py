"""
02_score_newspapers.py -- Cross-encoder scoring of newspaper articles.

Per congress:
  1. Load newspaper articles (same sampling as SBERT: 10K per congress, seed=42)
  2. Load topic centroids + topic info from 01_mine_topics
  3. For each article, find top-M relevant topics via cosine similarity
  4. Cross-encoder score: sim(article, R_indicator) and sim(article, D_indicator)
  5. topic_slant = sim_R - sim_D, overall_slant = mean across topics

Output per congress:
  article_slant_cong_{N}.parquet        -- overall slant per article
  article_topic_scores_cong_{N}.parquet -- per-topic scores for analysis
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.prism.config import (
    CROSS_ENCODER_MODEL, CROSS_ENCODER_BATCH_SIZE,
    TOP_M_TOPICS, SBERT_DIR, RAW_NEWSPAPERS, OUTPUT_DIR,
    NEWSPAPER_SAMPLE_PER_CONGRESS,
    SCORE_CONGRESSES, congress_to_years, get_windows,
)


def load_articles(congress):
    """Load and sample newspaper articles for a congress (same as SBERT)."""
    year1, year2 = congress_to_years(congress)
    dfs = []
    for yr in [year1, year2]:
        path = RAW_NEWSPAPERS / f"newspapers_{yr}.parquet"
        if not path.exists():
            print(f"    WARNING: {path.name} not found")
            continue
        df_yr = pd.read_parquet(path)
        df_yr["year"] = yr
        dfs.append(df_yr)

    if not dfs:
        return None

    articles = pd.concat(dfs, ignore_index=True)
    articles["congress"] = congress

    n_sample = min(NEWSPAPER_SAMPLE_PER_CONGRESS, len(articles))
    articles = articles.sample(n=n_sample, random_state=42).reset_index(drop=True)

    # Combine title + text (same as SBERT)
    articles["full_text"] = (
        articles["title"].fillna("") + " " + articles["text"].fillna("")
    ).str.strip()

    return articles


def find_relevant_topics(article_embeddings, topic_centroids, selected_cluster_ids, top_m):
    """Find top-M relevant topics per article using cosine similarity.

    Uses pre-computed SBERT newspaper embeddings for fast bi-encoder filtering.
    topic_centroids: (n_all_clusters, dim) from k-means
    selected_cluster_ids: indices of the polarized clusters we kept
    """
    # Extract centroids for selected clusters only
    sel_centroids = topic_centroids[selected_cluster_ids]  # (n_topics, dim)

    # Normalize for cosine similarity
    art_norm = article_embeddings / (np.linalg.norm(article_embeddings, axis=1, keepdims=True) + 1e-8)
    cen_norm = sel_centroids / (np.linalg.norm(sel_centroids, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity: (n_articles, n_topics)
    sim_matrix = art_norm @ cen_norm.T

    # Top-M topics per article
    top_m_actual = min(top_m, len(selected_cluster_ids))
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :top_m_actual]

    return top_indices, sim_matrix


def score_with_cross_encoder(cross_encoder, articles_text, topics_df, top_indices):
    """Score articles against R/D indicators using NLI cross-encoder.

    NLI model outputs 3 logits: [contradiction, entailment, neutral].
    Slant = entailment_prob(article, R_indicator) - entailment_prob(article, D_indicator)
    Positive = article framing aligns more with R position on this topic.

    Returns:
      article_scores: DataFrame with overall slant per article
      topic_scores: DataFrame with per-topic scores for each article
    """
    from scipy.special import softmax

    n_articles = len(articles_text)
    n_topics_per_article = top_indices.shape[1]

    # Prepare all pairs
    r_pairs = []
    d_pairs = []
    pair_map = []  # (article_idx, topic_idx_in_selected)

    for i in range(n_articles):
        art_text = articles_text[i]
        # Truncate article to ~80 words for cross-encoder (512 token limit)
        art_truncated = " ".join(art_text.split()[:80])

        for j in range(n_topics_per_article):
            topic_local_idx = top_indices[i, j]
            topic_row = topics_df.iloc[topic_local_idx]

            r_text = topic_row.get("r_indicator_text", "")
            d_text = topic_row.get("d_indicator_text", "")

            if not r_text or not d_text:
                continue

            r_pairs.append([art_truncated, r_text])
            d_pairs.append([art_truncated, d_text])
            pair_map.append((i, topic_local_idx))

    if not r_pairs:
        return None, None

    print(f"    NLI cross-encoder pairs: {len(r_pairs):,} R + {len(d_pairs):,} D")

    # NLI model returns logits for [contradiction, entailment, neutral]
    r_logits = cross_encoder.predict(
        r_pairs, batch_size=CROSS_ENCODER_BATCH_SIZE, show_progress_bar=True
    )
    d_logits = cross_encoder.predict(
        d_pairs, batch_size=CROSS_ENCODER_BATCH_SIZE, show_progress_bar=True
    )

    # Convert logits to probabilities
    r_probs = softmax(r_logits, axis=1)  # (n_pairs, 3)
    d_probs = softmax(d_logits, axis=1)

    # entailment is index 1: [contradiction=0, entailment=1, neutral=2]
    r_entail = r_probs[:, 1]
    d_entail = d_probs[:, 1]
    r_contra = r_probs[:, 0]
    d_contra = d_probs[:, 0]

    # Organize results
    topic_rows = []
    article_slants = {}  # article_idx -> list of topic_slant

    for idx, (art_i, topic_j) in enumerate(pair_map):
        # Slant = entailment(R) - entailment(D)
        # Positive = article framing agrees more with R indicator
        topic_slant = float(r_entail[idx]) - float(d_entail[idx])
        topic_row = topics_df.iloc[topic_j]

        topic_rows.append({
            "article_idx": art_i,
            "topic_idx": int(topic_row["topic_idx"]),
            "topic_label": topic_row["topic_label"],
            "entail_R": float(r_entail[idx]),
            "entail_D": float(d_entail[idx]),
            "contra_R": float(r_contra[idx]),
            "contra_D": float(d_contra[idx]),
            "topic_slant": topic_slant,
        })

        if art_i not in article_slants:
            article_slants[art_i] = []
        article_slants[art_i].append(topic_slant)

    topic_scores_df = pd.DataFrame(topic_rows)

    # Overall slant = mean across topics
    overall = []
    for art_i in range(n_articles):
        slants = article_slants.get(art_i, [])
        overall.append({
            "article_idx": art_i,
            "prism_slant": np.mean(slants) if slants else np.nan,
            "prism_n_topics": len(slants),
            "prism_slant_std": np.std(slants) if len(slants) > 1 else 0.0,
        })

    article_scores_df = pd.DataFrame(overall)

    # Summary stats
    print(f"    R entailment: mean={r_entail.mean():.4f}, std={r_entail.std():.4f}")
    print(f"    D entailment: mean={d_entail.mean():.4f}, std={d_entail.std():.4f}")
    print(f"    Slant (entail_R - entail_D): mean={np.mean(r_entail - d_entail):.6f}")

    return article_scores_df, topic_scores_df


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRISM Step 2: Cross-Encoder Newspaper Scoring")
    print("=" * 60)

    # Load cross-encoder
    print(f"\nLoading cross-encoder: {CROSS_ENCODER_MODEL}")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    all_slant = []

    windows = get_windows()
    for wi, window in enumerate(windows):
        prev_cong, curr_cong = window

        print(f"\n{'=' * 60}")
        print(f"Congress {curr_cong} (window {prev_cong}-{curr_cong})")
        print("=" * 60)

        # Skip if already scored
        out_path = OUTPUT_DIR / f"article_slant_cong_{curr_cong}.parquet"
        if out_path.exists():
            print(f"  Already scored, loading existing: {out_path.name}")
            all_slant.append(pd.read_parquet(out_path))
            continue

        # Cooling pause between congresses (prevent thermal throttling)
        if wi > 0:
            print(f"  Cooling pause (30s)...")
            time.sleep(30)

        # ── Load topics ──────────────────────────────────────────
        topics_path = OUTPUT_DIR / f"topics_window_{prev_cong}_{curr_cong}.parquet"
        centroids_path = OUTPUT_DIR / f"topic_centroids_window_{prev_cong}_{curr_cong}.npy"

        if not topics_path.exists():
            print(f"  Topics not found: {topics_path.name}, skipping")
            continue

        topics_df = pd.read_parquet(topics_path)
        centroids = np.load(centroids_path)
        selected_cluster_ids = topics_df["cluster_id"].values

        print(f"  Topics: {len(topics_df)} polarized clusters")

        # ── Load newspaper articles ──────────────────────────────
        articles = load_articles(curr_cong)
        if articles is None:
            print(f"  No articles for congress {curr_cong}")
            continue
        print(f"  Articles: {len(articles):,}")

        # ── Load SBERT newspaper embeddings for topic matching ───
        news_emb_path = SBERT_DIR / f"newspaper_embeddings_cong_{curr_cong}.npz"
        if not news_emb_path.exists():
            print(f"  No SBERT newspaper embeddings, skipping")
            continue

        news_emb = np.load(news_emb_path)["embeddings"]
        news_meta = pd.read_parquet(SBERT_DIR / f"newspaper_meta_cong_{curr_cong}.parquet")

        # SBERT embeddings match SBERT sampling (same seed=42)
        # So we can use them directly for topic matching
        print(f"  SBERT embeddings: {news_emb.shape}")

        # ── Find relevant topics per article ─────────────────────
        top_indices, sim_matrix = find_relevant_topics(
            news_emb, centroids, selected_cluster_ids, TOP_M_TOPICS
        )
        print(f"  Top-{TOP_M_TOPICS} topics assigned per article")

        # ── Cross-encoder scoring ────────────────────────────────
        print(f"  Scoring with cross-encoder...")
        articles_text = articles["full_text"].tolist()

        article_scores, topic_scores = score_with_cross_encoder(
            cross_encoder, articles_text, topics_df, top_indices
        )

        if article_scores is None:
            print(f"  No valid scores, skipping")
            continue

        # ── Merge with article metadata ──────────────────────────
        result = news_meta.copy()
        result = result.merge(article_scores, left_index=True, right_on="article_idx", how="left")
        result["congress"] = curr_cong
        result.drop(columns=["article_idx"], inplace=True, errors="ignore")

        # Save article-level slant
        result.to_parquet(out_path, index=False)

        # Save topic-level scores
        topic_scores["congress"] = curr_cong
        topic_out = OUTPUT_DIR / f"article_topic_scores_cong_{curr_cong}.parquet"
        topic_scores.to_parquet(topic_out, index=False)

        # Stats
        valid = result["prism_slant"].dropna()
        print(f"\n  Results: {len(valid):,} articles scored")
        print(f"    Slant: mean={valid.mean():.6f}, std={valid.std():.6f}")
        print(f"    Range: [{valid.min():.4f}, {valid.max():.4f}]")

        all_slant.append(result)

        # Cooling pause after each congress
        print(f"  Cooling pause (30s)...")
        time.sleep(30)

    # ── Combined ─────────────────────────────────────────────────
    if all_slant:
        combined = pd.concat(all_slant, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "article_slant_all.parquet", index=False)
        print(f"\n{'='*60}")
        print(f"Total articles scored: {len(combined):,}")
        print(f"Saved: article_slant_all.parquet")

        # Per-newspaper ranking
        paper_avg = combined.groupby("paper")["prism_slant"].mean().sort_values()
        print(f"\nNewspaper slant ranking (low=D, high=R):")
        for paper, slant in paper_avg.items():
            print(f"  {slant:+.6f}  {paper}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
