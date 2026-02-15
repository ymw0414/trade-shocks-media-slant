"""
03_compute_slant.py — Build partisan axis and compute newspaper slant scores.

For each 2-congress window (c-1, c):
  1. Load speech embeddings for congresses c-1 and c
  2. Compute μ_R (mean R core embedding) and μ_D (mean D core embedding)
  3. Partisan axis v = μ_R - μ_D
  4. Load newspaper embeddings for congress c
  5. slant_i = (e_i · v) / ||v||
  6. Normalize by partisan gap: gap = (μ_R · v̂) - (μ_D · v̂)

Output per congress:
  partisan_axis_cong_{N}.npz    (mu_R, mu_D, axis, gap)
  article_slant_cong_{N}.parquet (slant_raw, slant_normalized, + meta)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sbert.config import OUTPUT_DIR, get_windows, PARTISAN_CORE_ONLY


def load_speech_data(cong):
    """Load embeddings + meta for one congress."""
    emb_path = OUTPUT_DIR / f"speech_embeddings_cong_{cong}.npz"
    meta_path = OUTPUT_DIR / f"speech_meta_cong_{cong}.parquet"
    if not emb_path.exists():
        return None, None
    emb = np.load(emb_path)["embeddings"]
    meta = pd.read_parquet(meta_path)
    return emb, meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_slant = []

    for window in get_windows():
        prev_cong, curr_cong = window
        print(f"\n{'='*60}")
        print(f"Window ({prev_cong}, {curr_cong}) → scoring Congress {curr_cong}")

        # ── 1. Load speech embeddings for both congresses in window ──
        embs, metas = [], []
        for c in window:
            e, m = load_speech_data(c)
            if e is not None:
                embs.append(e)
                metas.append(m)
                print(f"  Congress {c}: {len(m):,} speeches loaded")
            else:
                print(f"  Congress {c}: no data found")

        if not embs:
            print(f"  Skipping window — no speech data")
            continue

        all_emb = np.vstack(embs)
        all_meta = pd.concat(metas, ignore_index=True)

        # ── 2. Compute partisan centroids ─────────────────────────
        if PARTISAN_CORE_ONLY:
            r_mask = all_meta["label_rep_core"].values == 1
            d_mask = all_meta["label_dem_core"].values == 1
        else:
            r_mask = all_meta["party"].values == "R"
            d_mask = all_meta["party"].values == "D"

        mu_R = all_emb[r_mask].mean(axis=0)
        mu_D = all_emb[d_mask].mean(axis=0)

        print(f"  μ_R from {r_mask.sum():,} R speeches")
        print(f"  μ_D from {d_mask.sum():,} D speeches")

        # ── 3. Partisan axis ──────────────────────────────────────
        axis = mu_R - mu_D
        axis_norm = np.linalg.norm(axis)
        axis_hat = axis / axis_norm  # unit vector

        # Partisan gap (for normalization, analogous to G&S)
        gap = float(mu_R @ axis_hat - mu_D @ axis_hat)
        print(f"  ||axis|| = {axis_norm:.4f}")
        print(f"  Partisan gap = {gap:.4f}")

        # Cosine similarity between μ_R and μ_D
        cos_sim = float(mu_R @ mu_D / (np.linalg.norm(mu_R) * np.linalg.norm(mu_D)))
        print(f"  cos(μ_R, μ_D) = {cos_sim:.4f}")

        # Save axis
        np.savez_compressed(
            OUTPUT_DIR / f"partisan_axis_cong_{curr_cong}.npz",
            mu_R=mu_R, mu_D=mu_D, axis=axis, axis_hat=axis_hat,
            gap=np.array([gap]), axis_norm=np.array([axis_norm]),
        )

        # ── 4. Load newspaper embeddings ──────────────────────────
        news_emb_path = OUTPUT_DIR / f"newspaper_embeddings_cong_{curr_cong}.npz"
        news_meta_path = OUTPUT_DIR / f"newspaper_meta_cong_{curr_cong}.parquet"
        if not news_emb_path.exists():
            print(f"  No newspaper embeddings for congress {curr_cong}, skipping")
            continue

        news_emb = np.load(news_emb_path)["embeddings"]
        news_meta = pd.read_parquet(news_meta_path)
        print(f"  Newspapers: {len(news_meta):,} articles")

        # ── 5. Compute slant ──────────────────────────────────────
        # Raw projection onto partisan axis
        slant_raw = news_emb @ axis_hat  # (n_articles,)

        # Normalized by partisan gap (G&S analog)
        slant_normalized = slant_raw / gap

        # ── 6. Save ───────────────────────────────────────────────
        result = news_meta.copy()
        result["slant_raw"] = slant_raw.astype(np.float32)
        result["slant_normalized"] = slant_normalized.astype(np.float32)
        result.to_parquet(OUTPUT_DIR / f"article_slant_cong_{curr_cong}.parquet", index=False)

        print(f"  Slant stats (normalized): mean={slant_normalized.mean():.4f}, "
              f"std={slant_normalized.std():.4f}, "
              f"min={slant_normalized.min():.4f}, max={slant_normalized.max():.4f}")

        all_slant.append(result)

    # ── Combined summary ──────────────────────────────────────
    if all_slant:
        combined = pd.concat(all_slant, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "article_slant_all.parquet", index=False)
        print(f"\n{'='*60}")
        print(f"Total articles scored: {len(combined):,}")
        print(f"Saved: article_slant_all.parquet")

        # Per-newspaper average
        paper_avg = combined.groupby("paper")["slant_normalized"].mean().sort_values()
        print(f"\nNewspaper slant ranking (normalized, low=D, high=R):")
        for paper, slant in paper_avg.items():
            print(f"  {slant:+.4f}  {paper}")


if __name__ == "__main__":
    main()
