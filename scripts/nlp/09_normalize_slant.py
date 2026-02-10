"""
09_normalize_slant.py

Normalize raw article-level slant scores using the Gentzkow & Shapiro
(2010) index for cross-year comparability.

Normalization:
  Compute mu_R and mu_D from ALL legislators (not just core) in the
  same rolling window. The partisan gap = mu_R - mu_D.

  net_slant_norm        = net_slant / gap
  right_norm            = right_intensity / gap
  left_norm             = left_intensity / gap
  politicization_norm   = politicization / gap

  All scores are in "partisan gap" units (1 unit = language distance
  between average Republican and average Democrat). No mu_D centering —
  the level shift is absorbed by year fixed effects in regressions.
  The gap also serves as a measure of congressional language polarization.

Rolling-window mapping:
  Model trained on (t-1, t) normalizes newspaper articles in congress t.
  mu_R, mu_D computed from ALL legislators in congresses (t-1, t).

Inputs:
  - data/processed/speeches/models/06_lasso_window_{prev}_{curr}.joblib
  - data/processed/speeches/05_tfidf_matrix.npz
  - data/processed/speeches/05_tfidf_meta.parquet
  - data/processed/newspapers/08_article_slant_cong_{cong}.parquet

Outputs:
  - data/processed/newspapers/09_article_slant_norm_cong_{cong}.parquet
  - data/processed/newspapers/09_normalization_params.csv
"""

import gc
import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

SPEECH_DIR = BASE_DIR / "data" / "processed" / "speeches"
MODEL_DIR = SPEECH_DIR / "models"
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"

SPEECH_TFIDF_PATH = SPEECH_DIR / "05_tfidf_matrix.npz"
SPEECH_META_PATH = SPEECH_DIR / "05_tfidf_meta.parquet"

OUT_DIR = NEWSPAPER_DIR


def compute_raw_scores(X, coef):
    """Compute raw slant decomposition from TF-IDF matrix and coefficients."""
    pos_mask = coef > 0
    neg_mask = coef < 0

    right = X[:, pos_mask].dot(coef[pos_mask])
    left = X[:, neg_mask].dot(np.abs(coef[neg_mask]))

    if hasattr(right, 'A1'):
        right = right.A1
    if hasattr(left, 'A1'):
        left = left.A1

    return right, left


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    congresses = list(range(100, 109))
    windows = [(t - 1, t) for t in congresses]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load speech data for computing legislator benchmarks
    print("Loading speech TF-IDF and metadata ...")
    X_speech = sp.load_npz(SPEECH_TFIDF_PATH)
    speech_meta = pd.read_parquet(SPEECH_META_PATH)
    print(f"  Speeches: {X_speech.shape[0]:,} legislators x {X_speech.shape[1]:,} features")

    # Restrict to intersection columns (from step 06)
    intersection_path = MODEL_DIR / "06_intersection_cols.npy"
    if intersection_path.exists():
        intersection_cols = np.load(intersection_path)
        X_speech = X_speech[:, intersection_cols]
        print(f"  Restricted to intersection: {X_speech.shape[1]:,} features")

    print("\nComputing normalization parameters and normalizing ...\n")

    pipeline_start = time.time()
    norm_params = []

    for i, (cong_prev, cong_curr) in enumerate(windows, 1):
        window_start = time.time()

        # 1. Load model
        model_path = MODEL_DIR / f"06_lasso_window_{cong_prev}_{cong_curr}.joblib"
        if not model_path.exists():
            print(f"  WARNING: {model_path.name} not found, skipping")
            continue
        model = joblib.load(model_path)
        coef = model.coef_[0]

        # 2. Compute mu_R, mu_D from ALL legislators in window
        mask_window = speech_meta["congress_int"].isin([cong_prev, cong_curr])
        mask_rep = mask_window & (speech_meta["party_code"] == "R")
        mask_dem = mask_window & (speech_meta["party_code"] == "D")

        idx_rep = speech_meta.index[mask_rep].values
        idx_dem = speech_meta.index[mask_dem].values

        right_rep, left_rep = compute_raw_scores(X_speech[idx_rep], coef)
        right_dem, left_dem = compute_raw_scores(X_speech[idx_dem], coef)

        mu_R = np.mean(right_rep - left_rep)
        mu_D = np.mean(right_dem - left_dem)
        gap = mu_R - mu_D

        norm_params.append({
            "congress": cong_curr,
            "window": f"{cong_prev}+{cong_curr}",
            "n_rep": len(idx_rep),
            "n_dem": len(idx_dem),
            "mu_R": mu_R,
            "mu_D": mu_D,
            "gap": gap,
        })

        # 3. Load raw article slant (from step 08)
        slant_path = NEWSPAPER_DIR / f"08_article_slant_cong_{cong_curr}.parquet"
        if not slant_path.exists():
            print(f"  WARNING: {slant_path.name} not found, skipping")
            continue
        raw = pd.read_parquet(slant_path)
        n_articles = len(raw)

        # 4. Normalize
        raw["net_slant_norm"] = raw["net_slant"] / gap
        raw["right_norm"] = raw["right_intensity"] / gap
        raw["left_norm"] = raw["left_intensity"] / gap
        raw["politicization_norm"] = raw["politicization"] / gap

        # 5. Save
        out_path = OUT_DIR / f"09_article_slant_norm_cong_{cong_curr}.parquet"
        raw.to_parquet(out_path)

        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / i
        remaining = avg_per * (len(windows) - i)

        print(f"  [{i}/{len(windows)}] Congress {cong_curr}: "
              f"{n_articles:,} articles  |  "
              f"mu_R={mu_R:>7.3f}  mu_D={mu_D:>7.3f}  gap={gap:>6.3f}  |  "
              f"norm_slant mean={raw['net_slant_norm'].mean():.3f} "
              f"std={raw['net_slant_norm'].std():.3f}  |  "
              f"{elapsed:.1f}s  ETA: {remaining:.0f}s")

        del model, coef, raw
        gc.collect()

    # Save normalization parameters
    norm_df = pd.DataFrame(norm_params)
    norm_df.to_csv(OUT_DIR / "09_normalization_params.csv", index=False)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Normalization parameters (G&S 2010 index)")
    print("=" * 72)

    print(f"\n  {'Congress':>8}  {'Window':>10}  {'N Rep':>6}  {'N Dem':>6}  "
          f"{'mu_R':>8}  {'mu_D':>8}  {'Gap':>8}")
    for p in norm_params:
        print(f"  {p['congress']:>8}  {p['window']:>10}  {p['n_rep']:>6}  {p['n_dem']:>6}  "
              f"{p['mu_R']:>8.3f}  {p['mu_D']:>8.3f}  {p['gap']:>8.3f}")

    avg_gap = np.mean([p["gap"] for p in norm_params])
    print(f"\n  Average partisan gap: {avg_gap:.3f}")
    print(f"  Interpretation: 0 = avg Democrat, 1 = avg Republican")

    total_time = time.time() - pipeline_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Saved to -> {OUT_DIR}")
    print("=" * 72)
