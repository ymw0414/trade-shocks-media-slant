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
  Model trained on congress t normalizes newspaper articles in congress t.
  mu_R, mu_D computed from ALL legislators in congress(es) in the window.

Inputs:
  - models/06_lasso_window_{cong}.joblib
  - speeches/05_tfidf_matrix.npz
  - speeches/05_tfidf_meta.parquet
  - newspapers/08_article_slant_cong_{cong}.parquet

Outputs:
  - newspapers/09_article_slant_norm_cong_{cong}.parquet
  - newspapers/09_normalization_params.csv
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
# Paths (from pipeline_config — change RUN_NAME there for new runs)
# ------------------------------------------------------------------
import pipeline_config as cfg

MODEL_DIR         = cfg.MODEL_DIR
NEWSPAPER_DIR     = cfg.NEWS_DIR
SPEECH_MATRIX_PATH = cfg.INPUT_SPEECH_DIR / "05_tfidf_matrix.npz"
SPEECH_META_PATH  = cfg.INPUT_SPEECH_DIR / "05_tfidf_meta.parquet"
OUT_DIR           = cfg.NEWS_DIR


def compute_raw_scores(X, coef):
    """Compute raw slant decomposition from feature matrix and coefficients."""
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
    windows = cfg.get_windows()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load speech data for computing legislator benchmarks
    print("Loading speech feature matrix and metadata ...")
    X_speech = sp.load_npz(SPEECH_MATRIX_PATH)
    speech_meta = pd.read_parquet(SPEECH_META_PATH)

    # Apply shared vocabulary mask if step 06 produced one
    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    if shared_vocab_path.exists():
        shared_vocab_mask = np.load(shared_vocab_path)
        X_speech = X_speech[:, shared_vocab_mask]
        print(f"  Applied shared vocabulary mask: {int(shared_vocab_mask.sum()):,} / {len(shared_vocab_mask):,} features")

    # Detect party column (matches step 05 output)
    PARTY_COL = "party" if "party" in speech_meta.columns else "party_code"

    print(f"  Speeches: {X_speech.shape[0]:,} docs x {X_speech.shape[1]:,} features")
    print(f"  Party column: '{PARTY_COL}'")

    NORM_METHOD = cfg.CONFIG.get("norm_method", "raw_gap")
    print(f"\nNormalization method: {NORM_METHOD}")
    if NORM_METHOD == "prob_direct":
        print("  -> Using P(R) directly as net_slant_norm (no gap division)")
    print("Computing normalization parameters and normalizing ...\n")

    pipeline_start = time.time()
    norm_params = []

    for i, window_congs in enumerate(windows, 1):
        cong_curr = window_congs[-1]
        window_start = time.time()

        # 1. Load model
        window_file = "_".join(str(c) for c in window_congs)
        model_path = MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
        if not model_path.exists():
            print(f"  WARNING: {model_path.name} not found, skipping")
            continue
        model = joblib.load(model_path)
        coef = model.coef_[0]

        # 2. Compute mu_R, mu_D from ALL legislators in window
        mask_window = speech_meta["congress_int"].isin(window_congs)
        mask_rep = mask_window & ((speech_meta[PARTY_COL] == "R") | (speech_meta[PARTY_COL] == 200))
        mask_dem = mask_window & ((speech_meta[PARTY_COL] == "D") | (speech_meta[PARTY_COL] == 100))

        idx_rep = speech_meta.index[mask_rep].values
        idx_dem = speech_meta.index[mask_dem].values

        if NORM_METHOD in ("prob_gap", "prob_direct"):
            # Probability-based: E[P(R)|R] and E[P(R)|D]
            mu_R = float(np.mean(model.predict_proba(X_speech[idx_rep])[:, 1]))
            mu_D = float(np.mean(model.predict_proba(X_speech[idx_dem])[:, 1]))
        else:
            # Raw score gap: E[X@coef|R] - E[X@coef|D]
            right_rep, left_rep = compute_raw_scores(X_speech[idx_rep], coef)
            right_dem, left_dem = compute_raw_scores(X_speech[idx_dem], coef)
            mu_R = np.mean(right_rep - left_rep)
            mu_D = np.mean(right_dem - left_dem)

        gap = mu_R - mu_D

        window_label = "+".join(str(c) for c in window_congs)
        norm_params.append({
            "congress": cong_curr,
            "window": window_label,
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
        if NORM_METHOD == "prob_direct":
            # Use P(R) directly — bounded [0,1], cross-year comparable
            raw["net_slant_norm"] = raw["prob_R"]
            raw["right_norm"] = raw["right_intensity"]
            raw["left_norm"] = raw["left_intensity"]
            raw["politicization_norm"] = raw["politicization"]
        else:
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
