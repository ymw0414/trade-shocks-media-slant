"""
08_project_slant.py

Project LASSO coefficients onto newspaper feature matrices to compute
article-level slant scores.

Scoring formula (Widmer methodology):
  Speech_Partisanship_n = sum(f_bn * phi_b)
  where f_bn = relative bigram frequency, phi_b = LASSO coefficient.

Decomposition (our addition):
  right_intensity  = X[:, pos] @ coef[pos]      (Republican language usage)
  left_intensity   = X[:, neg] @ |coef[neg]|     (Democratic language usage)
  net_slant        = right - left
  politicization   = right + left

Rolling-window mapping:
  Model trained on congress t scores newspaper articles in congress t.

Inputs:
  - models/06_lasso_window_{cong}.joblib
  - newspapers/07_newspaper_features_cong_{cong}.npz

Outputs (per congress):
  - newspapers/08_article_slant_cong_{cong}.parquet
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

MODEL_DIR     = cfg.MODEL_DIR
NEWSPAPER_DIR = cfg.INPUT_NEWS_DIR   # read step 07 outputs (may be shared)
OUT_DIR       = cfg.NEWS_DIR         # write step 08 outputs (run-specific)


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    windows = cfg.get_windows()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared vocabulary mask if step 06 produced one
    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = None
    if shared_vocab_path.exists():
        shared_vocab_mask = np.load(shared_vocab_path)
        n_keep = int(shared_vocab_mask.sum())
        print(f"Loaded shared vocabulary mask: {n_keep:,} / {len(shared_vocab_mask):,} features\n")

    print("Projecting LASSO coefficients onto newspaper features ...\n")

    pipeline_start = time.time()
    summary = []

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
        coef = model.coef_[0]  # shape (n_features,)

        pos_mask = coef > 0
        neg_mask = coef < 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        # 2. Load newspaper feature matrix
        feat_path = NEWSPAPER_DIR / f"07_newspaper_features_cong_{cong_curr}.npz"
        if not feat_path.exists():
            print(f"  WARNING: {feat_path.name} not found, skipping")
            continue
        X = sp.load_npz(feat_path)
        if shared_vocab_mask is not None:
            X = X[:, shared_vocab_mask]
        n_articles = X.shape[0]

        # 3. Compute scores
        right_intensity = X[:, pos_mask].dot(coef[pos_mask])
        left_intensity = X[:, neg_mask].dot(np.abs(coef[neg_mask]))

        # Ensure dense 1D arrays
        if hasattr(right_intensity, 'A1'):
            right_intensity = right_intensity.A1
        if hasattr(left_intensity, 'A1'):
            left_intensity = left_intensity.A1

        net_slant = right_intensity - left_intensity
        politicization = right_intensity + left_intensity

        # 3b. Predicted probability P(Republican) — bounded [0,1], no normalization needed
        prob_R = model.predict_proba(X)[:, 1]

        # 4. Save
        df = pd.DataFrame({
            "right_intensity": right_intensity,
            "left_intensity": left_intensity,
            "net_slant": net_slant,
            "politicization": politicization,
            "prob_R": prob_R,
        })

        out_path = OUT_DIR / f"08_article_slant_cong_{cong_curr}.parquet"
        df.to_parquet(out_path)

        # Stats
        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / i
        remaining = avg_per * (len(windows) - i)

        stats = {
            "congress": cong_curr,
            "n_articles": n_articles,
            "n_pos_coefs": int(n_pos),
            "n_neg_coefs": int(n_neg),
            "net_slant_mean": np.mean(net_slant),
            "net_slant_std": np.std(net_slant),
            "politicization_mean": np.mean(politicization),
            "prob_R_mean": np.mean(prob_R),
            "prob_R_std": np.std(prob_R),
        }
        summary.append(stats)

        print(f"  [{i}/{len(windows)}] Congress {cong_curr}: "
              f"{n_articles:,} articles  |  "
              f"coefs +{n_pos}/-{n_neg}  |  "
              f"net_slant mean={stats['net_slant_mean']:.4f} std={stats['net_slant_std']:.4f}  |  "
              f"{elapsed:.1f}s  ETA: {remaining:.0f}s")

        del model, coef, X, right_intensity, left_intensity, net_slant, politicization, prob_R, df
        gc.collect()

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY: Article-level slant projection")
    print("=" * 72)

    total_articles = sum(s["n_articles"] for s in summary)
    print(f"\n  Total articles scored: {total_articles:,}")

    print(f"\n  {'Congress':>8}  {'Articles':>10}  {'Coefs +/-':>12}  "
          f"{'Net Slant Mean':>15}  {'Net Slant Std':>14}  {'Politic Mean':>13}")
    for s in summary:
        print(f"  {s['congress']:>8}  {s['n_articles']:>10,}  "
              f"+{s['n_pos_coefs']}/{s['n_neg_coefs']:>5}  "
              f"{s['net_slant_mean']:>15.4f}  {s['net_slant_std']:>14.4f}  "
              f"{s['politicization_mean']:>13.4f}")

    total_time = time.time() - pipeline_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Saved to -> {OUT_DIR}")
    print("=" * 72)
