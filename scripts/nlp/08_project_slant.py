"""
08_project_slant.py

Project LASSO coefficients onto newspaper TF-IDF matrices to compute
article-level slant scores.

No predict_proba — the model was trained on partisan-core legislators
(extreme ~20%), so probability outputs are conceptually meaningless for
newspaper articles. Instead, we directly decompose the linear score:

  right_intensity  = X[:, pos] @ coef[pos]      (Republican language usage)
  left_intensity   = X[:, neg] @ |coef[neg]|     (Democratic language usage)
  net_slant        = right - left
  politicization   = right + left

Note: X is a TF-IDF matrix (sublinear_tf, L2-normed), not raw counts.
This differs from G&S's multinomial model but controls for document length
and word frequency effects.

Rolling-window mapping:
  Model trained on (t-1, t) scores newspaper articles in congress t.

Inputs:
  - data/processed/speeches/models/06_lasso_window_{prev}_{curr}.joblib
  - data/processed/newspapers/07_newspaper_tfidf_cong_{cong}.npz

Outputs (per congress):
  - data/processed/newspapers/08_article_slant_cong_{cong}.parquet
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

MODEL_DIR = BASE_DIR / "data" / "processed" / "speeches" / "models"
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"

OUT_DIR = NEWSPAPER_DIR


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    congresses = list(range(100, 109))
    windows = [(t - 1, t) for t in congresses]  # (99,100) ... (107,108)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load intersection column indices (from step 06)
    intersection_path = MODEL_DIR / "06_intersection_cols.npy"
    if intersection_path.exists():
        intersection_cols = np.load(intersection_path)
        print(f"Loaded intersection columns: {len(intersection_cols):,} features")
        use_intersection = True
    else:
        print("WARNING: 06_intersection_cols.npy not found, using full feature space")
        use_intersection = False

    print("Projecting LASSO coefficients onto newspaper TF-IDF ...\n")

    pipeline_start = time.time()
    summary = []

    for i, (cong_prev, cong_curr) in enumerate(windows, 1):
        window_start = time.time()

        # 1. Load model
        model_path = MODEL_DIR / f"06_lasso_window_{cong_prev}_{cong_curr}.joblib"
        if not model_path.exists():
            print(f"  WARNING: {model_path.name} not found, skipping")
            continue
        model = joblib.load(model_path)
        coef = model.coef_[0]  # shape (n_intersection_features,)

        pos_mask = coef > 0
        neg_mask = coef < 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        # 2. Load newspaper TF-IDF (restrict to intersection columns)
        tfidf_path = NEWSPAPER_DIR / f"07_newspaper_tfidf_cong_{cong_curr}.npz"
        if not tfidf_path.exists():
            print(f"  WARNING: {tfidf_path.name} not found, skipping")
            continue
        X = sp.load_npz(tfidf_path)
        n_articles = X.shape[0]

        if use_intersection:
            X = X[:, intersection_cols]

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

        # 4. Save
        df = pd.DataFrame({
            "right_intensity": right_intensity,
            "left_intensity": left_intensity,
            "net_slant": net_slant,
            "politicization": politicization,
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
        }
        summary.append(stats)

        print(f"  [{i}/{len(windows)}] Congress {cong_curr}: "
              f"{n_articles:,} articles  |  "
              f"coefs +{n_pos}/-{n_neg}  |  "
              f"net_slant mean={stats['net_slant_mean']:.4f} std={stats['net_slant_std']:.4f}  |  "
              f"{elapsed:.1f}s  ETA: {remaining:.0f}s")

        del model, coef, X, right_intensity, left_intensity, net_slant, politicization, df
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
