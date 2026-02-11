"""
06_train_lasso_3window.py

ROBUSTNESS: Train LASSO with 3-congress rolling windows instead of 2.

Windows: (98,99,100), (99,100,101), ..., (106,107,108)
More training data per window should stabilize coefficient selection,
particularly for the 102+103 window which had sparse coefficients.

Computes a fresh speech-newspaper vocabulary intersection from the
current TF-IDF files (avoids stale intersection column indices).

Outputs:
  - data/processed/speeches/models/06_lasso_3w_{w1}_{w2}_{w3}.joblib
  - data/processed/speeches/models/06_intersection_cols_3w.npy
  - data/processed/speeches/models/06_training_summary_3w.csv
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings("ignore", category=UserWarning)

import gc

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PROC_DIR = BASE_DIR / "data" / "processed" / "speeches"
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"
OUT_DIR = PROC_DIR / "models"

TFIDF_PATH = PROC_DIR / "05_tfidf_matrix.npz"
META_PATH = PROC_DIR / "05_tfidf_meta.parquet"

MIN_NEWSPAPER_DF = 100

# ------------------------------------------------------------------
# 1. Load TF-IDF matrix and metadata
# ------------------------------------------------------------------
print("Loading TF-IDF matrix ...")
X_all = sp.load_npz(TFIDF_PATH)

print("Loading metadata ...")
meta = pd.read_parquet(META_PATH)

n_full_features = X_all.shape[1]
print(f"  Matrix: {X_all.shape[0]:,} docs x {n_full_features:,} features")
print(f"  Congresses: {sorted(meta['congress_int'].unique())}")

# ------------------------------------------------------------------
# 2. Compute fresh speech-newspaper vocabulary intersection
# ------------------------------------------------------------------
print("\nComputing speech-newspaper vocabulary intersection ...")

newspaper_doc_count = np.zeros(n_full_features, dtype=np.int64)
for cong in range(100, 109):
    tfidf_path = NEWSPAPER_DIR / f"07_newspaper_tfidf_cong_{cong}.npz"
    if not tfidf_path.exists():
        print(f"  WARNING: {tfidf_path.name} not found, skipping")
        continue
    nX = sp.load_npz(tfidf_path)
    assert nX.shape[1] == n_full_features, \
        f"Newspaper TF-IDF cols ({nX.shape[1]}) != speech TF-IDF cols ({n_full_features})"
    col_nnz = np.diff(nX.tocsc().indptr)
    newspaper_doc_count += col_nnz
    print(f"  Congress {cong}: {nX.shape[0]:,} articles")
    del nX
    gc.collect()

newspaper_feature_mask = (newspaper_doc_count >= MIN_NEWSPAPER_DF)
speech_col_nnz = np.diff(X_all.tocsc().indptr)
speech_feature_mask = (speech_col_nnz > 0)

intersection_mask = newspaper_feature_mask & speech_feature_mask
intersection_cols = np.where(intersection_mask)[0]
n_intersection = len(intersection_cols)

print(f"  Speech features (non-zero): {speech_feature_mask.sum():,}")
print(f"  Newspaper features (DF >= {MIN_NEWSPAPER_DF}): {newspaper_feature_mask.sum():,}")
print(f"  Intersection: {n_intersection:,}")

np.save(OUT_DIR / "06_intersection_cols_3w.npy", intersection_cols)

X_all = X_all[:, intersection_cols]
print(f"  Restricted matrix: {X_all.shape[0]:,} docs x {X_all.shape[1]:,} features")

# ------------------------------------------------------------------
# 3. Define 3-congress rolling windows
# ------------------------------------------------------------------
WINDOWS = [(t - 2, t - 1, t) for t in range(100, 109)]

# ------------------------------------------------------------------
# 4. Train one model per window
# ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = []

total_windows = len(WINDOWS)
pipeline_start = time.time()

for wi, window_congs in enumerate(WINDOWS, 1):
    window_start = time.time()
    cong_curr = window_congs[-1]
    window_label = "+".join(str(c) for c in window_congs)
    print(f"\n[{wi}/{total_windows}] Window: Congress {window_label}")

    # Select rows for this window (partisan core only)
    mask_window = meta["congress_int"].isin(window_congs)
    mask_rep = meta["label_rep_core"] == 1
    mask_dem = meta["label_dem_core"] == 1
    mask_core = mask_window & (mask_rep | mask_dem)

    idx = meta.index[mask_core].values
    X = X_all[idx]
    y = meta.loc[idx, "label_rep_core"].values

    n_rep = int(y.sum())
    n_dem = int(len(y) - y.sum())
    print(f"  Training set: {len(y)} legislators ({n_rep} Rep core, {n_dem} Dem core)")

    model = LogisticRegressionCV(
        penalty="l1",
        solver="liblinear",
        Cs=np.logspace(-1, 4, 20),
        cv=5,
        max_iter=5000,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X, y)

    best_C = model.C_[0]
    train_acc = model.score(X, y)
    best_c_idx = np.where(model.Cs_ == best_C)[0][0]
    cv_scores = model.scores_[1][:, best_c_idx]
    cv_acc = cv_scores.mean()
    n_nonzero = int(np.sum(model.coef_[0] != 0))
    n_positive = int(np.sum(model.coef_[0] > 0))
    n_negative = int(np.sum(model.coef_[0] < 0))

    print(f"  Best C: {best_C:.4f}")
    print(f"  Train accuracy: {train_acc:.3f}  |  CV accuracy: {cv_acc:.3f}")
    print(f"  Non-zero coefs: {n_nonzero:,}  (+{n_positive:,} / -{n_negative:,})")

    window_file = "_".join(str(c) for c in window_congs)
    model_path = OUT_DIR / f"06_lasso_3w_{window_file}.joblib"
    joblib.dump(model, model_path)

    elapsed = time.time() - window_start
    total_elapsed = time.time() - pipeline_start
    avg_per_window = total_elapsed / wi
    remaining = avg_per_window * (total_windows - wi)
    print(f"  Time: {elapsed:.1f}s  |  Elapsed: {total_elapsed:.0f}s  |  ETA: {remaining:.0f}s")

    results.append({
        "window": window_label,
        "scores_congress": cong_curr,
        "n_rep_core": n_rep,
        "n_dem_core": n_dem,
        "best_C": best_C,
        "train_accuracy": train_acc,
        "cv_accuracy": cv_acc,
        "n_nonzero_coefs": n_nonzero,
        "n_positive_coefs": n_positive,
        "n_negative_coefs": n_negative,
    })

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY: 3-congress window models")
print("=" * 72)

summary = pd.DataFrame(results)
print(summary.to_string(index=False))
print("=" * 72)

summary.to_csv(OUT_DIR / "06_training_summary_3w.csv", index=False)
print(f"\nSaved models -> {OUT_DIR}")
