"""
06_train_lasso.py

Train rolling-window LASSO logistic regressions to learn partisan
brand language from congressional speeches.

Rolling-window rule:
  To score newspaper articles during Congress t, train on Congress (t-1) + t.
  -> 9 windows: (99+100), (100+101), ..., (107+108)

Model design:
  Binary classification: Republican Partisan Core (1) vs Democratic Partisan Core (0).
  Non-core legislators are excluded from training.
  This learns the language that best distinguishes each party's brand.

Vocabulary restriction (Widmer et al. 2025):
  Before training, the feature space is restricted to the INTERSECTION of
  congressional speech vocabulary and newspaper vocabulary.  This ensures
  LASSO selects only phrases that are (a) partisan in Congress AND (b) actually
  observable in newspaper text.  Procedural language, legislator-specific
  references, and other congressional jargon that never appears in newspapers
  are automatically excluded.

At inference time, the learned coefficients can be decomposed:
  - Positive coefficients  -> Right Intensity
  - Negative coefficients  -> Left Intensity
  - Net Slant              = Right - Left
  - Politicization         = Right + Left
  This distinguishes "neutral" articles from "polarized-but-balanced" ones.

Outputs per window:
  - Trained model (LogisticRegressionCV with L1 penalty)
  - Intersection column indices (06_intersection_cols.npy)
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

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PROC_DIR = BASE_DIR / "data" / "processed" / "speeches"
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"

TFIDF_PATH = PROC_DIR / "05_tfidf_matrix.npz"
META_PATH = PROC_DIR / "05_tfidf_meta.parquet"

OUT_DIR = PROC_DIR / "models"

# ------------------------------------------------------------------
# 1. Load TF-IDF matrix and metadata
# ------------------------------------------------------------------
print("Loading TF-IDF matrix ...")
X_all = sp.load_npz(TFIDF_PATH)

print("Loading metadata ...")
meta = pd.read_parquet(META_PATH)

n_full_features = X_all.shape[1]
print(f"  Matrix: {X_all.shape[0]:,} docs x {n_full_features:,} features")

# ------------------------------------------------------------------
# 1b. Compute intersection with newspaper vocabulary
# ------------------------------------------------------------------
print("\nComputing speech-newspaper vocabulary intersection ...")

# Find columns that have any non-zero value in ANY newspaper TF-IDF matrix
newspaper_feature_mask = np.zeros(n_full_features, dtype=bool)
for cong in range(100, 109):
    tfidf_path = NEWSPAPER_DIR / f"07_newspaper_tfidf_cong_{cong}.npz"
    if not tfidf_path.exists():
        print(f"  WARNING: {tfidf_path.name} not found, skipping")
        continue
    nX = sp.load_npz(tfidf_path)
    # For CSR matrix, find columns with any non-zero entry
    col_nnz = np.diff(nX.tocsc().indptr)  # number of non-zeros per column
    newspaper_feature_mask |= (col_nnz > 0)
    n_active = newspaper_feature_mask.sum()
    print(f"  Congress {cong}: {nX.shape[0]:,} articles, "
          f"cumulative newspaper features: {n_active:,}")
    del nX
    import gc; gc.collect()

# Also find columns with any non-zero value in speech TF-IDF
speech_col_nnz = np.diff(X_all.tocsc().indptr)
speech_feature_mask = (speech_col_nnz > 0)

# Intersection: features that appear in BOTH corpora
intersection_mask = newspaper_feature_mask & speech_feature_mask
intersection_cols = np.where(intersection_mask)[0]
n_intersection = len(intersection_cols)

print(f"\n  Speech features (non-zero): {speech_feature_mask.sum():,}")
print(f"  Newspaper features (non-zero): {newspaper_feature_mask.sum():,}")
print(f"  Intersection features: {n_intersection:,}")
print(f"  Dropped (speech-only): {speech_feature_mask.sum() - n_intersection:,}")

# Save intersection column indices for use in step 08
np.save(OUT_DIR / "06_intersection_cols.npy", intersection_cols)

# Restrict speech TF-IDF to intersection columns
X_all = X_all[:, intersection_cols]
print(f"  Restricted matrix: {X_all.shape[0]:,} docs x {X_all.shape[1]:,} features")

# ------------------------------------------------------------------
# 2. Define rolling windows
# ------------------------------------------------------------------
WINDOWS = [(t - 1, t) for t in range(100, 109)]  # (99,100) ... (107,108)

# ------------------------------------------------------------------
# 3. Train one model per window
# ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = []

total_windows = len(WINDOWS)
pipeline_start = time.time()

for wi, (cong_prev, cong_curr) in enumerate(WINDOWS, 1):
    window_start = time.time()
    print(f"\n[{wi}/{total_windows}] Window: Congress {cong_prev} + {cong_curr}")

    # Select rows for this window
    mask_window = meta["congress_int"].isin([cong_prev, cong_curr])

    # Keep only partisan core members (drop non-core)
    mask_rep = meta["label_rep_core"] == 1
    mask_dem = meta["label_dem_core"] == 1
    mask_core = mask_window & (mask_rep | mask_dem)

    idx = meta.index[mask_core].values
    X = X_all[idx]
    y = meta.loc[idx, "label_rep_core"].values  # 1=Rep core, 0=Dem core

    n_rep = int(y.sum())
    n_dem = int(len(y) - y.sum())
    print(f"  Training set: {len(y)} legislators ({n_rep} Rep core, {n_dem} Dem core)")

    # Train LASSO logistic regression with cross-validation
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

    # Results
    best_C = model.C_[0]
    train_acc = model.score(X, y)
    # Mean CV accuracy for the best C across folds
    best_c_idx = np.where(model.Cs_ == best_C)[0][0]
    cv_scores = model.scores_[1][:, best_c_idx]  # class 1, all folds, best C
    cv_acc = cv_scores.mean()
    n_nonzero = int(np.sum(model.coef_[0] != 0))
    n_positive = int(np.sum(model.coef_[0] > 0))
    n_negative = int(np.sum(model.coef_[0] < 0))

    print(f"  Best C: {best_C:.4f}")
    print(f"  Train accuracy: {train_acc:.3f}  |  CV accuracy: {cv_acc:.3f}")
    print(f"  Non-zero coefs: {n_nonzero:,}  (+{n_positive:,} / -{n_negative:,})")

    # Save model
    model_path = OUT_DIR / f"06_lasso_window_{cong_prev}_{cong_curr}.joblib"
    joblib.dump(model, model_path)

    elapsed = time.time() - window_start
    total_elapsed = time.time() - pipeline_start
    avg_per_window = total_elapsed / wi
    remaining = avg_per_window * (total_windows - wi)
    print(f"  Time: {elapsed:.1f}s  |  Elapsed: {total_elapsed:.0f}s  |  ETA: {remaining:.0f}s")

    results.append({
        "window": f"{cong_prev}+{cong_curr}",
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
# 4. Summary
# ------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY: All rolling-window models")
print("=" * 72)

summary = pd.DataFrame(results)
print(summary.to_string(index=False))
print("=" * 72)

# Save summary
summary.to_csv(OUT_DIR / "06_training_summary.csv", index=False)
print(f"\nSaved models -> {OUT_DIR}")
