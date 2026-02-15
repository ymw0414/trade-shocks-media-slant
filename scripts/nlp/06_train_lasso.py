"""
06_train_lasso.py

Train rolling-window LASSO logistic regressions to learn partisan
language from congressional speeches.

Widmer methodology (default):
  - Train on ALL Republican vs Democrat speeches (not just partisan core)
  - BIC for lambda/C selection: BIC = -2*LL + k*log(n)
    Selects regularization that balances fit and sparsity, yielding
    more interpretable partisan features.
  - 1-congress rolling windows: train on Congress t, score Congress t
  - No newspaper vocabulary intersection (handled by step 05 freq filter)

Legacy mode (optional, via pipeline_config):
  - Train on partisan core only (top/bottom 20% Nokken-Poole)
  - 5-fold CV for lambda selection
  - Multi-congress rolling windows

At inference time (step 08), the learned coefficients are decomposed:
  - Positive coefficients  -> Right Intensity (Republican language)
  - Negative coefficients  -> Left Intensity (Democratic language)
  - Net Slant              = Right - Left
  - Politicization         = Right + Left

Outputs per window:
  - Trained LogisticRegression model (.joblib)
  - 06_training_summary.csv
  - 06_top_partisan_phrases.csv (diagnostic)
"""

import gc
import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure text_analyzer importable (needed for vectorizer unpickling)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import text_analyzer  # noqa: F401

import pipeline_config as cfg

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
MATRIX_PATH     = cfg.INPUT_SPEECH_DIR / "05_tfidf_matrix.npz"
META_PATH       = cfg.INPUT_SPEECH_DIR / "05_tfidf_meta.parquet"
VECTORIZER_PATH = cfg.INPUT_SPEECH_DIR / "05_tfidf_vectorizer.joblib"
OUT_DIR         = cfg.MODEL_DIR

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
print("Loading feature matrix ...")
X_all = sp.load_npz(MATRIX_PATH)

print("Loading metadata ...")
meta = pd.read_parquet(META_PATH)

# Detect party column (depends on aggregation mode in step 05)
PARTY_COL = "party" if "party" in meta.columns else "party_code"

print(f"  Matrix: {X_all.shape[0]:,} docs x {X_all.shape[1]:,} features")
print(f"  Party column: '{PARTY_COL}'")

# Load feature names for diagnostics
print("Loading vectorizer for feature names ...")
vec = joblib.load(VECTORIZER_PATH)
n_features = X_all.shape[1]
feature_names = np.empty(n_features, dtype=object)
for feat, idx in vec.vocabulary_.items():
    if idx < n_features:
        feature_names[idx] = feat
del vec

# ------------------------------------------------------------------
# 1b. Shared vocabulary filter (remove phrases absent from newspapers)
# ------------------------------------------------------------------
SHARED_VOCAB_MIN_DF = cfg.CONFIG.get("shared_vocab_min_df")
shared_vocab_mask = None  # will be set if filter is active

if SHARED_VOCAB_MIN_DF is not None:
    print(f"\nApplying shared vocabulary filter (min newspaper DF = {SHARED_VOCAB_MIN_DF:.4%}) ...")
    news_dir = cfg.INPUT_NEWS_DIR
    total_docs = 0
    total_df = np.zeros(X_all.shape[1])

    for cong in cfg.get_congresses():
        path = news_dir / f"07_newspaper_tfidf_cong_{cong}.npz"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping")
            continue
        X_news = sp.load_npz(path)
        total_docs += X_news.shape[0]
        total_df += np.asarray((X_news > 0).sum(axis=0)).ravel()
        del X_news
        gc.collect()

    relative_df = total_df / total_docs
    shared_vocab_mask = relative_df >= SHARED_VOCAB_MIN_DF

    n_before = len(shared_vocab_mask)
    n_after = int(shared_vocab_mask.sum())
    print(f"  Newspaper articles scanned: {total_docs:,}")
    print(f"  Features passing filter: {n_after:,} / {n_before:,} "
          f"({n_after/n_before*100:.1f}%)")

    # Subset columns
    X_all = X_all[:, shared_vocab_mask]
    feature_names = feature_names[shared_vocab_mask]
    n_features = X_all.shape[1]
    print(f"  Filtered matrix: {X_all.shape[0]:,} docs x {n_features:,} features")

    # Save mask for step 08
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "06_shared_vocab_mask.npy", shared_vocab_mask)
    print(f"  Saved mask -> {OUT_DIR / '06_shared_vocab_mask.npy'}")

# ------------------------------------------------------------------
# 2. Configuration
# ------------------------------------------------------------------
WINDOWS   = cfg.get_windows()
Cs        = np.logspace(*cfg.CONFIG["lasso_Cs"])
USE_BIC   = cfg.CONFIG.get("lasso_lambda_selection", "bic") == "bic"
CORE_ONLY = cfg.CONFIG.get("partisan_core_only", False)
MAX_ITER  = cfg.CONFIG["lasso_max_iter"]
CV_FOLDS  = cfg.CONFIG["lasso_cv_folds"]

lambda_method = "BIC" if USE_BIC else f"{CV_FOLDS}-fold CV"
train_desc = "partisan core only" if CORE_ONLY else "all R vs D"
win_size = cfg.CONFIG["window_size"]

print(f"\nSettings:")
print(f"  Lambda selection: {lambda_method}")
print(f"  Training set:     {train_desc}")
print(f"  Windows:          {len(WINDOWS)} ({win_size}-congress)")
print(f"  C grid:           {len(Cs)} values [{Cs[0]:.4f} ... {Cs[-1]:.1f}]")
print(f"  Max iterations:   {MAX_ITER}")

# ------------------------------------------------------------------
# 3. Train one model per window
# ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = []
all_top_phrases = []

pipeline_start = time.time()

for wi, window_congs in enumerate(WINDOWS, 1):
    window_start = time.time()
    cong_curr = window_congs[-1]
    window_label = "+".join(str(c) for c in window_congs)
    print(f"\n{'=' * 60}")
    print(f"[{wi}/{len(WINDOWS)}] Window: Congress {window_label}")
    print(f"{'=' * 60}")

    # Select training rows
    mask_window = meta["congress_int"].isin(window_congs)

    if CORE_ONLY:
        mask_rep = meta["label_rep_core"] == 1
        mask_dem = meta["label_dem_core"] == 1
        mask_train = mask_window & (mask_rep | mask_dem)
        idx = meta.index[mask_train].values
        y = meta.loc[idx, "label_rep_core"].values.astype(int)
    else:
        mask_train = mask_window
        idx = meta.index[mask_train].values
        party_vals = meta.loc[idx, PARTY_COL].values
        y = ((party_vals == "R") | (party_vals == 200)).astype(int)

    X = X_all[idx]
    n_rep = int(y.sum())
    n_dem = len(y) - n_rep
    print(f"  Training: {len(y):,} speeches ({n_rep:,} Rep, {n_dem:,} Dem)")

    if USE_BIC:
        # ---- BIC selection with early stopping ----
        best_bic = np.inf
        best_model = None
        best_C = None
        bic_log = []
        bic_increasing_streak = 0
        BIC_EARLY_STOP = 5  # stop after 5 consecutive BIC increases

        for C in Cs:
            m = LogisticRegression(
                penalty="l1", solver="liblinear", C=C,
                max_iter=MAX_ITER, random_state=42,
            )
            m.fit(X, y)

            # Log-likelihood (numerically stable)
            proba = m.predict_proba(X)
            eps = 1e-15
            ll = np.sum(
                y * np.log(np.clip(proba[:, 1], eps, 1 - eps)) +
                (1 - y) * np.log(np.clip(proba[:, 0], eps, 1 - eps))
            )

            # BIC: k = non-zero coefs + intercept
            k = int(np.sum(m.coef_[0] != 0)) + 1
            n = len(y)
            bic = -2 * ll + k * np.log(n)
            acc = m.score(X, y)

            bic_log.append({"C": C, "BIC": bic, "k": k, "accuracy": acc})

            if bic < best_bic:
                best_bic = bic
                best_model = m
                best_C = C
                bic_increasing_streak = 0
            else:
                bic_increasing_streak += 1
                if bic_increasing_streak >= BIC_EARLY_STOP:
                    break

        model = best_model

        # Print BIC curve
        print(f"\n  BIC selection (best C = {best_C:.4f}, BIC = {best_bic:.1f}):")
        for entry in bic_log:
            marker = " <<<" if entry["C"] == best_C else ""
            print(f"    C={entry['C']:>10.4f}  BIC={entry['BIC']:>12.1f}  "
                  f"k={entry['k']:>5}  acc={entry['accuracy']:.3f}{marker}")

    else:
        # ---- CV selection ----
        model = LogisticRegressionCV(
            penalty="l1", solver="liblinear",
            Cs=Cs, cv=CV_FOLDS,
            max_iter=MAX_ITER, scoring="accuracy",
            random_state=42, n_jobs=int(os.environ.get("LASSO_N_JOBS", -1)),
        )
        model.fit(X, y)
        best_C = model.C_[0]

        # CV accuracy for the best C
        best_c_idx = np.where(model.Cs_ == best_C)[0][0]
        cv_scores = model.scores_[1][:, best_c_idx]
        cv_acc = cv_scores.mean()
        print(f"  CV accuracy: {cv_acc:.3f}")

    # ---- Results ----
    coef = model.coef_[0]
    train_acc = model.score(X, y)
    n_nonzero = int(np.sum(coef != 0))
    n_positive = int(np.sum(coef > 0))
    n_negative = int(np.sum(coef < 0))

    print(f"\n  Best C:         {best_C:.4f}")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Non-zero coefs: {n_nonzero:,} (+{n_positive:,} / -{n_negative:,})")

    # ---- Top partisan phrases (diagnostic) ----
    # Save top 15 per direction (R/D separately) so reports always have 10+
    top_per_dir = 15
    if n_nonzero > 0:
        pos_idx = np.where(coef > 0)[0]
        neg_idx = np.where(coef < 0)[0]

        # Top Republican (positive coefs, descending)
        if len(pos_idx) > 0:
            pos_sorted = pos_idx[np.argsort(coef[pos_idx])[::-1][:top_per_dir]]
        else:
            pos_sorted = np.array([], dtype=int)
        # Top Democratic (negative coefs, ascending = largest absolute)
        if len(neg_idx) > 0:
            neg_sorted = neg_idx[np.argsort(coef[neg_idx])[:top_per_dir]]
        else:
            neg_sorted = np.array([], dtype=int)

        top_idx = np.concatenate([pos_sorted, neg_sorted])

        print(f"\n  Top {len(pos_sorted)}R + {len(neg_sorted)}D partisan phrases:")
        for rank, fi in enumerate(top_idx, 1):
            direction = "R" if coef[fi] > 0 else "D"
            phrase = feature_names[fi] if feature_names[fi] is not None else f"feature_{fi}"
            print(f"    {rank:>2}. [{direction}] {phrase:<30s}  coef={coef[fi]:>+.6f}")
            all_top_phrases.append({
                "congress": cong_curr,
                "rank": rank,
                "phrase": phrase,
                "coefficient": float(coef[fi]),
                "direction": "Republican" if coef[fi] > 0 else "Democrat",
            })

    # ---- Save model ----
    window_file = "_".join(str(c) for c in window_congs)
    model_path = OUT_DIR / f"06_lasso_window_{window_file}.joblib"
    joblib.dump(model, model_path)

    elapsed = time.time() - window_start
    total_elapsed = time.time() - pipeline_start
    avg_per = total_elapsed / wi
    remaining = avg_per * (len(WINDOWS) - wi)
    print(f"\n  Time: {elapsed:.1f}s  |  Elapsed: {total_elapsed:.0f}s  |  ETA: {remaining:.0f}s")

    row = {
        "window": window_label,
        "scores_congress": cong_curr,
        "n_rep": n_rep,
        "n_dem": n_dem,
        "best_C": best_C,
        "train_accuracy": train_acc,
        "n_nonzero_coefs": n_nonzero,
        "n_positive_coefs": n_positive,
        "n_negative_coefs": n_negative,
    }
    if USE_BIC:
        row["bic"] = best_bic
    results.append(row)

    del model, coef, X, y
    gc.collect()

# ------------------------------------------------------------------
# 4. Sign consistency filter (zero out phrases that flip R/D)
# ------------------------------------------------------------------
SIGN_FILTER = cfg.CONFIG.get("sign_consistency_filter", False)

if SIGN_FILTER:
    print("\n" + "=" * 60)
    print("Applying sign consistency filter ...")
    print("=" * 60)

    # Collect signs across all windows
    sign_dict = {}  # feature_idx -> set of signs (+1 or -1)
    for window_congs in WINDOWS:
        window_file = "_".join(str(c) for c in window_congs)
        model_path = OUT_DIR / f"06_lasso_window_{window_file}.joblib"
        model = joblib.load(model_path)
        coef = model.coef_[0]
        for i in range(len(coef)):
            if coef[i] != 0:
                sign_dict.setdefault(i, set()).add(int(np.sign(coef[i])))
        del model

    # Identify flipping features
    flipping_idx = sorted(i for i, signs in sign_dict.items() if len(signs) > 1)
    flipping_phrases = [feature_names[i] for i in flipping_idx]

    print(f"  Phrases with consistent sign: {sum(1 for s in sign_dict.values() if len(s) == 1)}")
    print(f"  Phrases that flip direction:  {len(flipping_idx)}")
    if flipping_phrases:
        print(f"  Examples: {', '.join(flipping_phrases[:10])}")

    # Zero out flipping coefficients in all models and re-save
    total_zeroed = 0
    for window_congs in WINDOWS:
        window_file = "_".join(str(c) for c in window_congs)
        model_path = OUT_DIR / f"06_lasso_window_{window_file}.joblib"
        model = joblib.load(model_path)
        n_zeroed = 0
        for i in flipping_idx:
            if model.coef_[0][i] != 0:
                model.coef_[0][i] = 0.0
                n_zeroed += 1
        if n_zeroed > 0:
            joblib.dump(model, model_path)
        total_zeroed += n_zeroed
        cong_label = "_".join(str(c) for c in window_congs)
        print(f"  Window {cong_label}: zeroed {n_zeroed} coefficients")
        del model

    print(f"  Total coefficients zeroed: {total_zeroed}")

    # Save flipping phrases list for diagnostics
    if flipping_phrases:
        flip_df = pd.DataFrame({
            "feature_idx": flipping_idx,
            "phrase": flipping_phrases,
        })
        flip_df.to_csv(OUT_DIR / "06_flipping_phrases.csv", index=False)
        print(f"  Saved -> {OUT_DIR / '06_flipping_phrases.csv'}")

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
summary = pd.DataFrame(results)
summary.to_csv(OUT_DIR / "06_training_summary.csv", index=False)

if all_top_phrases:
    pd.DataFrame(all_top_phrases).to_csv(
        OUT_DIR / "06_top_partisan_phrases.csv", index=False
    )

print("\n" + "=" * 72)
print("SUMMARY: All rolling-window models")
print("=" * 72)
print(summary.to_string(index=False))
print(f"\nSaved models -> {OUT_DIR}")
print("=" * 72)
