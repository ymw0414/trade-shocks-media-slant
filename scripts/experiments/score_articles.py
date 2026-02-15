"""
score_articles.py

Apply all available LASSO models to the LLM evaluation sample articles.
For each article x model, compute net_slant (right_intensity - left_intensity).
"""

import gc
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", "C:/Users/ymw04/Dropbox/shifting_slant"))
RUNS_DIR = BASE_DIR / "data" / "processed" / "runs"
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
import text_analyzer  # noqa: F401

# Load articles
articles = pd.read_parquet(BASE_DIR / "data" / "processed" / "panel" / "llm_eval_sample.parquet")
print(f"Loaded {len(articles)} articles")

# Map year to congress
def year_to_congress(yr):
    return (yr - 1787) // 2

articles["congress"] = articles["year"].apply(year_to_congress)
print(f"Year -> Congress mapping: {dict(zip(articles['year'].unique(), articles['congress'].unique()))}")

# Load vectorizers (already filtered to final vocabulary, no need for idx)
print("\nLoading vectorizers...")
vec_sp = joblib.load(RUNS_DIR / "exp2_tfidf_speech" / "speeches" / "05_vectorizer.joblib")
vec_leg = joblib.load(RUNS_DIR / "exp2_tfidf_leg" / "speeches" / "05_vectorizer.joblib")

# Vectorize articles with both vectorizers
# The saved vectorizers already have filtered vocabulary, so transform()
# directly produces the correct feature space (no idx filtering needed)
print("Vectorizing with speech vectorizer...")
X_sp = vec_sp.transform(articles["text"].fillna(""))
row_sums = np.array(X_sp.sum(axis=1)).flatten()
row_sums[row_sums == 0] = 1
X_sp = sp.diags(1.0 / row_sums) @ X_sp
print(f"  Shape: {X_sp.shape}")

print("Vectorizing with legislator vectorizer...")
X_lg = vec_leg.transform(articles["text"].fillna(""))
row_sums2 = np.array(X_lg.sum(axis=1)).flatten()
row_sums2[row_sums2 == 0] = 1
X_lg = sp.diags(1.0 / row_sums2) @ X_lg
print(f"  Shape: {X_lg.shape}")

EXPERIMENTS = {
    "exp2_speech_all_bic":  ("speech", "Sp/All/BIC"),
    "exp2_speech_all_cv":   ("speech", "Sp/All/CV"),
    "exp2_speech_core_bic": ("speech", "Sp/Core/BIC"),
    "exp2_speech_core_cv":  ("speech", "Sp/Core/CV"),
    "exp2_leg_all_bic":     ("leg",    "Leg/All/BIC"),
    "exp2_leg_all_cv":      ("leg",    "Leg/All/CV"),
    "exp2_leg_core_bic":    ("leg",    "Leg/Core/BIC"),
    "exp2_leg_core_cv":     ("leg",    "Leg/Core/CV"),
}

# Score articles with each available model
all_scores = {}

for exp_name, (vec_type, short_label) in EXPERIMENTS.items():
    model_dir = RUNS_DIR / exp_name / "models"
    if not model_dir.exists():
        print(f"\n  SKIP {short_label}: no models directory")
        continue

    X = X_lg if vec_type == "leg" else X_sp
    scores_arr = np.full(len(articles), np.nan)

    # Group articles by congress and load each model once
    for cong, group in articles.groupby("congress"):
        model_path = model_dir / f"06_lasso_window_{cong}.joblib"
        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        coef = model.coef_[0]
        pos_mask = coef > 0
        neg_mask = coef < 0

        idx = group.index.values
        X_cong = X[idx]

        right_int = np.array(X_cong[:, pos_mask].dot(coef[pos_mask])).flatten()
        left_int = np.array(X_cong[:, neg_mask].dot(np.abs(coef[neg_mask]))).flatten()
        net_slant = right_int - left_int

        scores_arr[idx] = net_slant
        del model; gc.collect()

    all_scores[short_label] = scores_arr
    n_valid = np.sum(~np.isnan(scores_arr))
    n_right = np.sum(scores_arr[~np.isnan(scores_arr)] > 0)
    print(f"  {short_label}: scored {n_valid}/{len(articles)} articles, "
          f"{n_right} right-leaning ({n_right/n_valid*100:.0f}%)")

# Combine results
score_df = articles[["date", "paper", "title", "word_count", "year", "congress"]].copy()
for label, scores in all_scores.items():
    score_df[label] = scores

# Summary statistics
print("\n" + "=" * 70)
print("INTER-MODEL CORRELATION (net_slant)")
print("=" * 70)
model_cols = [c for c in score_df.columns if c.startswith("Sp/") or c.startswith("Leg/")]
if len(model_cols) > 1:
    corr = score_df[model_cols].corr()
    print(corr.to_string(float_format="%.3f"))

    print("\nSIGN AGREEMENT (% articles classified same direction)")
    for i, m1 in enumerate(model_cols):
        for m2 in model_cols[i+1:]:
            valid = score_df[[m1, m2]].dropna()
            if len(valid) > 0:
                agree = ((valid[m1] > 0) == (valid[m2] > 0)).mean()
                print(f"  {m1:>14s} vs {m2:<14s}: {agree:.1%}")

# Save
out_path = BASE_DIR / "data" / "processed" / "panel" / "article_model_scores.parquet"
score_df.to_parquet(out_path)
print(f"\nSaved -> {out_path}")
