"""
evaluate_experiments.py

Comprehensive evaluation of LASSO experiment configurations:
1. Training accuracy comparison across all 8 experiments
2. Cross-level validation (speech model → legislator data, vice versa)
3. Inter-model agreement on newspaper articles
4. LLM-based article evaluation (prepares sample for manual scoring)
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
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# Setup paths
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", "C:/Users/ymw04/Dropbox/shifting_slant"))
RUNS_DIR = BASE_DIR / "data" / "processed" / "runs"
NLP_DIR = Path(__file__).resolve().parent / "nlp"
sys.path.insert(0, str(NLP_DIR.parent / "utils"))
import text_analyzer  # noqa: F401 - needed for vectorizer unpickling

# All 8 experiment names
EXPERIMENTS = [
    "exp2_speech_all_bic",
    "exp2_speech_all_cv",
    "exp2_speech_core_bic",
    "exp2_speech_core_cv",
    "exp2_leg_all_bic",
    "exp2_leg_all_cv",
    "exp2_leg_core_bic",
    "exp2_leg_core_cv",
]

# Short labels for display
SHORT_LABELS = {
    "exp2_speech_all_bic":  "Sp/All/BIC",
    "exp2_speech_all_cv":   "Sp/All/CV",
    "exp2_speech_core_bic": "Sp/Core/BIC",
    "exp2_speech_core_cv":  "Sp/Core/CV",
    "exp2_leg_all_bic":     "Leg/All/BIC",
    "exp2_leg_all_cv":      "Leg/All/CV",
    "exp2_leg_core_bic":    "Leg/Core/BIC",
    "exp2_leg_core_cv":     "Leg/Core/CV",
}


def load_config(exp_name):
    config_path = BASE_DIR / "scripts" / "experiment_configs" / f"{exp_name}.json"
    with open(config_path) as f:
        return json.load(f)


def load_tfidf_data(tfidf_name):
    """Load step 05 output: feature matrix, metadata, vectorizer."""
    speech_dir = RUNS_DIR / tfidf_name / "speeches"
    X = sp.load_npz(speech_dir / "05_feature_matrix.npz")
    meta = pd.read_parquet(speech_dir / "05_speech_meta.parquet")
    vectorizer = joblib.load(speech_dir / "05_vectorizer.joblib")
    vocab_idx = np.load(speech_dir / "05_vocab_filter_idx.npy")
    return X, meta, vectorizer, vocab_idx


def load_model(exp_name, congress):
    """Load a trained LASSO model for a specific congress."""
    model_path = RUNS_DIR / exp_name / "models" / f"06_lasso_window_{congress}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


# =====================================================================
# PART 1: Training Accuracy Comparison
# =====================================================================
def compare_training_accuracy():
    print("=" * 80)
    print("PART 1: TRAINING ACCURACY COMPARISON")
    print("=" * 80)

    rows = []
    for exp in EXPERIMENTS:
        summary_path = RUNS_DIR / exp / "models" / "06_training_summary.csv"
        if not summary_path.exists():
            print(f"  MISSING: {exp}")
            continue
        df = pd.read_csv(summary_path)
        avg_acc = df["train_accuracy"].mean()
        min_acc = df["train_accuracy"].min()
        max_acc = df["train_accuracy"].max()
        avg_coefs = df["n_nonzero_coefs"].mean()
        rows.append({
            "experiment": SHORT_LABELS[exp],
            "avg_accuracy": avg_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "avg_nonzero_coefs": avg_coefs,
            "n_windows": len(df),
        })
        print(f"  {SHORT_LABELS[exp]:>14s}: avg={avg_acc:.3f}  range=[{min_acc:.3f}, {max_acc:.3f}]  "
              f"avg_coefs={avg_coefs:.0f}")

    return pd.DataFrame(rows)


# =====================================================================
# PART 2: Cross-Level Validation
# =====================================================================
def cross_level_validation():
    """
    Test how speech-level models perform on legislator-level data and vice versa.
    This is the core question: does aggregation level matter for classification?
    """
    print("\n" + "=" * 80)
    print("PART 2: CROSS-LEVEL VALIDATION")
    print("=" * 80)

    # Load both TF-IDF datasets
    print("\nLoading speech-level data...")
    X_sp, meta_sp, vec_sp, idx_sp = load_tfidf_data("exp2_tfidf_speech")
    print(f"  Shape: {X_sp.shape}")

    print("Loading legislator-level data...")
    X_leg, meta_leg, vec_leg, idx_leg = load_tfidf_data("exp2_tfidf_leg")
    print(f"  Shape: {X_leg.shape}")

    # Get party labels
    y_sp = (meta_sp["party"] == "R").astype(int).values
    y_leg = (meta_leg["party"] == "R").astype(int).values

    results = []

    # For each congress, test cross-level
    for cong in range(100, 109):
        # Speech mask
        sp_mask = meta_sp["congress_int"] == cong
        X_sp_cong = X_sp[sp_mask.values]
        y_sp_cong = y_sp[sp_mask.values]

        # Legislator mask
        leg_mask = meta_leg["congress_int"] == cong
        X_leg_cong = X_leg[leg_mask.values]
        y_leg_cong = y_leg[leg_mask.values]

        # Test 1: Speech model → speech data (baseline)
        for exp_sp in ["exp2_speech_all_cv", "exp2_speech_core_cv"]:
            model = load_model(exp_sp, cong)
            if model is None:
                continue
            pred_sp = model.predict(X_sp_cong)
            acc_sp = accuracy_score(y_sp_cong, pred_sp)
            results.append({
                "congress": cong, "model": SHORT_LABELS[exp_sp],
                "eval_data": "speech", "accuracy": acc_sp,
                "n_obs": len(y_sp_cong), "type": "same-level"
            })
            del model; gc.collect()

        # Test 2: Leg model → leg data (baseline)
        for exp_leg in ["exp2_leg_all_cv", "exp2_leg_core_cv"]:
            model = load_model(exp_leg, cong)
            if model is None:
                continue
            pred_leg = model.predict(X_leg_cong)
            acc_leg = accuracy_score(y_leg_cong, pred_leg)
            results.append({
                "congress": cong, "model": SHORT_LABELS[exp_leg],
                "eval_data": "legislator", "accuracy": acc_leg,
                "n_obs": len(y_leg_cong), "type": "same-level"
            })
            del model; gc.collect()

        # Test 3: Speech model → legislator data (cross-level)
        # Need to align vocabularies: speech model expects speech vectorizer's feature space
        # Both use same vectorizer base but different vocab_filter_idx
        # This is complicated by different vocabulary sizes.
        # Simpler approach: re-vectorize legislator text with speech vectorizer (if possible)
        # For now, skip if vocabularies don't match

        # Test 4: Leg model → speech data (cross-level)
        # Same vocabulary alignment issue

    del X_sp, X_leg, meta_sp, meta_leg
    gc.collect()

    if results:
        df = pd.DataFrame(results)
        print("\nSame-level accuracy by model and congress:")
        pivot = df.pivot_table(values="accuracy", index="model",
                               columns="congress", aggfunc="first")
        pivot["mean"] = pivot.mean(axis=1)
        print(pivot.to_string(float_format="%.3f"))
        return df
    return pd.DataFrame()


# =====================================================================
# PART 3: Cross-Level with Re-vectorization
# =====================================================================
def cross_level_revectorize():
    """
    For true cross-level testing, re-vectorize one level's text using
    the other level's vectorizer, then apply models.
    """
    print("\n" + "=" * 80)
    print("PART 3: CROSS-LEVEL RE-VECTORIZATION TEST")
    print("=" * 80)

    # Load raw speech text and labels
    LABELS_PATH = BASE_DIR / "data" / "processed" / "speeches" / "04_partisan_labels.parquet"
    SPEECHES_PATH = BASE_DIR / "data" / "processed" / "speeches" / "03_speeches_clean.parquet"

    labels = pd.read_parquet(LABELS_PATH)
    labels["speech_id"] = labels["speech_id"].astype(str)

    # Load speech vectorizer and leg vectorizer
    vec_sp = joblib.load(RUNS_DIR / "exp2_tfidf_speech" / "speeches" / "05_vectorizer.joblib")
    idx_sp = np.load(RUNS_DIR / "exp2_tfidf_speech" / "speeches" / "05_vocab_filter_idx.npy")

    vec_leg = joblib.load(RUNS_DIR / "exp2_tfidf_leg" / "speeches" / "05_vectorizer.joblib")
    idx_leg = np.load(RUNS_DIR / "exp2_tfidf_leg" / "speeches" / "05_vocab_filter_idx.npy")

    # For a single congress, load raw text and test
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    test_cong = 106  # 1999-2000

    print(f"\nLoading speeches for Congress {test_cong}...")
    pf = pq.ParquetFile(SPEECHES_PATH)
    chunks = []
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["speech_id", "speech", "congress"])
        mask = pc.equal(tbl.column("congress"), f"{test_cong:03d}")
        filtered = tbl.filter(mask)
        if len(filtered) > 0:
            chunks.append(filtered.select(["speech_id", "speech"]).to_pandas())
    speeches = pd.concat(chunks, ignore_index=True)
    speeches["speech_id"] = speeches["speech_id"].astype(str)

    merged = labels.merge(speeches, on="speech_id", how="inner")
    merged = merged[merged["congress_int"] == test_cong].reset_index(drop=True)
    del speeches, chunks

    # Filter to R/D only
    merged = merged[merged["party"].isin(["R", "D"])].reset_index(drop=True)
    y_true = (merged["party"] == "R").astype(int).values

    print(f"  {len(merged)} speeches (R={sum(y_true)}, D={len(y_true)-sum(y_true)})")

    # Vectorize with SPEECH vectorizer at speech level
    X_sp_raw = vec_sp.transform(merged["speech"])
    X_sp = X_sp_raw[:, idx_sp]
    # Normalize to L1 (relative freq)
    row_sums = np.array(X_sp.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    X_sp = sp.diags(1.0 / row_sums) @ X_sp

    # Vectorize with LEG vectorizer at speech level
    X_lg_raw = vec_leg.transform(merged["speech"])
    X_lg = X_lg_raw[:, idx_leg]
    row_sums2 = np.array(X_lg.sum(axis=1)).flatten()
    row_sums2[row_sums2 == 0] = 1
    X_lg = sp.diags(1.0 / row_sums2) @ X_lg

    # Now aggregate to legislator level for cross-testing
    merged["icpsr"] = merged["icpsr"].astype(str)
    leg_groups = merged.groupby("icpsr")
    leg_ids = sorted(leg_groups.groups.keys())

    results = []

    # Speech model on speech data (baseline)
    for exp_name in ["exp2_speech_all_cv", "exp2_speech_core_cv"]:
        model = load_model(exp_name, test_cong)
        if model is None:
            continue
        pred = model.predict(X_sp)
        acc = accuracy_score(y_true, pred)
        print(f"\n  {SHORT_LABELS[exp_name]} on speech data: {acc:.3f}")
        results.append({"model": SHORT_LABELS[exp_name], "eval": "speech->speech", "accuracy": acc})

        # Speech model on aggregated legislator level
        leg_preds = []
        leg_true = []
        for icpsr in leg_ids:
            mask = merged["icpsr"] == icpsr
            idx = np.where(mask.values)[0]
            if len(idx) == 0:
                continue
            # Aggregate: average prediction scores
            scores = model.decision_function(X_sp[idx])
            avg_score = scores.mean()
            leg_preds.append(1 if avg_score > 0 else 0)
            leg_true.append(y_true[idx[0]])  # All same party

        leg_acc = accuracy_score(leg_true, leg_preds)
        print(f"  {SHORT_LABELS[exp_name]} on legislator (aggregated): {leg_acc:.3f}  (n={len(leg_true)})")
        results.append({"model": SHORT_LABELS[exp_name], "eval": "speech->legislator_agg", "accuracy": leg_acc})
        del model; gc.collect()

    # Leg model on leg data (baseline) and on speech data
    for exp_name in ["exp2_leg_all_cv", "exp2_leg_core_cv"]:
        model = load_model(exp_name, test_cong)
        if model is None:
            continue

        # Leg model on speech-level data (using leg vectorizer)
        pred = model.predict(X_lg)
        acc = accuracy_score(y_true, pred)
        print(f"\n  {SHORT_LABELS[exp_name]} on speech data (leg vocab): {acc:.3f}")
        results.append({"model": SHORT_LABELS[exp_name], "eval": "leg->speech", "accuracy": acc})

        # Leg model on aggregated legislator level (using leg vectorizer)
        leg_preds = []
        leg_true2 = []
        for icpsr in leg_ids:
            mask = merged["icpsr"] == icpsr
            idx = np.where(mask.values)[0]
            if len(idx) == 0:
                continue
            scores = model.decision_function(X_lg[idx])
            avg_score = scores.mean()
            leg_preds.append(1 if avg_score > 0 else 0)
            leg_true2.append(y_true[idx[0]])

        leg_acc = accuracy_score(leg_true2, leg_preds)
        print(f"  {SHORT_LABELS[exp_name]} on legislator (aggregated): {leg_acc:.3f}  (n={len(leg_true2)})")
        results.append({"model": SHORT_LABELS[exp_name], "eval": "leg->legislator_agg", "accuracy": leg_acc})
        del model; gc.collect()

    del X_sp, X_lg, merged
    gc.collect()

    return pd.DataFrame(results)


# =====================================================================
# PART 4: Newspaper Domain Transfer (Inter-Model Agreement)
# =====================================================================
def newspaper_domain_transfer():
    """
    Apply all models to newspaper articles and compare:
    - Model predictions (sign agreement)
    - Correlation of net_slant scores
    """
    print("\n" + "=" * 80)
    print("PART 4: NEWSPAPER DOMAIN TRANSFER")
    print("=" * 80)

    # We need step 07 transformed newspaper features for each experiment
    # Since step 07 hasn't been run for exp2 configs, we'll use the main pipeline's
    # newspaper data and re-project with each model's coefficients

    # Load newspaper text from a sample congress
    test_cong = 106
    meta_path = BASE_DIR / "data" / "processed" / "newspapers" / f"07_newspaper_meta_cong_{test_cong}.parquet"
    tfidf_path = BASE_DIR / "data" / "processed" / "newspapers" / f"07_newspaper_tfidf_cong_{test_cong}.npz"

    if not meta_path.exists():
        print("  Main pipeline newspaper data not found. Skipping.")
        return pd.DataFrame()

    meta = pd.read_parquet(meta_path)
    # Main pipeline TF-IDF was built with main vectorizer, not exp2 vectorizers
    # So we can only compare exp2 models if they share the same feature space

    # Alternative: load raw newspaper articles and vectorize with each experiment's vectorizer
    print(f"\nLoading raw newspaper articles for Congress {test_cong}...")
    raw_path = BASE_DIR / "data" / "processed" / "newspapers" / f"04_newspaper_labeled_cong_{test_cong}.parquet"
    raw_meta = pd.read_parquet(raw_path)

    # Raw articles don't have text - need raw CSVs
    # Map congress to years
    cong_years = {test_cong: [1999, 2000]}
    all_articles = []
    for yr in cong_years[test_cong]:
        csv_path = BASE_DIR / "data" / "raw" / "newspapers" / "scrapes_since_1980" / f"nl_newsbank_com_{yr}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, usecols=["date", "paper", "text", "title", "word_count"])
            all_articles.append(df)
            print(f"  Loaded {len(df)} articles from {yr}")

    if not all_articles:
        print("  No raw articles found. Skipping.")
        return pd.DataFrame()

    articles = pd.concat(all_articles, ignore_index=True)
    del all_articles

    # Filter: keep only articles with enough text (100-2000 words)
    articles = articles[articles["word_count"].between(100, 2000)].reset_index(drop=True)

    # Sample 500 articles for tractability
    if len(articles) > 500:
        articles = articles.sample(n=500, random_state=42).reset_index(drop=True)

    print(f"  Sampled {len(articles)} articles from {len(articles['paper'].unique())} papers")

    # Vectorize with both speech and leg vectorizers
    vec_sp = joblib.load(RUNS_DIR / "exp2_tfidf_speech" / "speeches" / "05_vectorizer.joblib")
    idx_sp = np.load(RUNS_DIR / "exp2_tfidf_speech" / "speeches" / "05_vocab_filter_idx.npy")

    vec_leg = joblib.load(RUNS_DIR / "exp2_tfidf_leg" / "speeches" / "05_vectorizer.joblib")
    idx_leg = np.load(RUNS_DIR / "exp2_tfidf_leg" / "speeches" / "05_vocab_filter_idx.npy")

    X_sp_raw = vec_sp.transform(articles["text"])
    X_sp = X_sp_raw[:, idx_sp]
    row_sums = np.array(X_sp.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    X_sp = sp.diags(1.0 / row_sums) @ X_sp

    X_lg_raw = vec_leg.transform(articles["text"])
    X_lg = X_lg_raw[:, idx_lg]
    row_sums2 = np.array(X_lg.sum(axis=1)).flatten()
    row_sums2[row_sums2 == 0] = 1
    X_lg = sp.diags(1.0 / row_sums2) @ X_lg

    del X_sp_raw, X_lg_raw

    # Project with each model
    scores = {}
    for exp_name in EXPERIMENTS:
        model = load_model(exp_name, test_cong)
        if model is None:
            continue

        cfg = load_config(exp_name)
        is_leg = cfg.get("aggregate_to_legislator", False)
        X = X_lg if is_leg else X_sp

        coef = model.coef_[0]
        # Direct projection (same as step 08)
        raw_scores = X.dot(coef)

        # Decompose
        pos_mask = coef > 0
        neg_mask = coef < 0
        right_int = X[:, pos_mask].dot(coef[pos_mask])
        left_int = X[:, neg_mask].dot(np.abs(coef[neg_mask]))
        net_slant = right_int - left_int

        scores[SHORT_LABELS[exp_name]] = net_slant
        del model; gc.collect()

    if not scores:
        print("  No models available for comparison.")
        return pd.DataFrame()

    score_df = pd.DataFrame(scores)
    score_df["paper"] = articles["paper"].values
    score_df["title"] = articles["title"].values

    # Correlation matrix
    print("\nInter-model correlation (net_slant on 500 newspaper articles):")
    model_cols = [c for c in score_df.columns if c not in ["paper", "title"]]
    corr = score_df[model_cols].corr()
    print(corr.to_string(float_format="%.3f"))

    # Sign agreement
    print("\nSign agreement (% of articles with same R/D direction):")
    for i, m1 in enumerate(model_cols):
        for m2 in model_cols[i+1:]:
            sign_agree = ((score_df[m1] > 0) == (score_df[m2] > 0)).mean()
            print(f"  {m1} vs {m2}: {sign_agree:.1%}")

    # Save for further analysis
    score_df.to_parquet(BASE_DIR / "data" / "processed" / "panel" / "newspaper_model_comparison.parquet")
    print(f"\nSaved -> data/processed/panel/newspaper_model_comparison.parquet")

    return score_df


# =====================================================================
# PART 5: Prepare LLM Evaluation Sample
# =====================================================================
def prepare_llm_sample():
    """
    Sample articles for LLM evaluation: ~100 articles per year for 3 years.
    Save article text + metadata for LLM to read and score.
    """
    print("\n" + "=" * 80)
    print("PART 5: LLM EVALUATION SAMPLE")
    print("=" * 80)

    sample_years = [1993, 1997, 2001, 2003]
    n_per_year = 100
    all_samples = []

    for yr in sample_years:
        csv_path = BASE_DIR / "data" / "raw" / "newspapers" / "scrapes_since_1980" / f"nl_newsbank_com_{yr}.csv"
        if not csv_path.exists():
            print(f"  {yr}: file not found")
            continue

        df = pd.read_csv(csv_path)
        # Filter: news-like articles, 200-1500 words, exclude sports/obituaries
        df = df[df["word_count"].between(200, 1500)]
        # Basic content filter
        exclude_keywords = ["obituar", "box score", "stock price", "lottery", "weather",
                           "tv listing", "classified", "birth", "wedding", "engagement"]
        if "title" in df.columns:
            mask = ~df["title"].str.lower().str.contains("|".join(exclude_keywords), na=False)
            df = df[mask]
        if "text" in df.columns:
            # Exclude if first 50 chars contain sports scores pattern
            first_50 = df["text"].str[:50].str.lower()
            df = df[~first_50.str.contains(r"\d+\s*-\s*\d+", na=False)]

        if len(df) > n_per_year:
            df = df.sample(n=n_per_year, random_state=42 + yr)

        df["year"] = yr
        df["sample_id"] = range(len(df))
        all_samples.append(df)
        print(f"  {yr}: sampled {len(df)} articles from {df['paper'].nunique()} papers")

    if all_samples:
        sample = pd.concat(all_samples, ignore_index=True)
        out_path = BASE_DIR / "data" / "processed" / "panel" / "llm_eval_sample.parquet"
        sample.to_parquet(out_path)
        print(f"\nTotal: {len(sample)} articles saved to {out_path}")
        return sample

    return pd.DataFrame()


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0,
                       help="Run specific part (1-5) or 0 for all")
    args = parser.parse_args()

    if args.part == 0 or args.part == 1:
        compare_training_accuracy()

    if args.part == 0 or args.part == 2:
        cross_level_validation()

    if args.part == 0 or args.part == 3:
        cross_level_revectorize()

    if args.part == 0 or args.part == 4:
        newspaper_domain_transfer()

    if args.part == 0 or args.part == 5:
        prepare_llm_sample()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
