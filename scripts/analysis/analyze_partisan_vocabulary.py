"""
analyze_partisan_vocabulary.py

Analyze LASSO-selected partisan vocabulary:
  1. Top R-coded and D-coded features (coefficient magnitude)
  2. Newspaper document frequency (how often each feature appears)
  3. Frequency split by high vs low NAFTA vulnerability
  4. Frequency split by pre vs post NAFTA
  5. Contribution = coef x frequency (which features DRIVE the slant score)

Goal: understand WHY R̃ increases but L̃ doesn't after NAFTA.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import pipeline_config as cfg
import text_analyzer  # noqa: F401 — needed for vectorizer unpickling

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
MODEL_DIR = cfg.MODEL_DIR
NEWS_DIR = cfg.INPUT_NEWS_DIR
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"

NAFTA_YEAR = 1994


def load_feature_names():
    """Load vectorizer and shared vocab mask to get filtered feature names."""
    vec_path = cfg.INPUT_SPEECH_DIR / "05_feature_vectorizer.joblib"
    vec = joblib.load(vec_path)
    all_features = vec.get_feature_names_out()

    mask_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    mask = np.load(mask_path)

    filtered = all_features[mask]
    print(f"  Vocabulary: {len(all_features)} total -> {len(filtered)} after shared vocab filter")
    return filtered, mask


def load_avg_coefficients(feature_names):
    """Load all LASSO models and average coefficients across congresses."""
    windows = cfg.get_windows()
    all_coefs = []

    for window_congs in windows:
        window_file = "_".join(str(c) for c in window_congs)
        model_path = MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
        model = joblib.load(model_path)
        all_coefs.append(model.coef_[0])

    coef_matrix = np.array(all_coefs)  # (n_windows, n_features)
    avg_coef = coef_matrix.mean(axis=0)
    # Also track how many windows have nonzero coef
    n_nonzero = (coef_matrix != 0).sum(axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "avg_coef": avg_coef,
        "n_windows_nonzero": n_nonzero,
        "direction": np.where(avg_coef > 0, "R", np.where(avg_coef < 0, "D", "zero")),
    })
    print(f"  Coefficients: {(df['direction']=='R').sum()} R-coded, "
          f"{(df['direction']=='D').sum()} D-coded, "
          f"{(df['direction']=='zero').sum()} zero")
    return df, coef_matrix


def compute_newspaper_frequencies(feature_names, shared_mask):
    """
    Compute feature frequencies in newspapers, split by vulnerability and period.
    Uses vectorized sparse matrix operations for speed.

    Loads the L1-relative-frequency feature matrices (not TF-IDF).
    """
    # Load panel for vulnerability mapping
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel[panel["cz"].notna() & panel["vulnerability1990_scaled"].notna()].copy()
    panel = panel[panel["year"] <= 2004].copy()

    # Median vulnerability
    paper_vuln = panel.groupby("paper")["vulnerability1990_scaled"].first()
    vuln_median = paper_vuln.median()
    hi_vuln_papers = set(paper_vuln[paper_vuln > vuln_median].index)
    lo_vuln_papers = set(paper_vuln[paper_vuln <= vuln_median].index)
    print(f"  Vulnerability median: {vuln_median:.4f}")
    print(f"  High-vuln papers: {len(hi_vuln_papers)}, Low-vuln: {len(lo_vuln_papers)}")

    windows = cfg.get_windows()
    n_features = len(feature_names)

    # Accumulators: {group: {"doc_count": array, "freq_sum": array, "n": int}}
    groups = ["all", "pre", "post", "hi", "lo", "hi_pre", "hi_post", "lo_pre", "lo_post"]
    acc = {g: {"doc_count": np.zeros(n_features), "freq_sum": np.zeros(n_features), "n": 0}
           for g in groups}

    for window_congs in windows:
        cong = window_congs[-1]

        # Load newspaper feature matrix
        X = sp.load_npz(NEWS_DIR / f"07_newspaper_features_cong_{cong}.npz")
        X = X[:, shared_mask].tocsr()
        meta = pd.read_parquet(NEWS_DIR / f"07_newspaper_meta_cong_{cong}.parquet")

        # Build boolean masks for each group (vectorized)
        papers = meta["paper"].values
        years = meta["year"].values
        is_pre = years < NAFTA_YEAR
        is_post = ~is_pre
        is_hi = np.array([p in hi_vuln_papers for p in papers])
        is_lo = np.array([p in lo_vuln_papers for p in papers])

        masks = {
            "all": np.ones(len(meta), dtype=bool),
            "pre": is_pre, "post": is_post,
            "hi": is_hi, "lo": is_lo,
            "hi_pre": is_hi & is_pre, "hi_post": is_hi & is_post,
            "lo_pre": is_lo & is_pre, "lo_post": is_lo & is_post,
        }

        X_binary = (X > 0).astype(np.float32)

        for g, mask in masks.items():
            n_sel = mask.sum()
            if n_sel == 0:
                continue
            Xsub = X[mask]
            Xbin = X_binary[mask]
            # Sum across articles -> (n_features,) dense arrays
            acc[g]["doc_count"] += np.asarray(Xbin.sum(axis=0)).ravel()
            acc[g]["freq_sum"] += np.asarray(Xsub.sum(axis=0)).ravel()
            acc[g]["n"] += n_sel

        print(f"    Congress {cong}: {len(meta):,} articles processed")

    # Build result DataFrame
    result = pd.DataFrame({"feature": feature_names})
    for g in groups:
        n = acc[g]["n"]
        if n > 0:
            result[f"doc_freq_{g}"] = acc[g]["doc_count"] / n
            result[f"mean_freq_{g}"] = acc[g]["freq_sum"] / n
        print(f"  Group '{g}': {n:,} articles")

    return result


def analyze_results(coef_df, freq_df):
    """Merge and analyze: which features drive the R̃ increase?"""
    df = coef_df.merge(freq_df, on="feature")

    # Contribution = |coef| x mean_freq (how much each feature contributes to slant score)
    df["contrib_all"] = df["avg_coef"].abs() * df["mean_freq_all"]
    df["contrib_hi_post"] = df["avg_coef"].abs() * df["mean_freq_hi_post"]
    df["contrib_lo_post"] = df["avg_coef"].abs() * df["mean_freq_lo_post"]
    df["contrib_hi_pre"] = df["avg_coef"].abs() * df["mean_freq_hi_pre"]
    df["contrib_lo_pre"] = df["avg_coef"].abs() * df["mean_freq_lo_pre"]

    # Change: hi_post - hi_pre (did frequency change in vulnerable areas after NAFTA?)
    df["freq_change_hi"] = df["doc_freq_hi_post"] - df["doc_freq_hi_pre"]
    df["freq_change_lo"] = df["doc_freq_lo_post"] - df["doc_freq_lo_pre"]
    df["freq_did"] = df["freq_change_hi"] - df["freq_change_lo"]  # diff-in-diff of frequency

    df["freq_change_hi_mean"] = df["mean_freq_hi_post"] - df["mean_freq_hi_pre"]
    df["freq_change_lo_mean"] = df["mean_freq_lo_post"] - df["mean_freq_lo_pre"]
    df["freq_did_mean"] = df["freq_change_hi_mean"] - df["freq_change_lo_mean"]

    # Signed contribution DiD: positive = pushes slant rightward in hi-vuln areas post-NAFTA
    df["signed_contrib_did"] = df["avg_coef"] * df["freq_did_mean"]

    return df


def print_report(df):
    """Print comprehensive report."""
    R = df[df["direction"] == "R"].copy()
    D = df[df["direction"] == "D"].copy()

    # ======== SECTION 1: Top R-coded features by contribution ========
    print("\n" + "=" * 100)
    print("  TOP 30 R-CODED FEATURES BY OVERALL CONTRIBUTION (|coef| x freq)")
    print("=" * 100)
    top_R = R.nlargest(30, "contrib_all")
    print(f"  {'Feature':<25} {'Coef':>8} {'DocFreq%':>9} {'MeanFreq':>8} "
          f"{'Contrib':>8} {'FreqDiD':>8} {'ContribDiD':>10}")
    print("  " + "-" * 90)
    for _, r in top_R.iterrows():
        print(f"  {r['feature']:<25} {r['avg_coef']:>8.2f} {r['doc_freq_all']*100:>8.2f}% "
              f"{r['mean_freq_all']:>8.5f} {r['contrib_all']:>8.4f} "
              f"{r['freq_did']*100:>7.3f}% {r['signed_contrib_did']:>10.5f}")

    # ======== SECTION 2: Top D-coded features by contribution ========
    print("\n" + "=" * 100)
    print("  TOP 30 D-CODED FEATURES BY OVERALL CONTRIBUTION (|coef| x freq)")
    print("=" * 100)
    top_D = D.nlargest(30, "contrib_all")
    print(f"  {'Feature':<25} {'Coef':>8} {'DocFreq%':>9} {'MeanFreq':>8} "
          f"{'Contrib':>8} {'FreqDiD':>8} {'ContribDiD':>10}")
    print("  " + "-" * 90)
    for _, r in top_D.iterrows():
        print(f"  {r['feature']:<25} {r['avg_coef']:>8.2f} {r['doc_freq_all']*100:>8.2f}% "
              f"{r['mean_freq_all']:>8.5f} {r['contrib_all']:>8.4f} "
              f"{r['freq_did']*100:>7.3f}% {r['signed_contrib_did']:>10.5f}")

    # ======== SECTION 3: Features with largest positive DiD contribution ========
    # (features that push slant rightward MORE in hi-vuln areas post-NAFTA)
    print("\n" + "=" * 100)
    print("  TOP 30 FEATURES BY SIGNED CONTRIBUTION DiD")
    print("  (positive = pushes slant rightward in hi-vuln areas after NAFTA)")
    print("=" * 100)
    top_did = df[df["avg_coef"] != 0].nlargest(30, "signed_contrib_did")
    print(f"  {'Feature':<25} {'Dir':>3} {'Coef':>8} {'DocFreq%':>9} "
          f"{'Hi Pre->Post':>13} {'Lo Pre->Post':>13} {'FreqDiD':>8} {'ContribDiD':>10}")
    print("  " + "-" * 100)
    for _, r in top_did.iterrows():
        hi_change = f"{r['doc_freq_hi_pre']*100:.2f}->{r['doc_freq_hi_post']*100:.2f}"
        lo_change = f"{r['doc_freq_lo_pre']*100:.2f}->{r['doc_freq_lo_post']*100:.2f}"
        print(f"  {r['feature']:<25} {r['direction']:>3} {r['avg_coef']:>8.2f} "
              f"{r['doc_freq_all']*100:>8.2f}% {hi_change:>13} {lo_change:>13} "
              f"{r['freq_did']*100:>7.3f}% {r['signed_contrib_did']:>10.5f}")

    # ======== SECTION 4: Aggregate statistics ========
    print("\n" + "=" * 100)
    print("  AGGREGATE CONTRIBUTION ANALYSIS")
    print("=" * 100)

    for direction, label in [("R", "R-coded"), ("D", "D-coded")]:
        sub = df[df["direction"] == direction]
        total_contrib_hi_pre = (sub["avg_coef"].abs() * sub["mean_freq_hi_pre"]).sum()
        total_contrib_hi_post = (sub["avg_coef"].abs() * sub["mean_freq_hi_post"]).sum()
        total_contrib_lo_pre = (sub["avg_coef"].abs() * sub["mean_freq_lo_pre"]).sum()
        total_contrib_lo_post = (sub["avg_coef"].abs() * sub["mean_freq_lo_post"]).sum()

        print(f"\n  {label} features ({len(sub)}):")
        print(f"    Total contribution (|coef| x freq):")
        print(f"      Hi-vuln pre:  {total_contrib_hi_pre:.4f}")
        print(f"      Hi-vuln post: {total_contrib_hi_post:.4f}  (Delta = {total_contrib_hi_post - total_contrib_hi_pre:+.4f})")
        print(f"      Lo-vuln pre:  {total_contrib_lo_pre:.4f}")
        print(f"      Lo-vuln post: {total_contrib_lo_post:.4f}  (Delta = {total_contrib_lo_post - total_contrib_lo_pre:+.4f})")
        print(f"      DiD:          {(total_contrib_hi_post - total_contrib_hi_pre) - (total_contrib_lo_post - total_contrib_lo_pre):+.4f}")

    # ======== SECTION 5: Trade-related keyword search ========
    print("\n" + "=" * 100)
    print("  TRADE-RELATED FEATURES")
    print("=" * 100)
    trade_keywords = ["trade", "tariff", "import", "export", "nafta", "manufactur",
                      "factor", "worker", "job", "employ", "unemploy", "econom",
                      "foreign", "compet", "industri", "plant", "wage",
                      "mexico", "mexican", "canada", "canadian", "china", "chinese",
                      "free trade", "trade agreement", "trade deficit"]
    trade_mask = df["feature"].apply(
        lambda x: any(kw in x for kw in trade_keywords))
    trade_df = df[trade_mask & (df["avg_coef"] != 0)].sort_values("avg_coef", ascending=False)

    if len(trade_df) > 0:
        print(f"\n  Found {len(trade_df)} trade-related features:")
        print(f"    R-coded: {(trade_df['direction']=='R').sum()}")
        print(f"    D-coded: {(trade_df['direction']=='D').sum()}")
        print(f"\n  {'Feature':<30} {'Dir':>3} {'Coef':>8} {'DocFreq%':>9} "
              f"{'FreqDiD%':>9} {'ContribDiD':>10}")
        print("  " + "-" * 80)
        for _, r in trade_df.iterrows():
            print(f"  {r['feature']:<30} {r['direction']:>3} {r['avg_coef']:>8.2f} "
                  f"{r['doc_freq_all']*100:>8.2f}% {r['freq_did']*100:>8.3f}% "
                  f"{r['signed_contrib_did']:>10.5f}")
    else:
        print("  No trade-related features found in vocabulary.")

    return df


def main():
    print("=" * 80)
    print("  PARTISAN VOCABULARY ANALYSIS")
    print("=" * 80)

    print("\n[1] Loading feature names ...")
    feature_names, shared_mask = load_feature_names()

    print("\n[2] Loading LASSO coefficients (averaged across 9 congresses) ...")
    coef_df, coef_matrix = load_avg_coefficients(feature_names)

    print("\n[3] Computing newspaper feature frequencies ...")
    print("    (this processes all articles across 9 congresses, may take a few minutes)")
    freq_df = compute_newspaper_frequencies(feature_names, shared_mask)

    print("\n[4] Analyzing results ...")
    df = analyze_results(coef_df, freq_df)

    print_report(df)

    # Save full results
    out_path = cfg.TAB_DIR / "partisan_vocabulary_analysis.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  Full results saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
