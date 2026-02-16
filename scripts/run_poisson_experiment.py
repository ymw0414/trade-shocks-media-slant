"""
run_poisson_experiment.py

Experiment: replace the binary logistic LASSO (step 06) with a
multinomial/Poisson log-frequency-ratio model, keeping the entire
rest of the pipeline identical.

Approach:
  For each congress window, compute party-specific average relative
  word frequencies, take the log-ratio (Poisson MLE), and apply L1
  soft-thresholding with 5-fold CV to select sparsity.

  This produces coefficient vectors in the same format as the logistic
  LASSO.  Steps 08-16 run unchanged.

Usage:
  python scripts/run_poisson_experiment.py
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Ensure text_analyzer importable (needed for vectorizer unpickling)
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
import text_analyzer  # noqa: F401

# -- Paths ------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PYTHON = sys.executable

EXP_NAME = "exp_poisson"
SOURCE_RUN = "exp_shvocab_cv"

SOURCE_DIR = BASE_DIR / "data" / "processed" / "runs" / SOURCE_RUN
EXP_DIR = BASE_DIR / "data" / "processed" / "runs" / EXP_NAME

# Speech DTM lives in the shared input dir (exp_unigram_gst), not the run dir
SPEECH_DIR = BASE_DIR / "data" / "processed" / "runs" / "exp_unigram_gst" / "speeches"
NEWS_INPUT_DIR = SOURCE_DIR / "newspapers"
MODEL_DIR = EXP_DIR / "models"
NEWS_DIR = EXP_DIR / "newspapers"
PANEL_DIR = EXP_DIR / "panel"
FIG_DIR = EXP_DIR / "output" / "figures"
TAB_DIR = EXP_DIR / "output" / "tables"

# Scripts directory
SCRIPTS_DIR = BASE_DIR / "scripts"
NLP_DIR = SCRIPTS_DIR / "nlp"
ECON_DIR = SCRIPTS_DIR / "econ"
ANALYSIS_DIR = SCRIPTS_DIR / "analysis"


# =====================================================================
# Create sklearn-compatible model from Poisson coefficients
# =====================================================================
from sklearn.linear_model import LogisticRegression


def make_sklearn_model(coef, intercept, n_features):
    """Create a sklearn LogisticRegression with injected coefficients.

    This avoids pickling issues: downstream scripts (08, 09) load a
    standard sklearn object that they already know how to handle.
    """
    model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0)
    # Manually set fitted attributes
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([coef])            # shape (1, n_features)
    model.intercept_ = np.array([intercept])
    model.n_features_in_ = n_features
    return model


# =====================================================================
# Training: Poisson log-frequency-ratio with soft-threshold + CV
# =====================================================================
def compute_log_ratios(X, y, eps=1e-8):
    """Compute log(f_R / f_D) from L1-normalized feature matrix."""
    rep_mask = y == 1
    dem_mask = y == 0
    f_R = np.asarray(X[rep_mask].mean(axis=0)).ravel() + eps
    f_D = np.asarray(X[dem_mask].mean(axis=0)).ravel() + eps
    return np.log(f_R) - np.log(f_D)


def soft_threshold(beta, lam):
    """L1 soft-threshold: sign(β) * max(|β| - λ, 0)."""
    return np.sign(beta) * np.maximum(np.abs(beta) - lam, 0.0)


def score_accuracy(X, y, beta):
    """Classification accuracy using sign of X @ beta."""
    scores = X @ beta
    if hasattr(scores, "A1"):
        scores = scores.A1
    preds = (scores > 0).astype(int)
    return float(np.mean(preds == y))


def calibrate_intercept(X, y, beta, lr=0.1, n_iter=100):
    """Fit intercept via gradient descent on log-loss (Platt scaling)."""
    scores = X @ beta
    if hasattr(scores, "A1"):
        scores = scores.A1
    intercept = 0.0
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(scores + intercept, -30, 30)))
        grad = np.mean(p - y)
        intercept -= lr * grad
    return intercept


def train_poisson_window(X, y, cv_folds=5, n_lambdas=60, random_state=42):
    """Train Poisson model for one congress window.

    Returns: (PoissonSlantModel, best_lambda, cv_accuracy, n_nonzero)
    """
    # Full-data log-ratios
    beta_full = compute_log_ratios(X, y)

    # Lambda grid: 0 to ~99th percentile of |beta|
    lam_max = np.percentile(np.abs(beta_full), 99.5)
    lambdas = np.linspace(0, lam_max, n_lambdas)

    # 5-fold CV
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)
    folds = list(kf.split(X, y))

    # Pre-compute fold-specific log-ratios (expensive part)
    fold_betas = []
    for train_idx, _ in folds:
        fold_betas.append(compute_log_ratios(X[train_idx], y[train_idx]))

    # Evaluate each lambda
    best_lam = 0.0
    best_acc = 0.0
    cv_log = []

    for lam in lambdas:
        fold_accs = []
        for fi, (train_idx, val_idx) in enumerate(folds):
            beta_thresh = soft_threshold(fold_betas[fi], lam)
            acc = score_accuracy(X[val_idx], y[val_idx], beta_thresh)
            fold_accs.append(acc)
        mean_acc = np.mean(fold_accs)
        n_nz = int(np.sum(soft_threshold(beta_full, lam) != 0))
        cv_log.append({"lambda": lam, "cv_accuracy": mean_acc, "n_nonzero": n_nz})

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lam = lam

    # Final model on full data
    beta_final = soft_threshold(beta_full, best_lam)
    intercept = calibrate_intercept(X, y, beta_final)

    model = make_sklearn_model(beta_final, intercept, X.shape[1])

    n_nonzero = int(np.sum(beta_final != 0))
    n_pos = int(np.sum(beta_final > 0))
    n_neg = int(np.sum(beta_final < 0))

    return model, best_lam, best_acc, n_nonzero, n_pos, n_neg, cv_log


# =====================================================================
# Step 06 replacement: train all windows
# =====================================================================
def train_all_windows():
    """Train Poisson models for all congress windows."""
    print("=" * 72)
    print("STEP 06 (Poisson): Training MNL/Poisson log-ratio models")
    print("=" * 72)

    # Load feature matrix (L1-normalized relative frequencies)
    matrix_path = SPEECH_DIR / "05_feature_matrix.npz"
    meta_path = SPEECH_DIR / "05_feature_meta.parquet"
    vec_path = SPEECH_DIR / "05_feature_vectorizer.joblib"

    print(f"\nLoading feature matrix: {matrix_path}")
    X_all = sp.load_npz(matrix_path)
    meta = pd.read_parquet(meta_path)

    PARTY_COL = "party" if "party" in meta.columns else "party_code"
    print(f"  Matrix: {X_all.shape[0]:,} x {X_all.shape[1]:,}")

    # Load feature names
    vec = joblib.load(vec_path)
    n_features = X_all.shape[1]
    feature_names = np.empty(n_features, dtype=object)
    for feat, idx in vec.vocabulary_.items():
        if idx < n_features:
            feature_names[idx] = feat
    del vec

    # Load or copy shared vocabulary mask from source run
    source_mask_path = SOURCE_DIR / "models" / "06_shared_vocab_mask.npy"
    shared_vocab_mask = None
    if source_mask_path.exists():
        shared_vocab_mask = np.load(source_mask_path)
        X_all = X_all[:, shared_vocab_mask]
        feature_names = feature_names[shared_vocab_mask]
        n_features = X_all.shape[1]
        print(f"  Applied shared vocab mask: {n_features:,} features")

        # Save mask in experiment models dir
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        np.save(MODEL_DIR / "06_shared_vocab_mask.npy", shared_vocab_mask)

    # Congress windows (1-congress, 100-108)
    congresses = list(range(100, 109))
    windows = [(c,) for c in congresses]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    all_top_phrases = []

    pipeline_start = time.time()

    for wi, window_congs in enumerate(windows, 1):
        cong_curr = window_congs[-1]
        window_label = "+".join(str(c) for c in window_congs)
        window_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"[{wi}/{len(windows)}] Window: Congress {window_label}")
        print(f"{'=' * 60}")

        # Select training rows
        mask_window = meta["congress_int"].isin(window_congs)
        idx = meta.index[mask_window].values
        party_vals = meta.loc[idx, PARTY_COL].values
        y = ((party_vals == "R") | (party_vals == 200)).astype(int)

        X = X_all[idx]
        n_rep = int(y.sum())
        n_dem = len(y) - n_rep
        print(f"  Training: {len(y):,} speeches ({n_rep:,} Rep, {n_dem:,} Dem)")

        # Train Poisson model
        model, best_lam, cv_acc, n_nonzero, n_pos, n_neg, cv_log = \
            train_poisson_window(X, y, cv_folds=5)

        train_acc = model.score(X, y)
        coef = model.coef_[0]

        print(f"\n  Best lambda:    {best_lam:.6f}")
        print(f"  CV accuracy:    {cv_acc:.3f}")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Non-zero coefs: {n_nonzero:,} (+{n_pos:,} / -{n_neg:,})")

        # Top partisan phrases
        top_per_dir = 15
        if n_nonzero > 0:
            pos_idx = np.where(coef > 0)[0]
            neg_idx = np.where(coef < 0)[0]

            if len(pos_idx) > 0:
                pos_sorted = pos_idx[np.argsort(coef[pos_idx])[::-1][:top_per_dir]]
            else:
                pos_sorted = np.array([], dtype=int)
            if len(neg_idx) > 0:
                neg_sorted = neg_idx[np.argsort(coef[neg_idx])[:top_per_dir]]
            else:
                neg_sorted = np.array([], dtype=int)

            top_idx = np.concatenate([pos_sorted, neg_sorted])

            print(f"\n  Top {len(pos_sorted)}R + {len(neg_sorted)}D phrases:")
            for rank, fi in enumerate(top_idx, 1):
                direction = "R" if coef[fi] > 0 else "D"
                phrase = feature_names[fi] if feature_names[fi] is not None else f"feat_{fi}"
                print(f"    {rank:>2}. [{direction}] {phrase:<30s}  coef={coef[fi]:>+.6f}")
                all_top_phrases.append({
                    "congress": cong_curr,
                    "rank": rank,
                    "phrase": phrase,
                    "coefficient": float(coef[fi]),
                    "direction": "Republican" if coef[fi] > 0 else "Democrat",
                })

        # Save model
        window_file = "_".join(str(c) for c in window_congs)
        model_path = MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
        joblib.dump(model, model_path)

        elapsed = time.time() - window_start
        total_elapsed = time.time() - pipeline_start
        avg_per = total_elapsed / wi
        remaining = avg_per * (len(windows) - wi)
        print(f"\n  Time: {elapsed:.1f}s  |  Elapsed: {total_elapsed:.0f}s  |  ETA: {remaining:.0f}s")

        results.append({
            "window": window_label,
            "scores_congress": cong_curr,
            "n_rep": n_rep,
            "n_dem": n_dem,
            "best_lambda": best_lam,
            "train_accuracy": train_acc,
            "cv_accuracy": cv_acc,
            "n_nonzero_coefs": n_nonzero,
            "n_positive_coefs": n_pos,
            "n_negative_coefs": n_neg,
        })

        del model, coef, X, y
        gc.collect()

    # Save summaries
    summary = pd.DataFrame(results)
    summary.to_csv(MODEL_DIR / "06_training_summary.csv", index=False)

    if all_top_phrases:
        pd.DataFrame(all_top_phrases).to_csv(
            MODEL_DIR / "06_top_partisan_phrases.csv", index=False
        )

    print("\n" + "=" * 72)
    print("SUMMARY: Poisson log-ratio models")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nSaved -> {MODEL_DIR}")

    return summary


# =====================================================================
# Run downstream pipeline steps via subprocess
# =====================================================================
def run_step(script_path, env_extra=None, label=""):
    """Run a pipeline step as a subprocess."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    print(f"\n{'-' * 60}")
    print(f"Running: {label or script_path.name}")
    print(f"{'-' * 60}")

    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(NLP_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Step failed: {script_path.name}")

    # Print last few lines of output
    lines = result.stdout.strip().split("\n")
    for line in lines[-15:]:
        print(f"  {line}")

    return result


def run_pipeline():
    """Run steps 08 → 09 → 10 → 11 → 14 → 15 → 16."""
    print("\n" + "=" * 72)
    print("Running downstream pipeline (steps 08-16)")
    print("=" * 72)

    # Config override for steps that import pipeline_config
    config_override = {
        "run_name": EXP_NAME,
        "input_speech_dir": "data/processed/runs/exp_unigram_gst/speeches",
        "input_news_dir": "data/processed/runs/exp_unigram_gst/newspapers",
        "shared_vocab_min_df": 0.0001,
        "norm_method": "raw_gap",
    }
    config_path = EXP_DIR / "config_override.json"
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_override, f, indent=2)

    env_base = {
        "PIPELINE_CONFIG_OVERRIDE": str(config_path),
    }

    # Env vars for steps that use explicit overrides (11-16)
    env_downstream = {
        **env_base,
        "PIPELINE_NEWS_DIR": str(NEWS_DIR),
        "PIPELINE_PANEL_DIR": str(PANEL_DIR),
        "PIPELINE_FIG_DIR": str(FIG_DIR),
        "PIPELINE_TAB_DIR": str(TAB_DIR),
    }

    # Step 08: Project slant
    run_step(NLP_DIR / "08_project_slant.py", env_base,
             "Step 08: Project Poisson coefficients onto newspapers")

    # Step 09: Normalize
    run_step(NLP_DIR / "09_normalize_slant.py", env_base,
             "Step 09: Normalize by partisan gap")

    # Step 10: Aggregate to newspaper-year panel
    run_step(NLP_DIR / "10_aggregate_slant.py", env_base,
             "Step 10: Aggregate to newspaper-year panel")

    # Step 11: Merge geography
    run_step(ECON_DIR / "11_merge_geography.py", env_downstream,
             "Step 11: Merge geography")

    # Step 14: Merge panel
    run_step(ECON_DIR / "14_merge_panel.py", env_downstream,
             "Step 14: Merge regression panel")

    # Step 15: Event study
    run_step(ANALYSIS_DIR / "15_event_study.py", env_downstream,
             "Step 15: Event study")

    # Step 16: DiD
    run_step(ANALYSIS_DIR / "16_did_regression.py", env_downstream,
             "Step 16: DiD regressions")


# =====================================================================
# Comparison report generator
# =====================================================================
def generate_report():
    """Generate LaTeX comparison report: Logistic LASSO vs Poisson."""
    print("\n" + "=" * 72)
    print("Generating comparison report")
    print("=" * 72)

    # Load training summaries
    logit_summary = pd.read_csv(SOURCE_DIR / "models" / "06_training_summary.csv")
    poiss_summary = pd.read_csv(MODEL_DIR / "06_training_summary.csv")

    # Load DiD results
    logit_did = pd.read_csv(SOURCE_DIR / "output" / "tables" / "did_results.csv")
    poiss_did = pd.read_csv(TAB_DIR / "did_results.csv")

    # Load normalization params
    logit_norm = pd.read_csv(SOURCE_DIR / "newspapers" / "09_normalization_params.csv")
    poiss_norm = pd.read_csv(NEWS_DIR / "09_normalization_params.csv")

    # -- Build LaTeX --
    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs,threeparttable,caption,graphicx,float}")
    lines.append(r"\usepackage{amsmath,amssymb}")
    lines.append(r"\usepackage{setspace}")
    lines.append(r"\singlespacing")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\begin{center}")
    lines.append(r"{\Large\bfseries Poisson vs.\ Logistic LASSO: Comparison Report} \\[12pt]")
    lines.append(r"{\normalsize Generated: \today}")
    lines.append(r"\end{center}")
    lines.append(r"\vspace{12pt}")

    # -- Section 1: Model Diagnostics --
    lines.append(r"\section*{1. Model Diagnostics}")
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Training Summary: Logistic LASSO vs.\ Poisson Log-Ratio}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{@{} c rr rr rr @{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Non-zero features} & \multicolumn{2}{c}{Positive / Negative} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Congress & Logistic & Poisson & Logistic & Poisson & Logistic & Poisson \\")
    lines.append(r"\midrule")

    for _, (lr, pr) in enumerate(zip(logit_summary.itertuples(),
                                      poiss_summary.itertuples())):
        cong = lr.scores_congress
        l_acc = lr.train_accuracy
        p_acc = pr.train_accuracy
        l_nz = lr.n_nonzero_coefs
        p_nz = pr.n_nonzero_coefs
        l_pn = f"+{lr.n_positive_coefs}/$-${lr.n_negative_coefs}"
        p_pn = f"+{pr.n_positive_coefs}/$-${pr.n_negative_coefs}"
        lines.append(
            f"  {cong} & {l_acc:.3f} & {p_acc:.3f} & "
            f"{l_nz:,} & {p_nz:,} & {l_pn} & {p_pn} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # -- Section 2: Normalization Parameters --
    lines.append(r"\section*{2. Normalization Parameters (Partisan Gap)}")
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Partisan Gap ($\mu_R - \mu_D$) by Congress}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{@{} c rr rr rr @{}}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{$\mu_R$} & \multicolumn{2}{c}{$\mu_D$} & \multicolumn{2}{c}{Gap} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Congress & Logistic & Poisson & Logistic & Poisson & Logistic & Poisson \\")
    lines.append(r"\midrule")

    for _, (ln, pn) in enumerate(zip(logit_norm.itertuples(),
                                      poiss_norm.itertuples())):
        cong = ln.congress
        lines.append(
            f"  {cong} & {ln.mu_R:.3f} & {pn.mu_R:.3f} & "
            f"{ln.mu_D:.3f} & {pn.mu_D:.3f} & "
            f"{ln.gap:.3f} & {pn.gap:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # -- Section 3: DiD Results --
    lines.append(r"\section*{3. Difference-in-Differences Results}")
    lines.append("")

    # Build comparison: spec3 (fullest specification) side by side
    def did_row(depvar, label, logit_df, poiss_df, spec="spec3"):
        lr = logit_df[(logit_df["depvar"] == depvar) & (logit_df["spec"] == spec)]
        pr = poiss_df[(poiss_df["depvar"] == depvar) & (poiss_df["spec"] == spec)]
        if lr.empty or pr.empty:
            return None

        lr = lr.iloc[0]
        pr = pr.iloc[0]

        def fmt_coef(row):
            coef = row["coef"]
            se = row["se"]
            pval = row.get("pval", row.get("pvalue", row.get("p_value", 1.0)))
            stars = ""
            if pval < 0.01:
                stars = "$^{***}$"
            elif pval < 0.05:
                stars = "$^{**}$"
            elif pval < 0.10:
                stars = "$^{*}$"
            sign = "$-$" if coef < 0 else ""
            return f"{sign}{abs(coef):.3f}{stars}", f"({se:.3f})"

        l_coef, l_se = fmt_coef(lr)
        p_coef, p_se = fmt_coef(pr)

        return (
            f"  {label}\n"
            f"      & {l_coef} & {p_coef} \\\\\n"
            f"      & {l_se} & {p_se} \\\\"
        )

    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{DiD Estimates (Spec 3): Logistic LASSO vs.\ Poisson}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{@{} l cc @{}}")
    lines.append(r"\toprule")
    lines.append(r"& Logistic & Poisson \\")
    lines.append(r"\midrule")
    lines.append(r"\addlinespace[4pt]")
    lines.append(r"\multicolumn{3}{@{}l}{\emph{Panel A: Aggregate slant}} \\")
    lines.append(r"\addlinespace[3pt]")

    outcomes_A = [
        ("net_slant_norm", r"Net slant ($\tilde{S}$)"),
        ("right_norm", r"Republican slant ($\tilde{R}$)"),
        ("left_norm", r"Democratic slant ($\tilde{L}$)"),
        ("politicization_norm", r"Politicization ($\tilde{P}$)"),
    ]
    for depvar, label in outcomes_A:
        row = did_row(depvar, label, logit_did, poiss_did)
        if row:
            lines.append(row)
            lines.append(r"\addlinespace[2pt]")

    lines.append(r"\addlinespace[4pt]")
    lines.append(r"\multicolumn{3}{@{}l}{\emph{Panel B: Extensive margin}} \\")
    lines.append(r"\addlinespace[3pt]")

    outcomes_B = [
        ("ext_R", "Share R-leaning"),
        ("ext_D", "Share D-leaning"),
        ("ext_nonzero", "Share non-zero"),
    ]
    for depvar, label in outcomes_B:
        row = did_row(depvar, label, logit_did, poiss_did)
        if row:
            lines.append(row)
            lines.append(r"\addlinespace[2pt]")

    lines.append(r"\addlinespace[4pt]")
    lines.append(r"\multicolumn{3}{@{}l}{\emph{Panel C: Intensive margin}} \\")
    lines.append(r"\addlinespace[3pt]")

    outcomes_C = [
        ("int_right_norm", r"Republican slant ($\tilde{R}$)"),
        ("int_left_norm", r"Democratic slant ($\tilde{L}$)"),
    ]
    for depvar, label in outcomes_C:
        row = did_row(depvar, label, logit_did, poiss_did)
        if row:
            lines.append(row)
            lines.append(r"\addlinespace[2pt]")

    lines.append(r"\midrule")
    lines.append(r"Paper FE, Year FE, Div.$\times$Year FE & Yes & Yes \\")
    lines.append(r"Manuf.\ share $\times$ Year & Yes & Yes \\")
    lines.append(r"China shock $\times$ Year & Yes & Yes \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # -- Section 4: Event study figures --
    lines.append(r"\section*{4. Event Study Figures}")
    lines.append("")

    fig_pairs = [
        ("event_study_net_slant_norm.png", "Net Slant (Normalized)"),
        ("event_study_ext_R_vs_D.png", "Extensive Margin: Share R vs.~D"),
        ("event_study_int_R_vs_D.png", "Intensive Margin: R vs.~D"),
        ("event_study_politicization_norm.png", "Politicization (Normalized)"),
    ]

    for figname, caption in fig_pairs:
        logit_fig = SOURCE_DIR / "output" / "figures" / figname
        poiss_fig = FIG_DIR / figname
        if logit_fig.exists() and poiss_fig.exists():
            lines.append(r"\begin{figure}[H]")
            lines.append(r"\centering")
            lines.append(r"\begin{minipage}{0.48\textwidth}")
            lines.append(r"\centering")
            lines.append(r"\includegraphics[width=\textwidth]{" + str(logit_fig).replace("\\", "/") + "}")
            lines.append(r"\subcaption{Logistic LASSO}")
            lines.append(r"\end{minipage}\hfill")
            lines.append(r"\begin{minipage}{0.48\textwidth}")
            lines.append(r"\centering")
            lines.append(r"\includegraphics[width=\textwidth]{" + str(poiss_fig).replace("\\", "/") + "}")
            lines.append(r"\subcaption{Poisson Log-Ratio}")
            lines.append(r"\end{minipage}")
            lines.append(r"\caption{" + caption + "}")
            lines.append(r"\end{figure}")
            lines.append("")

    lines.append(r"\end{document}")

    # Write report
    report_path = EXP_DIR / "poisson_comparison_report.tex"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")

    # Try to compile
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             str(report_path.name)],
            cwd=str(report_path.parent),
            capture_output=True, timeout=60,
        )
        pdf_path = report_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"  PDF compiled: {pdf_path}")
    except Exception as e:
        print(f"  PDF compilation skipped: {e}")

    return report_path


# =====================================================================
# Main
# =====================================================================
def main():
    overall_start = time.time()

    print("=" * 72)
    print("  EXPERIMENT: MNL/Poisson Log-Frequency-Ratio Model")
    print(f"  Source run:  {SOURCE_RUN}")
    print(f"  Experiment:  {EXP_NAME}")
    print("=" * 72)

    # Create directories
    for d in [MODEL_DIR, NEWS_DIR, PANEL_DIR, FIG_DIR, TAB_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Phase 1: Train Poisson models
    train_summary = train_all_windows()

    # Phase 2: Run downstream pipeline
    run_pipeline()

    # Phase 3: Generate comparison report
    report_path = generate_report()

    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 72)
    print(f"EXPERIMENT COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Results: {EXP_DIR}")
    print(f"  Report:  {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
