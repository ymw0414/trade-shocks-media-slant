"""
run_slant_experiments.py

Run multiple LASSO pipeline experiments to find the best configuration.
All experiments use GST procedural filter.

Experiments (8 total):
  2 aggregation levels (speech / legislator)
  × 2 training samples (all R/D / core 20%)
  × 2 lambda selection (BIC / CV)

Phase 1: Run step 05 (TF-IDF) for 2 aggregation levels
Phase 2: Run step 06 (LASSO) for 8 configs, reusing step 05 outputs
Phase 3: Compare training accuracies → pick top configs
Phase 4: Run steps 07-10 for top configs
Phase 5: Face validity evaluation
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
SCRIPTS_DIR = BASE_DIR / "scripts" / "nlp"
PYTHON = sys.executable
EXP_DIR = BASE_DIR / "data" / "processed" / "runs"
CONFIG_DIR = BASE_DIR / "scripts" / "experiment_configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Base config shared by all experiments ───────────────────────────
BASE_CONFIG = {
    "congress_range": [99, 108],
    "window_size": 1,
    "ngram_range": [2, 2],
    "bigrams_only": True,
    "use_relative_freq": True,
    "freq_filter_mode": "widmer",
    "filter_gst_procedural": True,       # ALL experiments use GST filter
    "sign_consistency_filter": False,
    "shared_vocab_min_df": None,
    "norm_method": "raw_gap",
    "speech_sample_frac": None,           # full speech data
    "newspaper_sample_frac": None,
    "lasso_max_iter": 5000,
    "lasso_cv_folds": 5,
    "lasso_Cs": [-1, 4, 20],
    "min_speech_words": None,
    "newspaper_df_floor": None,
}


def make_config(run_name, aggregate, core_only, lambda_sel, input_speech_dir=None):
    """Create experiment config dict."""
    cfg = dict(BASE_CONFIG)
    cfg["run_name"] = run_name
    cfg["aggregate_to_legislator"] = aggregate
    cfg["partisan_core_only"] = core_only
    cfg["lasso_lambda_selection"] = lambda_sel
    if input_speech_dir:
        cfg["input_speech_dir"] = input_speech_dir
    return cfg


def save_config(name, cfg):
    """Save config JSON and return path."""
    path = CONFIG_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return str(path)


def run_step(step_num, config_path, timeout=1800):
    """Run a pipeline step with config override."""
    step_file = {
        5: "05_build_tfidf.py",
        6: "06_train_lasso.py",
        7: "07_prepare_newspapers.py",
        8: "08_compute_slant.py",
        9: "09_normalize_slant.py",
        10: "10_aggregate_slant.py",
    }[step_num]

    env = os.environ.copy()
    env["PIPELINE_CONFIG_OVERRIDE"] = config_path

    cmd = [PYTHON, str(SCRIPTS_DIR / step_file)]
    print(f"\n{'='*60}")
    print(f"  Running step {step_num:02d} with config: {Path(config_path).stem}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        timeout=timeout, cwd=str(SCRIPTS_DIR),
        encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s)")
        print(f"  STDERR: {result.stderr[-1000:]}")
        return False
    else:
        # Print last few lines of output
        lines = result.stdout.strip().split("\n")
        for line in lines[-8:]:
            print(f"  {line}")
        print(f"  OK ({elapsed:.0f}s)")
        return True


def collect_results():
    """Collect training summaries from all experiment runs."""
    import pandas as pd

    results = []
    for run_dir in sorted(EXP_DIR.glob("exp2_*")):
        summary_path = run_dir / "models" / "06_training_summary.csv"
        config_path = run_dir / "config.json"

        if not summary_path.exists():
            continue

        summary = pd.read_csv(summary_path)
        with open(config_path) as f:
            cfg = json.load(f)

        avg_acc = summary["train_accuracy"].mean()
        avg_nonzero = summary["n_nonzero_coefs"].mean()

        results.append({
            "run_name": run_dir.name,
            "aggregate": cfg.get("aggregate_to_legislator", False),
            "core_only": cfg.get("partisan_core_only", False),
            "lambda_sel": cfg.get("lasso_lambda_selection", "bic"),
            "avg_accuracy": avg_acc,
            "avg_nonzero_coefs": avg_nonzero,
            "min_accuracy": summary["train_accuracy"].min(),
            "max_accuracy": summary["train_accuracy"].max(),
        })

    return pd.DataFrame(results).sort_values("avg_accuracy", ascending=False)


# ══════════════════════════════════════════════════════════════════════
# Define experiments
# ══════════════════════════════════════════════════════════════════════

# Phase 1: Step 05 runs (2 unique TF-IDF builds)
STEP05_RUNS = {
    "exp2_tfidf_speech": make_config(
        "exp2_tfidf_speech", aggregate=False, core_only=False, lambda_sel="bic"
    ),
    "exp2_tfidf_leg": make_config(
        "exp2_tfidf_leg", aggregate=True, core_only=False, lambda_sel="bic"
    ),
}

# Phase 2: Step 06 runs (8 LASSO experiments)
# input_speech_dir paths are relative to BASE_DIR
SPEECH_INPUT = "data/processed/runs/exp2_tfidf_speech/speeches"
LEG_INPUT = "data/processed/runs/exp2_tfidf_leg/speeches"

STEP06_RUNS = {
    # Speech-level group
    "exp2_speech_all_bic": make_config(
        "exp2_speech_all_bic", aggregate=False, core_only=False,
        lambda_sel="bic", input_speech_dir=SPEECH_INPUT),
    "exp2_speech_all_cv": make_config(
        "exp2_speech_all_cv", aggregate=False, core_only=False,
        lambda_sel="cv", input_speech_dir=SPEECH_INPUT),
    "exp2_speech_core_bic": make_config(
        "exp2_speech_core_bic", aggregate=False, core_only=True,
        lambda_sel="bic", input_speech_dir=SPEECH_INPUT),
    "exp2_speech_core_cv": make_config(
        "exp2_speech_core_cv", aggregate=False, core_only=True,
        lambda_sel="cv", input_speech_dir=SPEECH_INPUT),
    # Legislator-level group
    "exp2_leg_all_bic": make_config(
        "exp2_leg_all_bic", aggregate=True, core_only=False,
        lambda_sel="bic", input_speech_dir=LEG_INPUT),
    "exp2_leg_all_cv": make_config(
        "exp2_leg_all_cv", aggregate=True, core_only=False,
        lambda_sel="cv", input_speech_dir=LEG_INPUT),
    "exp2_leg_core_bic": make_config(
        "exp2_leg_core_bic", aggregate=True, core_only=True,
        lambda_sel="bic", input_speech_dir=LEG_INPUT),
    "exp2_leg_core_cv": make_config(
        "exp2_leg_core_cv", aggregate=True, core_only=True,
        lambda_sel="cv", input_speech_dir=LEG_INPUT),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0,
                        help="1=step05, 2=step06, 3=results, 4=full pipeline for top")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of top configs for full pipeline")
    args = parser.parse_args()

    if args.phase == 0 or args.phase == 1:
        print("\n" + "=" * 60)
        print("  PHASE 1: Building TF-IDF feature matrices (step 05)")
        print("=" * 60)
        for name, cfg in STEP05_RUNS.items():
            config_path = save_config(name, cfg)
            run_step(5, config_path, timeout=3600)

    if args.phase == 0 or args.phase == 2:
        print("\n" + "=" * 60)
        print("  PHASE 2: Training LASSO models (step 06)")
        print("=" * 60)
        for name, cfg in STEP06_RUNS.items():
            config_path = save_config(name, cfg)
            run_step(6, config_path, timeout=1800)

    if args.phase == 0 or args.phase == 3:
        print("\n" + "=" * 60)
        print("  PHASE 3: Comparing results")
        print("=" * 60)
        results = collect_results()
        if len(results) > 0:
            print("\n" + results.to_string(index=False))
            print(f"\n  Best: {results.iloc[0]['run_name']} "
                  f"(accuracy={results.iloc[0]['avg_accuracy']:.1%})")
        else:
            print("  No results found yet.")

    if args.phase == 4:
        print("\n" + "=" * 60)
        print(f"  PHASE 4: Full pipeline for top {args.top_n} configs")
        print("=" * 60)
        results = collect_results()
        top_names = results.head(args.top_n)["run_name"].tolist()

        for name in top_names:
            cfg = STEP06_RUNS.get(name)
            if cfg is None:
                print(f"  WARNING: {name} not in STEP06_RUNS, skipping")
                continue
            config_path = save_config(name, cfg)
            for step in [7, 8, 9, 10]:
                ok = run_step(step, config_path, timeout=3600)
                if not ok:
                    print(f"  Stopping pipeline for {name} at step {step}")
                    break
