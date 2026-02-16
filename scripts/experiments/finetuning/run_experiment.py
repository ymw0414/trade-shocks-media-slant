"""
run_experiment.py

Run a single NLP pipeline experiment with custom configuration.

For "cheap" experiments (reuse step 05/07 outputs from main run):
  python run_experiment.py exp_win2

For "expensive" experiments (re-run step 05 and 07):
  python run_experiment.py exp_unigrams --full

Experiment configs are defined in experiments.json.
Results go to data/processed/runs/{exp_name}/.

Usage:
  python run_experiment.py <exp_name> [--full] [--from-step N]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
SCRIPTS_DIR = BASE_DIR / "scripts"
EXPERIMENTS_FILE = SCRIPTS_DIR / "finetuning" / "experiments.json"
PYTHON = sys.executable


def load_experiments():
    with open(EXPERIMENTS_FILE) as f:
        return json.load(f)


def run_step(script_path, env, label, cwd=None):
    """Run a pipeline step as a subprocess."""
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {label}")
    print(f"  Script:  {script_path}")
    print(f"{'=' * 70}\n")

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        env=env,
        cwd=str(cwd or script_path.parent),
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit code {result.returncode})")
        return False

    print(f"\n  {label} completed in {elapsed:.1f}s")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", help="Experiment name from experiments.json")
    parser.add_argument("--full", action="store_true",
                        help="Re-run steps 05 and 07 (expensive)")
    parser.add_argument("--from-step", type=int, default=None,
                        help="Start from this step number")
    args = parser.parse_args()

    experiments = load_experiments()
    if args.exp_name not in experiments:
        print(f"Unknown experiment: {args.exp_name}")
        print(f"Available: {list(experiments.keys())}")
        sys.exit(1)

    exp = experiments[args.exp_name]
    run_name = args.exp_name
    overrides = exp.get("overrides", {})
    is_full = args.full or exp.get("requires_full", False)

    print(f"\n{'#' * 70}")
    print(f"  EXPERIMENT: {run_name}")
    print(f"  Description: {exp.get('description', '')}")
    print(f"  Mode: {'FULL (05->07->06->08->09->10->11->14->15->16)' if is_full else 'CHEAP (06->08->09->10->11->14->15->16)'}")
    print(f"  Overrides: {json.dumps(overrides, indent=4)}")
    print(f"{'#' * 70}\n")

    # Build override config
    config_override = {"run_name": run_name}
    config_override.update(overrides)

    # For cheap experiments: point to main run's step 05/07 outputs
    if not is_full:
        config_override.setdefault("input_speech_dir", "data/processed/speeches")
        config_override.setdefault("input_news_dir", "data/processed/newspapers")

    # Write override JSON to run directory
    run_dir = BASE_DIR / "data" / "processed" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    override_path = run_dir / "config_override.json"
    with open(override_path, "w") as f:
        json.dump(config_override, f, indent=2)

    # Build environment with override
    env = os.environ.copy()
    env["PIPELINE_CONFIG_OVERRIDE"] = str(override_path)

    # For steps 11, 14, 15, 16: set path env vars
    news_dir = run_dir / "newspapers"
    panel_dir = run_dir / "panel"
    fig_dir = run_dir / "output" / "figures"
    tab_dir = run_dir / "output" / "tables"

    news_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    env["PIPELINE_NEWS_DIR"] = str(news_dir)
    env["PIPELINE_PANEL_DIR"] = str(panel_dir)
    env["PIPELINE_FIG_DIR"] = str(fig_dir)
    env["PIPELINE_TAB_DIR"] = str(tab_dir)

    # Define pipeline steps
    steps = []
    if is_full:
        steps.append((5, SCRIPTS_DIR / "nlp" / "05_build_features.py", "Step 05: Build features"))
        steps.append((7, SCRIPTS_DIR / "nlp" / "07_prepare_newspapers.py", "Step 07: Transform newspapers"))

    steps.extend([
        (6,  SCRIPTS_DIR / "nlp" / "06_train_lasso.py",       "Step 06: Train LASSO"),
        (8,  SCRIPTS_DIR / "nlp" / "08_project_slant.py",     "Step 08: Project slant"),
        (9,  SCRIPTS_DIR / "nlp" / "09_normalize_slant.py",   "Step 09: Normalize slant"),
        (10, SCRIPTS_DIR / "nlp" / "10_aggregate_slant.py",   "Step 10: Aggregate to panel"),
        (11, SCRIPTS_DIR / "econ" / "11_merge_geography.py",  "Step 11: Merge geography"),
        (14, SCRIPTS_DIR / "econ" / "14_merge_panel.py",      "Step 14: Merge regression panel"),
        (15, SCRIPTS_DIR / "analysis" / "15_event_study.py",  "Step 15: Event study"),
        (16, SCRIPTS_DIR / "analysis" / "16_did_regression.py","Step 16: DiD regressions"),
    ])

    # Filter by --from-step
    if args.from_step is not None:
        steps = [(n, p, l) for n, p, l in steps if n >= args.from_step]

    # Run pipeline
    pipeline_start = time.time()
    for step_num, script_path, label in steps:
        # Steps using pipeline_config need nlp/ in path
        cwd = SCRIPTS_DIR / "nlp" if step_num in (5, 6, 7, 8, 9, 10) else script_path.parent
        ok = run_step(script_path, env, label, cwd=cwd)
        if not ok:
            print(f"\nPipeline FAILED at {label}")
            sys.exit(1)

    total = time.time() - pipeline_start
    print(f"\n{'#' * 70}")
    print(f"  EXPERIMENT '{run_name}' COMPLETED in {total:.0f}s ({total/60:.1f}min)")
    print(f"  Results:  {run_dir}")
    print(f"  Figures:  {fig_dir}")
    print(f"  Tables:   {tab_dir}")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
