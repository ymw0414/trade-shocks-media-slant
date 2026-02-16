"""
Quick dev pipeline runner.

Usage:
  python scripts/run_dev.py exp_unigram          # 25% subsample (fast)
  python scripts/run_dev.py exp_unigram --full    # full sample
  python scripts/run_dev.py exp_unigram --from 08 # resume from step 08

Creates a temporary config override with speech_sample_frac=0.25 (unless --full),
then runs the pipeline: 05 -> 07 -> 06 -> 08 -> 09 -> 10 -> 14 -> 15 -> 16
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
RUNS = BASE / "data" / "processed" / "runs"
PYTHON = sys.executable
NLP_DIR = BASE / "scripts" / "nlp"
ECON_DIR = BASE / "scripts" / "econ"
ANALYSIS_DIR = BASE / "scripts" / "analysis"

STEPS = [
    ("05", NLP_DIR / "05_build_features.py"),
    ("07", NLP_DIR / "07_prepare_newspapers.py"),
    ("06", NLP_DIR / "06_train_lasso.py"),
    ("08", NLP_DIR / "08_project_slant.py"),
    ("09", NLP_DIR / "09_normalize_slant.py"),
    ("10", NLP_DIR / "10_aggregate_slant.py"),
    ("11", ECON_DIR / "11_merge_geography.py"),
    ("14", ECON_DIR / "14_merge_panel.py"),
    ("15", ANALYSIS_DIR / "15_event_study.py"),
    ("16", ANALYSIS_DIR / "16_did_regression.py"),
]

DEV_SAMPLE_FRAC = 0.25


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_dev.py <exp_name> [--full] [--from STEP]")
        print("  --full: skip speech subsampling (run on full data)")
        print("  --from STEP: resume from step (e.g. --from 08)")
        sys.exit(1)

    exp_name = sys.argv[1]
    full_mode = "--full" in sys.argv

    # Parse --from
    start_step = None
    if "--from" in sys.argv:
        idx = sys.argv.index("--from")
        if idx + 1 < len(sys.argv):
            start_step = sys.argv[idx + 1]

    # Load existing config override if it exists
    run_dir = RUNS / exp_name
    override_path = run_dir / "config_override.json"

    if override_path.exists():
        with open(override_path) as f:
            config = json.load(f)
        print(f"Loaded existing config: {override_path}")
    else:
        config = {"run_name": exp_name}
        print(f"No existing config, creating new run: {exp_name}")

    # Add dev sampling unless --full
    if not full_mode:
        config["speech_sample_frac"] = DEV_SAMPLE_FRAC
        config["newspaper_sample_frac"] = 0.10  # 10% of articles (~5.6M -> ~560K)
        print(f"[DEV MODE] speech_sample_frac={DEV_SAMPLE_FRAC}, newspaper_sample_frac=0.10")
    else:
        config.pop("speech_sample_frac", None)
        config.pop("newspaper_sample_frac", None)
        print("[FULL MODE] No subsampling")

    # Write temporary config
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_config = run_dir / "_dev_config.json"
    with open(tmp_config, "w") as f:
        json.dump(config, f, indent=2)

    # Set up environment with all pipeline path overrides
    news_dir = run_dir / "newspapers"
    panel_dir = run_dir / "panel"
    fig_dir = run_dir / "output" / "figures"
    tab_dir = run_dir / "output" / "tables"

    for d in [news_dir, panel_dir, fig_dir, tab_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PIPELINE_CONFIG_OVERRIDE"] = str(tmp_config)
    env["PIPELINE_NEWS_DIR"] = str(news_dir)
    env["PIPELINE_PANEL_DIR"] = str(panel_dir)
    env["PIPELINE_FIG_DIR"] = str(fig_dir)
    env["PIPELINE_TAB_DIR"] = str(tab_dir)

    # Determine which steps to run
    skipping = start_step is not None
    steps_to_run = []
    for step_name, script_path in STEPS:
        if skipping:
            if step_name == start_step:
                skipping = False
            else:
                continue
        steps_to_run.append((step_name, script_path))

    if start_step:
        print(f"Resuming from step {start_step}")

    # Run pipeline
    total_start = time.time()
    for step_name, script_path in steps_to_run:
        if not script_path.exists():
            print(f"\n{'='*60}")
            print(f"SKIP step {step_name}: {script_path.name} not found")
            continue

        print(f"\n{'='*60}")
        print(f"STEP {step_name}: {script_path.name}")
        print(f"{'='*60}")

        step_start = time.time()
        result = subprocess.run(
            [PYTHON, str(script_path)],
            env=env,
            cwd=str(NLP_DIR),
        )
        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\nFAILED at step {step_name} ({elapsed:.0f}s)")
            sys.exit(1)
        print(f"  [{step_name} done in {elapsed:.0f}s]")

    total = time.time() - total_start

    # Clean up temp config
    tmp_config.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {exp_name} ({total:.0f}s = {total/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
