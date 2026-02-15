"""
run_all_cheap.py

Run all "cheap" experiments (those that reuse step 05/07 outputs from main).
Then run the comparison script to rank them.

Usage:
  python run_all_cheap.py [--only exp1,exp2,exp3]
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of experiments to run")
    args = parser.parse_args()

    with open(EXPERIMENTS_FILE) as f:
        experiments = json.load(f)

    # Filter to cheap experiments
    cheap = {k: v for k, v in experiments.items() if not v.get("requires_full", False)}

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        cheap = {k: v for k, v in cheap.items() if k in names}

    print(f"Running {len(cheap)} cheap experiments:")
    for name, exp in cheap.items():
        print(f"  - {name}: {exp.get('description', '')}")

    results = {}
    total_start = time.time()

    for name in cheap:
        print(f"\n\n{'#' * 70}")
        print(f"  STARTING EXPERIMENT: {name}")
        print(f"{'#' * 70}\n")

        exp_start = time.time()
        result = subprocess.run(
            [PYTHON, str(SCRIPTS_DIR / "finetuning" / "run_experiment.py"), name],
            cwd=str(SCRIPTS_DIR / "finetuning"),
        )
        elapsed = time.time() - exp_start
        results[name] = {
            "success": result.returncode == 0,
            "elapsed": elapsed,
        }

        status = "OK" if result.returncode == 0 else "FAILED"
        print(f"\n  {name}: {status} ({elapsed:.0f}s)")

    # Summary
    total = time.time() - total_start
    print(f"\n\n{'#' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE ({total:.0f}s = {total/60:.1f}min)")
    print(f"{'#' * 70}")

    for name, r in results.items():
        status = "OK" if r["success"] else "FAILED"
        print(f"  {name:<25} {status:>8}  {r['elapsed']:>6.0f}s")

    n_ok = sum(1 for r in results.values() if r["success"])
    print(f"\n  {n_ok}/{len(results)} succeeded")

    # Run comparison if at least 1 succeeded
    if n_ok > 0:
        print(f"\n\nRunning comparison analysis ...")
        subprocess.run(
            [PYTHON, str(SCRIPTS_DIR / "finetuning" / "compare_experiments.py")],
            cwd=str(SCRIPTS_DIR / "finetuning"),
        )


if __name__ == "__main__":
    main()
