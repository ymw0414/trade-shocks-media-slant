"""
run_rebuild_pipeline.py

Re-run the full NLP pipeline from step 06 onward after rebuilding
newspaper features (step 07). Then run 3-window robustness.

Run order:
  06 (main, 2-window) -> 08 -> 09 -> 10 -> 11 -> 14
  06_train_lasso_3window -> robustness_3window
"""

import os
import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PYTHON = sys.executable

SCRIPTS = [
    ("Main: Train LASSO (2-window)", "scripts/nlp/06_train_lasso.py"),
    ("Main: Project slant", "scripts/nlp/08_project_slant.py"),
    ("Main: Normalize slant", "scripts/nlp/09_normalize_slant.py"),
    ("Main: Aggregate to panel", "scripts/nlp/10_aggregate_slant.py"),
    ("Main: Merge geography", "scripts/econ/11_merge_geography.py"),
    ("Main: Merge panel", "scripts/econ/14_merge_panel.py"),
    ("Main: Event study", "scripts/analysis/15_event_study.py"),
    ("Main: DiD regressions", "scripts/analysis/16_did_regression.py"),
    ("Robustness: Train LASSO (3-window)", "scripts/nlp/06_train_lasso_3window.py"),
    ("Robustness: 3-window regressions", "scripts/analysis/robustness_3window.py"),
]

if __name__ == "__main__":
    pipeline_start = time.time()

    for i, (label, script_rel) in enumerate(SCRIPTS, 1):
        script_path = BASE_DIR / script_rel
        print(f"\n{'='*72}")
        print(f"[{i}/{len(SCRIPTS)}] {label}")
        print(f"  {script_path}")
        print(f"{'='*72}\n")

        step_start = time.time()
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(BASE_DIR),
        )

        elapsed = time.time() - step_start
        total = time.time() - pipeline_start

        if result.returncode != 0:
            print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
            print(f"  Total elapsed: {total:.0f}s")
            sys.exit(1)

        print(f"\n  Completed in {elapsed:.0f}s (total: {total:.0f}s)")

    total = time.time() - pipeline_start
    print(f"\n{'='*72}")
    print(f"ALL DONE in {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*72}")
