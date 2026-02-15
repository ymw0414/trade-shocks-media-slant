"""
run_experiments.py

Run 6 experiment variants for unigram+bigram + GST filter model:
  1. speech-level, all R/D, BIC      (exp_unigram_gst)
  2. legislator-level, all R/D, BIC  (exp_uni_gst_leg)
  3. legislator-level, core 20%, BIC (exp_uni_gst_leg_core)
  4. speech-level, all R/D, CV       (exp_uni_gst_cv)
  5. legislator-level, all R/D, CV   (exp_uni_gst_leg_cv)
  6. legislator-level, core 20%, CV  (exp_uni_gst_leg_core_cv)

Strategy:
  - 2x step 05 (speech-level + legislator-level)
  - 2x step 07 (one per vocabulary)
  - 6x step 06 (one per experiment)
  - 6x steps 08-16 (one per experiment)

Usage:
  python scripts/run_experiments.py [--from STEP] [--only EXP_NUMS]

Examples:
  python scripts/run_experiments.py              # run all from scratch
  python scripts/run_experiments.py --from 06    # skip step 05/07, run from step 06
  python scripts/run_experiments.py --only 1,4   # only run experiments 1 and 4
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get("SHIFTING_SLANT_DIR", "C:/Users/ymw04/Dropbox/shifting_slant"))
PYTHON = sys.executable
SCRIPTS = BASE / "scripts"

# ── Experiment definitions ──────────────────────────────────────────
EXPERIMENTS = {
    1: {
        "run_name": "exp_unigram_gst",
        "label": "Speech, All R/D, BIC",
        "bigrams_only": False,
        "filter_gst_procedural": True,
        "aggregate_to_legislator": False,
        "partisan_core_only": False,
        "lasso_lambda_selection": "bic",
        "group": "speech",  # shares step 05/07 with group
    },
    2: {
        "run_name": "exp_uni_gst_leg",
        "label": "Legislator, All R/D, BIC",
        "bigrams_only": False,
        "filter_gst_procedural": True,
        "aggregate_to_legislator": True,
        "partisan_core_only": False,
        "lasso_lambda_selection": "bic",
        "group": "legislator",
    },
    3: {
        "run_name": "exp_uni_gst_leg_core",
        "label": "Legislator, Core 20%, BIC",
        "bigrams_only": False,
        "filter_gst_procedural": True,
        "aggregate_to_legislator": True,
        "partisan_core_only": True,
        "lasso_lambda_selection": "bic",
        "group": "legislator",
    },
    4: {
        "run_name": "exp_uni_gst_cv",
        "label": "Speech, All R/D, CV",
        "bigrams_only": False,
        "filter_gst_procedural": True,
        "aggregate_to_legislator": False,
        "partisan_core_only": False,
        "lasso_lambda_selection": "cv",
        "group": "speech",
    },
    5: {
        "run_name": "exp_uni_gst_leg_cv",
        "label": "Legislator, All R/D, CV",
        "bigrams_only": False,
        "filter_gst_procedural": True,
        "aggregate_to_legislator": True,
        "partisan_core_only": False,
        "lasso_lambda_selection": "cv",
        "group": "legislator",
    },
    6: {
        "run_name": "exp_uni_gst_leg_core_cv",
        "label": "Legislator, Core 20%, CV",
        "bigrams_only": False,
        "filter_gst_procedural": True,
        "aggregate_to_legislator": True,
        "partisan_core_only": True,
        "lasso_lambda_selection": "cv",
        "group": "legislator",
    },
}

# Group leaders: experiments that run step 05 and step 07
# Other experiments in the same group reuse the leader's speech/newspaper data
GROUP_LEADERS = {
    "speech": 1,      # exp_unigram_gst runs step 05/07
    "legislator": 2,  # exp_uni_gst_leg runs step 05/07
}


def make_override(exp_cfg, override_path):
    """Write experiment override JSON."""
    override = {
        "run_name": exp_cfg["run_name"],
        "bigrams_only": exp_cfg["bigrams_only"],
        "filter_gst_procedural": exp_cfg["filter_gst_procedural"],
        "aggregate_to_legislator": exp_cfg["aggregate_to_legislator"],
        "partisan_core_only": exp_cfg["partisan_core_only"],
        "lasso_lambda_selection": exp_cfg["lasso_lambda_selection"],
        "speech_sample_frac": None,
        "newspaper_sample_frac": None,
    }
    # If not the group leader, point to the leader's outputs
    group = exp_cfg["group"]
    leader_num = GROUP_LEADERS[group]
    leader_name = EXPERIMENTS[leader_num]["run_name"]

    if exp_cfg["run_name"] != leader_name:
        # Point to leader's speech/newspaper directories
        leader_speech = f"data/processed/runs/{leader_name}/speeches"
        leader_news = f"data/processed/runs/{leader_name}/newspapers"
        override["input_speech_dir"] = leader_speech
        override["input_news_dir"] = leader_news

    override_path.parent.mkdir(parents=True, exist_ok=True)
    with open(override_path, "w") as f:
        json.dump(override, f, indent=2)
    return override


def run_step(step_script, override_path, label=""):
    """Run a pipeline step with the given override."""
    env = os.environ.copy()
    env["SHIFTING_SLANT_DIR"] = str(BASE)
    env["PIPELINE_CONFIG_OVERRIDE"] = str(override_path)
    env["PYTHONPATH"] = str(SCRIPTS / "utils")

    script_path = SCRIPTS / step_script
    print(f"\n{'='*60}")
    print(f"  Running {step_script} [{label}]")
    print(f"  Override: {override_path.name}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        env=env,
        cwd=str(BASE),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        return False
    print(f"  Completed in {elapsed:.0f}s")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_step", default="05",
                        help="Start from this step (default: 05)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated experiment numbers (default: all)")
    args = parser.parse_args()

    from_step = int(args.from_step)
    if args.only:
        exp_nums = [int(x) for x in args.only.split(",")]
    else:
        exp_nums = list(EXPERIMENTS.keys())

    print(f"Running experiments: {exp_nums}")
    print(f"Starting from step: {from_step:02d}")

    exp_dir = BASE / "experiments"
    exp_dir.mkdir(exist_ok=True)

    # ── Phase 1: Step 05 for group leaders ──
    if from_step <= 5:
        groups_needed = set(EXPERIMENTS[n]["group"] for n in exp_nums)
        for group in sorted(groups_needed):
            leader_num = GROUP_LEADERS[group]
            leader_cfg = EXPERIMENTS[leader_num]
            override_path = exp_dir / f"{leader_cfg['run_name']}.json"
            make_override(leader_cfg, override_path)
            ok = run_step("nlp/05_build_tfidf.py", override_path,
                          label=f"Step 05 [{leader_cfg['label']}]")
            if not ok:
                print(f"ABORT: Step 05 failed for {leader_cfg['run_name']}")
                sys.exit(1)

    # ── Phase 2: Step 07 for group leaders ──
    if from_step <= 7:
        groups_needed = set(EXPERIMENTS[n]["group"] for n in exp_nums)
        for group in sorted(groups_needed):
            leader_num = GROUP_LEADERS[group]
            leader_cfg = EXPERIMENTS[leader_num]
            override_path = exp_dir / f"{leader_cfg['run_name']}.json"
            # Ensure override exists
            if not override_path.exists():
                make_override(leader_cfg, override_path)
            ok = run_step("nlp/07_prepare_newspapers.py", override_path,
                          label=f"Step 07 [{leader_cfg['label']}]")
            if not ok:
                print(f"ABORT: Step 07 failed for {leader_cfg['run_name']}")
                sys.exit(1)

    # ── Phase 3: Step 06 for each experiment ──
    if from_step <= 6:
        for n in exp_nums:
            cfg = EXPERIMENTS[n]
            override_path = exp_dir / f"{cfg['run_name']}.json"
            make_override(cfg, override_path)
            ok = run_step("nlp/06_train_lasso.py", override_path,
                          label=f"Step 06 [{cfg['label']}]")
            if not ok:
                print(f"WARNING: Step 06 failed for {cfg['run_name']}")

    # ── Phase 4: Steps 08-16 for each experiment ──
    post_steps = [
        "nlp/08_project_slant.py",
        "nlp/09_normalize_slant.py",
        "nlp/10_aggregate_slant.py",
        "econ/11_merge_geography.py",
        "econ/14_merge_panel.py",
        "analysis/15_event_study.py",
        "analysis/16_did_regression.py",
    ]

    for n in exp_nums:
        cfg = EXPERIMENTS[n]
        override_path = exp_dir / f"{cfg['run_name']}.json"

        for step_script in post_steps:
            step_num = int(step_script.split("/")[1][:2])
            if step_num < from_step:
                continue
            ok = run_step(step_script, override_path,
                          label=f"{cfg['label']}")
            if not ok:
                print(f"WARNING: {step_script} failed for {cfg['run_name']}")
                break

    # ── Phase 5: Comparison summary ──
    print("\n" + "=" * 72)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 72)

    import pandas as pd
    results = []
    for n in exp_nums:
        cfg = EXPERIMENTS[n]
        did_path = (BASE / "data" / "processed" / "runs" / cfg["run_name"]
                    / "output" / "tables" / "did_results.csv")
        if did_path.exists():
            df = pd.read_csv(did_path)
            df["experiment"] = cfg["run_name"]
            df["exp_label"] = cfg["label"]
            results.append(df)

    if results:
        all_did = pd.concat(results, ignore_index=True)
        summary_path = BASE / "output" / "tables" / "experiment_comparison.csv"
        all_did.to_csv(summary_path, index=False)
        print(f"\nComparison saved -> {summary_path}")

        # Print summary for key outcomes
        key_outcomes = ["int_R", "int_D", "net_slant_norm", "ext_nonzero"]
        for outcome in key_outcomes:
            sub = all_did[(all_did["depvar"] == outcome) & (all_did["spec"] == "spec1")]
            if len(sub) > 0:
                print(f"\n  {outcome} (Spec 1):")
                for _, row in sub.iterrows():
                    sig = "**" if row["pval"] < 0.05 else "*" if row["pval"] < 0.1 else ""
                    print(f"    {row['exp_label']:35s}  b={row['coef']:+.4f}  "
                          f"p={row['pval']:.3f}{sig}")


if __name__ == "__main__":
    main()
