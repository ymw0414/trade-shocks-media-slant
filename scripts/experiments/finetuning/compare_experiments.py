"""
compare_experiments.py

Compare event study results across all completed experiments.
Produces:
  1. A summary table ranking experiments by pre-trend quality
  2. Side-by-side event study plots for key outcomes
  3. A combined comparison figure

Metrics for "pretty" event study plots:
  - Pre-trend flatness: mean |coef| in pre-NAFTA years (lower = better)
  - Pre-trend max: max |coef| in pre-NAFTA years (lower = better)
  - Post-trend strength: mean |coef| in post-NAFTA years (higher = better)
  - Post-significance: fraction of post-NAFTA coefficients significant at 5%
  - Pre/post ratio: post_strength / pre_flatness (higher = cleaner break)

Usage:
  python compare_experiments.py [--outcome net_slant_norm_econ]
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
RUNS_DIR = BASE_DIR / "data" / "processed" / "runs"
OUT_DIR = BASE_DIR / "output" / "finetuning"

BASE_YEAR = 1993
NAFTA_YEAR = 1994

# Key outcomes to compare
KEY_OUTCOMES = [
    ("net_slant_norm", "Net Slant"),
    ("politicization_norm", "Politicization"),
    ("right_norm_econ", "Right Intensity (Econ)"),
    ("politicization_norm_econ", "Politicization (Econ)"),
    ("econ_share", "Econ Article Share"),
    ("net_slant_norm_econ", "Net Slant (Econ)"),
]


def load_experiment_results(run_name):
    """Load event study coefficients for an experiment."""
    if run_name == "main":
        tab_dir = BASE_DIR / "output" / "tables"
    else:
        tab_dir = RUNS_DIR / run_name / "output" / "tables"

    es_path = tab_dir / "event_study_coefficients.csv"
    did_path = tab_dir / "did_results.csv"

    if not es_path.exists():
        return None, None

    es = pd.read_csv(es_path)
    did = pd.read_csv(did_path) if did_path.exists() else None
    return es, did


def compute_quality_metrics(coefs_df, depvar, spec="controls"):
    """Compute event study quality metrics for one outcome."""
    sub = coefs_df[(coefs_df["depvar"] == depvar) & (coefs_df["spec"] == spec)].copy()
    if len(sub) == 0:
        return None

    pre = sub[sub["year"] < NAFTA_YEAR]
    post = sub[sub["year"] >= NAFTA_YEAR]

    # Exclude base year from pre
    pre = pre[pre["year"] != BASE_YEAR]

    if len(pre) == 0 or len(post) == 0:
        return None

    pre_abs = pre["coef"].abs()
    post_abs = post["coef"].abs()

    # Significance: CI doesn't cross zero
    post_sig = ((post["ci_lo"] > 0) | (post["ci_hi"] < 0)).mean()
    pre_sig = ((pre["ci_lo"] > 0) | (pre["ci_hi"] < 0)).mean()

    pre_flatness = pre_abs.mean()
    pre_max = pre_abs.max()
    post_strength = post_abs.mean()
    ratio = post_strength / max(pre_flatness, 1e-6)

    return {
        "pre_flatness": pre_flatness,
        "pre_max": pre_max,
        "post_strength": post_strength,
        "post_sig_pct": post_sig * 100,
        "pre_sig_pct": pre_sig * 100,
        "ratio": ratio,
    }


def plot_comparison(all_results, depvar, label, out_path):
    """Plot event study comparison across experiments."""
    fig, axes = plt.subplots(2, max(3, (len(all_results) + 1) // 2),
                             figsize=(18, 10), squeeze=False)
    axes = axes.flatten()

    for i, (run_name, (es, _)) in enumerate(all_results.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        sub = es[(es["depvar"] == depvar) & (es["spec"] == "controls")]
        if len(sub) == 0:
            ax.set_title(run_name, fontsize=10)
            continue

        yrs = sub["year"].values
        coef = sub["coef"].values
        ci_lo = sub["ci_lo"].values
        ci_hi = sub["ci_hi"].values

        ax.fill_between(yrs, ci_lo, ci_hi, alpha=0.2, color="#2d2d2d")
        ax.plot(yrs, coef, "o-", color="#2d2d2d", markersize=3, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.7)

        metrics = compute_quality_metrics(es, depvar)
        subtitle = ""
        if metrics:
            subtitle = f"pre={metrics['pre_flatness']:.3f}  post={metrics['post_strength']:.3f}  ratio={metrics['ratio']:.1f}"

        ax.set_title(f"{run_name}\n{subtitle}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for j in range(len(all_results), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Event Study Comparison: {label} (with controls)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", default=None,
                        help="Single outcome to compare (default: all key outcomes)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all completed experiments
    print("Scanning for completed experiments ...\n")
    all_results = {}

    # Always include main
    es, did = load_experiment_results("main")
    if es is not None:
        all_results["main"] = (es, did)
        print(f"  main: {len(es)} rows")

    # Scan run directories
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "main":
                continue
            es, did = load_experiment_results(run_dir.name)
            if es is not None:
                all_results[run_dir.name] = (es, did)
                print(f"  {run_dir.name}: {len(es)} rows")

    if len(all_results) < 2:
        print("\nNeed at least 2 experiments to compare. Run some experiments first.")
        return

    # Compute quality metrics
    outcomes = [(args.outcome, args.outcome)] if args.outcome else KEY_OUTCOMES

    print(f"\n{'=' * 90}")
    print("QUALITY METRICS (controls spec)")
    print(f"{'=' * 90}")

    for depvar, label in outcomes:
        print(f"\n  {label} ({depvar}):")
        print(f"  {'Experiment':<22} {'Pre Flat':>9} {'Pre Max':>9} {'Post Str':>9} "
              f"{'Post Sig%':>9} {'Pre Sig%':>9} {'Ratio':>7}")
        print(f"  {'-' * 78}")

        ranked = []
        for run_name, (es, _) in all_results.items():
            metrics = compute_quality_metrics(es, depvar)
            if metrics:
                ranked.append((run_name, metrics))

        # Sort by ratio (higher = better pre/post contrast)
        ranked.sort(key=lambda x: x[1]["ratio"], reverse=True)

        for run_name, m in ranked:
            marker = " <<<" if run_name == ranked[0][0] else ""
            print(f"  {run_name:<22} {m['pre_flatness']:>9.4f} {m['pre_max']:>9.4f} "
                  f"{m['post_strength']:>9.4f} {m['post_sig_pct']:>8.1f}% "
                  f"{m['pre_sig_pct']:>8.1f}% {m['ratio']:>7.1f}{marker}")

    # Generate comparison plots
    print(f"\nGenerating comparison plots ...")
    for depvar, label in outcomes:
        out_path = OUT_DIR / f"compare_{depvar}.png"
        plot_comparison(all_results, depvar, label, out_path)

    # DiD summary
    print(f"\n{'=' * 90}")
    print("DiD RESULTS (spec3: full controls)")
    print(f"{'=' * 90}")

    did_outcomes = [
        ("net_slant_norm", "Net Slant"),
        ("politicization_norm_econ", "Polit. (Econ)"),
        ("econ_share", "Econ Share"),
        ("right_norm_econ", "Right (Econ)"),
    ]

    print(f"\n  {'Experiment':<22}", end="")
    for depvar, label in did_outcomes:
        print(f" {label:>16}", end="")
    print()
    print(f"  {'-' * (22 + 16 * len(did_outcomes))}")

    for run_name, (_, did) in all_results.items():
        if did is None:
            continue
        print(f"  {run_name:<22}", end="")
        for depvar, label in did_outcomes:
            row = did[(did["depvar"] == depvar) & (did["spec"] == "spec3")]
            if len(row) == 0:
                print(f" {'N/A':>16}", end="")
            else:
                r = row.iloc[0]
                stars = ""
                if r["pval"] < 0.01:
                    stars = "***"
                elif r["pval"] < 0.05:
                    stars = "**"
                elif r["pval"] < 0.1:
                    stars = "*"
                print(f" {r['coef']:>+8.4f}{stars:>3} ({r['se']:.4f})", end="")
        print()

    # Save ranking
    ranking_rows = []
    for depvar, label in KEY_OUTCOMES:
        for run_name, (es, _) in all_results.items():
            metrics = compute_quality_metrics(es, depvar)
            if metrics:
                ranking_rows.append({"outcome": depvar, "experiment": run_name, **metrics})

    if ranking_rows:
        ranking_df = pd.DataFrame(ranking_rows)
        ranking_path = OUT_DIR / "experiment_ranking.csv"
        ranking_df.to_csv(ranking_path, index=False, float_format="%.4f")
        print(f"\n  Ranking saved: {ranking_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
