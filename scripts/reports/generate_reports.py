"""
Generate per-experiment model reports in /report/ folder.

Each report contains:
  - Model settings & difference from main
  - Training diagnostics (accuracy, features per congress)
  - Top 10 positive/negative betas (for 3 representative congresses)
  - Normalization parameters (partisan gap before/after)
  - DiD results: raw + normalized where available
  - Event study highlights
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
RUNS = BASE / "data" / "processed" / "runs"
REPORT_DIR = BASE / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── Main run defaults ─────────────────────────────────────────────
MAIN_DEFAULTS = {
    "window_size": 1,
    "partisan_core_only": False,
    "lasso_lambda_selection": "bic",
    "bigrams_only": True,
    "aggregate_to_legislator": False,
    "freq_filter_mode": "widmer",
    "use_relative_freq": True,
    "newspaper_df_floor": None,
    "shared_vocab_min_df": None,
    "sign_consistency_filter": False,
    "norm_method": "raw_gap",
}

SETTING_LABELS = {
    "window_size": "Rolling window size",
    "partisan_core_only": "Train on partisan core only",
    "lasso_lambda_selection": "Lambda selection method",
    "bigrams_only": "Bigrams only",
    "aggregate_to_legislator": "Aggregate to legislator-congress",
    "use_relative_freq": "Relative frequency vectorizer",
    "freq_filter_mode": "Frequency filter mode",
    "newspaper_df_floor": "Newspaper DF floor",
    "shared_vocab_min_df": "Shared vocab min DF",
    "sign_consistency_filter": "Sign consistency filter",
    "norm_method": "Normalization method",
}

# ── Paths for each run ────────────────────────────────────────────
def get_paths(run_name):
    """Return dict of file paths for a run."""
    if run_name == "main":
        run_dir = RUNS / "main"
        model_dir = BASE / "data" / "processed" / "speeches" / "models"
        news_dir = BASE / "data" / "processed" / "newspapers"
        tab_dir = BASE / "output" / "tables"
    else:
        run_dir = RUNS / run_name
        model_dir = run_dir / "models"
        news_dir = run_dir / "newspapers"
        tab_dir = run_dir / "output" / "tables"
    return {
        "config": run_dir / "config_override.json",
        "training": model_dir / "06_training_summary.csv",
        "phrases": model_dir / "06_top_partisan_phrases.csv",
        "norm": news_dir / "09_normalization_params.csv",
        "did": tab_dir / "did_results.csv",
        "es": tab_dir / "event_study_coefficients.csv",
    }


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def fmt_coef(row):
    """Format coefficient with SE and stars."""
    s = stars(row["pval"])
    return f"{row['coef']:.4f}{s} ({row['se']:.4f})"


def generate_report(run_name):
    """Generate markdown report for a single run."""
    paths = get_paths(run_name)
    lines = []

    # ── 1. Header & Settings ──────────────────────────────────────
    if run_name == "main":
        lines.append(f"# Model Report: Main (Baseline)")
        lines.append("")
        lines.append("This is the **baseline specification** following Widmer et al. (2025).")
    else:
        # Read config override
        try:
            with open(paths["config"]) as f:
                override = json.load(f)
        except FileNotFoundError:
            override = {}

        desc = override.get("description", run_name)
        lines.append(f"# Model Report: {run_name}")
        lines.append("")

    lines.append("")
    lines.append("## 1. Model Settings")
    lines.append("")

    if run_name == "main":
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        for k, v in MAIN_DEFAULTS.items():
            label = SETTING_LABELS.get(k, k)
            lines.append(f"| {label} | `{v}` |")
    else:
        # Show full settings with differences highlighted
        with open(paths["config"]) as f:
            override = json.load(f)

        # Remove non-setting keys
        skip_keys = {"run_name", "input_speech_dir", "input_news_dir"}
        diff_keys = {k for k in override if k not in skip_keys}

        lines.append("| Setting | Value | Main (baseline) | Changed? |")
        lines.append("|---------|-------|-----------------|----------|")
        for k, default_v in MAIN_DEFAULTS.items():
            label = SETTING_LABELS.get(k, k)
            actual_v = override.get(k, default_v)
            changed = "**YES**" if k in diff_keys else ""
            lines.append(f"| {label} | `{actual_v}` | `{default_v}` | {changed} |")

        lines.append("")
        lines.append("### Differences from main:")
        lines.append("")
        for k in diff_keys:
            label = SETTING_LABELS.get(k, k)
            lines.append(f"- **{label}**: `{MAIN_DEFAULTS.get(k, '-')}` -> `{override[k]}`")

    lines.append("")

    # ── 2. Training Diagnostics ───────────────────────────────────
    lines.append("## 2. Training Diagnostics")
    lines.append("")

    try:
        train = pd.read_csv(paths["training"])
        lines.append(f"| Congress | Window | Rep. speeches | Dem. speeches | Accuracy | Features (+) | Features (-) | Total | Best C |")
        lines.append(f"|----------|--------|---------------|---------------|----------|-------------|-------------|-------|--------|")
        for _, r in train.iterrows():
            w = r["window"]
            sc = int(r["scores_congress"])
            nr = int(r["n_rep"])
            nd = int(r["n_dem"])
            acc = f"{r['train_accuracy']:.3f}"
            np_ = int(r["n_positive_coefs"])
            nn = int(r["n_negative_coefs"])
            tot = int(r["n_nonzero_coefs"])
            c = f"{r['best_C']:.3f}"
            lines.append(f"| {sc} | {w} | {nr:,} | {nd:,} | {acc} | {np_} | {nn} | {tot} | {c} |")

        avg_acc = train["train_accuracy"].mean()
        lines.append("")
        lines.append(f"**Average accuracy**: {avg_acc:.3f}")
        lines.append(f"**Accuracy range**: {train['train_accuracy'].min():.3f} -{train['train_accuracy'].max():.3f}")
        lines.append(f"**Total features range**: {int(train['n_nonzero_coefs'].min())} -{int(train['n_nonzero_coefs'].max())}")
    except FileNotFoundError:
        lines.append("*Training summary not found.*")

    lines.append("")

    # ── 3. Top 10 Partisan Phrases ────────────────────────────────
    lines.append("## 3. Top 10 Partisan Phrases")
    lines.append("")

    try:
        phrases = pd.read_csv(paths["phrases"])
        # Show 3 representative congresses: 100, 104, 108
        for cong in [100, 104, 108]:
            sub = phrases[phrases["congress"] == cong]
            if sub.empty:
                continue

            lines.append(f"### Congress {cong}")
            lines.append("")

            # Top 10 Republican (positive)
            rep = sub[sub["coefficient"] > 0].nlargest(10, "coefficient")
            dem = sub[sub["coefficient"] < 0].nsmallest(10, "coefficient")

            lines.append("**Republican phrases (beta > 0):**")
            lines.append("")
            lines.append("| Rank | Phrase | Coefficient |")
            lines.append("|------|--------|-------------|")
            for i, (_, r) in enumerate(rep.iterrows(), 1):
                lines.append(f"| {i} | {r['phrase']} | {r['coefficient']:.3f} |")

            lines.append("")
            lines.append("**Democratic phrases (beta < 0):**")
            lines.append("")
            lines.append("| Rank | Phrase | Coefficient |")
            lines.append("|------|--------|-------------|")
            for i, (_, r) in enumerate(dem.iterrows(), 1):
                lines.append(f"| {i} | {r['phrase']} | {r['coefficient']:.3f} |")

            lines.append("")
    except FileNotFoundError:
        lines.append("*Phrases file not found.*")

    lines.append("")

    # ── 4. Normalization Parameters ───────────────────────────────
    lines.append("## 4. Normalization Parameters (Partisan Gap)")
    lines.append("")
    lines.append("The partisan gap Delta_t = mu_R - mu_D measures the distance between the average")
    lines.append("Republican and Democratic legislator's net slant in each congress.")
    lines.append("All newspaper scores are divided by Delta_t for cross-year comparability.")
    lines.append("")

    try:
        norm = pd.read_csv(paths["norm"])
        lines.append("| Congress | Window | mu_R | mu_D | Gap (Delta_t) | N(Rep) | N(Dem) |")
        lines.append("|----------|--------|-----|-----|-----------|--------|--------|")
        for _, r in norm.iterrows():
            lines.append(
                f"| {int(r['congress'])} | {r['window']} | "
                f"{r['mu_R']:.4f} | {r['mu_D']:.4f} | "
                f"**{r['gap']:.4f}** | {int(r['n_rep']):,} | {int(r['n_dem']):,} |"
            )
        lines.append("")
        lines.append(f"**Gap range**: {norm['gap'].min():.4f} -{norm['gap'].max():.4f}")
        lines.append(f"**Mean gap**: {norm['gap'].mean():.4f}")
    except FileNotFoundError:
        lines.append("*Normalization parameters not found.*")

    lines.append("")

    # ── 5. DiD Results ────────────────────────────────────────────
    lines.append("## 5. Difference-in-Differences Results")
    lines.append("")
    lines.append("Spec 1: Paper FE + Year FE + Division x Year FE")
    lines.append("Spec 2: + Manufacturing share x Year")
    lines.append("Spec 3: + China shock x Year")
    lines.append("")

    try:
        did = pd.read_csv(paths["did"])

        # Check if we have raw variables
        has_raw = "net_slant" in did["depvar"].values

        # ── Panel A: All Articles ──
        lines.append("### Panel A: All Articles")
        lines.append("")

        if has_raw:
            # Show raw AND normalized side by side
            lines.append("#### Raw (pre-normalization)")
            lines.append("")
            lines.append("| Outcome | Spec 1 | Spec 2 | Spec 3 |")
            lines.append("|---------|--------|--------|--------|")
            for dv, label in [
                ("net_slant", "Net Slant"),
                ("right_intensity", "Right Intensity"),
                ("left_intensity", "Left Intensity"),
                ("politicization", "Politicization"),
            ]:
                sub = did[did["depvar"] == dv]
                if sub.empty:
                    continue
                cols = []
                for spec in ["spec1", "spec2", "spec3"]:
                    row = sub[sub["spec"] == spec].iloc[0]
                    cols.append(fmt_coef(row))
                lines.append(f"| {label} | {cols[0]} | {cols[1]} | {cols[2]} |")
            lines.append("")

            lines.append("#### Normalized (post-normalization)")
            lines.append("")

        lines.append("| Outcome | Spec 1 | Spec 2 | Spec 3 |")
        lines.append("|---------|--------|--------|--------|")
        for dv, label in [
            ("net_slant_norm", "Net Slant"),
            ("right_norm", "Right Intensity"),
            ("left_norm", "Left Intensity"),
            ("politicization_norm", "Politicization"),
        ]:
            sub = did[did["depvar"] == dv]
            if sub.empty:
                continue
            cols = []
            for spec in ["spec1", "spec2", "spec3"]:
                row = sub[sub["spec"] == spec].iloc[0]
                cols.append(fmt_coef(row))
            lines.append(f"| {label} | {cols[0]} | {cols[1]} | {cols[2]} |")

        lines.append("")

        # ── Panel B: Economic Articles ──
        lines.append("### Panel B: Economic Articles")
        lines.append("")

        if has_raw:
            lines.append("#### Raw (pre-normalization)")
            lines.append("")
            lines.append("| Outcome | Spec 1 | Spec 2 | Spec 3 |")
            lines.append("|---------|--------|--------|--------|")
            for dv, label in [
                ("net_slant_econ", "Net Slant (Econ)"),
                ("right_intensity_econ", "Right Intensity (Econ)"),
                ("left_intensity_econ", "Left Intensity (Econ)"),
                ("politicization_econ", "Politicization (Econ)"),
            ]:
                sub = did[did["depvar"] == dv]
                if sub.empty:
                    continue
                cols = []
                for spec in ["spec1", "spec2", "spec3"]:
                    row = sub[sub["spec"] == spec].iloc[0]
                    cols.append(fmt_coef(row))
                lines.append(f"| {label} | {cols[0]} | {cols[1]} | {cols[2]} |")
            lines.append("")

            lines.append("#### Normalized (post-normalization)")
            lines.append("")

        lines.append("| Outcome | Spec 1 | Spec 2 | Spec 3 |")
        lines.append("|---------|--------|--------|--------|")
        for dv, label in [
            ("net_slant_norm_econ", "Net Slant (Econ)"),
            ("right_norm_econ", "Right Intensity (Econ)"),
            ("left_norm_econ", "Left Intensity (Econ)"),
            ("politicization_norm_econ", "Politicization (Econ)"),
            ("econ_share", "Econ Article Share"),
        ]:
            sub = did[did["depvar"] == dv]
            if sub.empty:
                continue
            cols = []
            for spec in ["spec1", "spec2", "spec3"]:
                row = sub[sub["spec"] == spec].iloc[0]
                cols.append(fmt_coef(row))
            lines.append(f"| {label} | {cols[0]} | {cols[1]} | {cols[2]} |")

        lines.append("")
        lines.append(f"N = {int(did['N'].iloc[0]):,} (all articles), "
                      f"{int(did[did['depvar'].str.contains('econ')]['N'].iloc[0]):,} (econ articles)")

    except FileNotFoundError:
        lines.append("*DiD results not found.*")

    lines.append("")

    # ── 6. Event Study Highlights ─────────────────────────────────
    lines.append("## 6. Event Study Highlights")
    lines.append("")

    try:
        es = pd.read_csv(paths["es"])

        # Focus on net_slant_norm and right_norm, controls spec
        for dv, label in [
            ("net_slant_norm", "Net Slant (Norm)"),
            ("right_norm", "Right Intensity (Norm)"),
            ("left_norm", "Left Intensity (Norm)"),
        ]:
            # Try "controls" spec first, fall back to "baseline"
            sub = es[(es["depvar"] == dv) & (es["spec"] == "controls")]
            if sub.empty:
                sub = es[(es["depvar"] == dv) & (es["spec"] == "baseline")]
            if sub.empty:
                continue

            spec_used = sub["spec"].iloc[0]
            lines.append(f"### {label} ({spec_used})")
            lines.append("")
            lines.append("| Year | Coefficient | SE | 95% CI | Sig |")
            lines.append("|------|------------|-----|--------|-----|")

            for _, r in sub.sort_values("year").iterrows():
                yr = int(r["year"])
                coef = r["coef"]
                se = r["se"]
                ci_lo = r["ci_lo"]
                ci_hi = r["ci_hi"]
                # Determine significance from CI (does it cross zero?)
                sig = ""
                if ci_lo > 0 or ci_hi < 0:
                    # Approximate p from coef/se
                    if se > 0:
                        from scipy.stats import norm as spnorm
                        p_approx = 2 * (1 - spnorm.cdf(abs(coef / se)))
                        sig = stars(p_approx)
                marker = " [base]" if yr == 1993 else ""
                lines.append(
                    f"| {yr}{marker} | {coef:.4f}{sig} | {se:.4f} | "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}] | {'Y' if sig else ''} |"
                )
            lines.append("")

            # Summarize pre/post
            pre = sub[sub["year"] < 1993]
            post = sub[sub["year"] > 1993]
            if not pre.empty and not post.empty:
                pre_max = pre["coef"].abs().max()
                post_max = post["coef"].abs().max()
                post_sig = post[(post["ci_lo"] > 0) | (post["ci_hi"] < 0)]
                lines.append(f"- Pre-period max |coef|: {pre_max:.4f}")
                lines.append(f"- Post-period max |coef|: {post_max:.4f}")
                lines.append(f"- Post-period significant years: {len(post_sig)} / {len(post)}")
                if not post_sig.empty:
                    sig_years = sorted(post_sig["year"].astype(int).tolist())
                    lines.append(f"- Significant years: {sig_years}")
                lines.append("")

    except (FileNotFoundError, ImportError):
        lines.append("*Event study data not found or scipy not available.*")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated automatically. Run: `{run_name}`*")

    return "\n".join(lines)


# ── Generate all reports ──────────────────────────────────────────
if __name__ == "__main__":
    # Get all experiment directories
    exp_names = sorted([d.name for d in RUNS.iterdir() if d.is_dir()])

    # Put main first
    if "main" in exp_names:
        exp_names.remove("main")
        exp_names.insert(0, "main")

    print(f"Generating reports for {len(exp_names)} runs...")

    for name in exp_names:
        print(f"  -> {name}...", end=" ")
        try:
            report = generate_report(name)
            out_path = REPORT_DIR / f"{name}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"OK -> {out_path.name}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Reports saved to {REPORT_DIR}")
