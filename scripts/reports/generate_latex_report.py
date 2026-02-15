"""
Generate a single merged LaTeX report for all experiment runs.
Output: report/all_experiments.tex
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

# ── Main run defaults
MAIN_DEFAULTS = {
    "window_size": 1,
    "partisan_core_only": False,
    "lasso_lambda_selection": "bic",
    "bigrams_only": True,
    "aggregate_to_legislator": False,
    "use_relative_freq": True,
    "shared_vocab_min_df": None,
    "sign_consistency_filter": False,
    "norm_method": "raw_gap",
}

SETTING_LABELS = {
    "window_size": "Rolling window size",
    "partisan_core_only": "Partisan core only",
    "lasso_lambda_selection": "Lambda selection",
    "bigrams_only": "Bigrams only",
    "aggregate_to_legislator": "Aggregate to legislator",
    "use_relative_freq": "Relative frequency",
    "shared_vocab_min_df": "Shared vocab min DF",
    "sign_consistency_filter": "Sign consistency filter",
    "norm_method": "Normalization method",
}

EXP_DESCRIPTIONS = {
    "main": "Baseline (Widmer et al.)",
    "exp_win2": "2-congress rolling windows",
    "exp_win3": "3-congress rolling windows",
    "exp_core": "Partisan core only (top/bottom 20\\% Nokken-Poole)",
    "exp_cv": "5-fold CV for lambda selection (instead of BIC)",
    "exp_core_win2": "Partisan core + 2-congress windows",
    "exp_core_win3": "Partisan core + 3-congress windows",
    "exp_cv_win2": "CV lambda + 2-congress windows",
    "exp_cv_win3": "CV lambda + 3-congress windows",
    "exp_core_cv_win3": "Core + CV + 3-congress windows (all non-Widmer)",
    "exp_win2_sign": "2-congress windows + sign consistency filter",
    "exp_win2_shared": "2-congress windows + shared vocab (0.01\\%)",
    "exp_win2_sign_shared": "2-congress windows + sign filter + shared vocab",
    "exp_clean_vocab": "Shared vocab + sign consistency filter",
    "exp_proba": "Shared vocab + sign filter + prob\\_direct normalization",
}


def get_paths(run_name):
    if run_name == "main":
        model_dir = BASE / "data" / "processed" / "speeches" / "models"
        news_dir = BASE / "data" / "processed" / "newspapers"
        tab_dir = BASE / "output" / "tables"
        run_dir = RUNS / "main"
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


def esc(s):
    """Escape LaTeX special chars in data strings."""
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def stars(p):
    if p < 0.01: return "^{***}"
    if p < 0.05: return "^{**}"
    if p < 0.10: return "^{*}"
    return ""


def fmt_coef_tex(row):
    """Format coefficient with stars and SE on next line."""
    s = stars(row["pval"])
    sign = "$-$" if row["coef"] < 0 else ""
    return f"{sign}{abs(row['coef']):.4f}${ s }$ & ({row['se']:.4f})"


def fmt_coef_inline(row):
    """Single-cell: coef(se) with stars."""
    s = stars(row["pval"])
    return f"${row['coef']:.4f}${s}$"


def generate_section(run_name, is_main=False):
    """Generate LaTeX section for one run."""
    paths = get_paths(run_name)
    lines = []
    desc = EXP_DESCRIPTIONS.get(run_name, run_name)

    if is_main:
        lines.append(f"\\section{{{desc}}}")
        lines.append(f"\\label{{sec:{run_name}}}")
    else:
        lines.append(f"\\section{{{esc(run_name)}: {desc}}}")
        lines.append(f"\\label{{sec:{run_name}}}")

    lines.append("")

    # ── 1. Settings ───────────────────────────────────────────────
    lines.append("\\subsection{Model Settings}")
    lines.append("")

    if is_main:
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\begin{tabular}{@{} l l @{}}")
        lines.append("\\toprule")
        lines.append("Setting & Value \\\\")
        lines.append("\\midrule")
        for k, v in MAIN_DEFAULTS.items():
            label = SETTING_LABELS.get(k, k)
            lines.append(f"{esc(label)} & \\texttt{{{esc(str(v))}}} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    else:
        try:
            with open(paths["config"]) as f:
                override = json.load(f)
        except FileNotFoundError:
            override = {}

        skip_keys = {"run_name", "input_speech_dir", "input_news_dir"}
        diff_keys = {k for k in override if k not in skip_keys}

        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\begin{tabular}{@{} l l l c @{}}")
        lines.append("\\toprule")
        lines.append("Setting & Value & Main & Changed \\\\")
        lines.append("\\midrule")
        for k, default_v in MAIN_DEFAULTS.items():
            label = SETTING_LABELS.get(k, k)
            actual_v = override.get(k, default_v)
            chg = "$\\checkmark$" if k in diff_keys else ""
            lines.append(f"{esc(label)} & \\texttt{{{esc(str(actual_v))}}} & \\texttt{{{esc(str(default_v))}}} & {chg} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        if diff_keys:
            lines.append("")
            lines.append("\\noindent\\textbf{Changes from baseline:} ")
            changes = []
            for k in sorted(diff_keys):
                label = SETTING_LABELS.get(k, k)
                changes.append(f"{label}: \\texttt{{{esc(str(MAIN_DEFAULTS.get(k, '---')))}}} $\\to$ \\texttt{{{esc(str(override[k]))}}}")
            lines.append("; ".join(changes) + ".")

    lines.append("")

    # ── 2. Training Diagnostics ───────────────────────────────────
    lines.append("\\subsection{Training Diagnostics}")
    lines.append("")

    try:
        train = pd.read_csv(paths["training"])
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\renewcommand{\\arraystretch}{1.1}")
        lines.append("\\begin{tabular}{@{} c l r r c r r r c @{}}")
        lines.append("\\toprule")
        lines.append("Congress & Window & Rep. & Dem. & Accuracy & $\\hat{\\beta}>0$ & $\\hat{\\beta}<0$ & Total & Best $C$ \\\\")
        lines.append("\\midrule")
        for _, r in train.iterrows():
            w = str(r["window"]).replace("+", "$+$")
            sc = int(r["scores_congress"])
            nr = f"{int(r['n_rep']):,}"
            nd = f"{int(r['n_dem']):,}"
            acc = f"{r['train_accuracy']:.3f}"
            np_ = int(r["n_positive_coefs"])
            nn = int(r["n_negative_coefs"])
            tot = int(r["n_nonzero_coefs"])
            c_val = f"{r['best_C']:.3f}"
            lines.append(f"{sc} & {w} & {nr} & {nd} & {acc} & {np_} & {nn} & {tot} & {c_val} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        avg_acc = train["train_accuracy"].mean()
        lines.append("")
        lines.append(f"Average accuracy: {avg_acc:.3f} "
                      f"(range {train['train_accuracy'].min():.3f}--{train['train_accuracy'].max():.3f}). "
                      f"Total features: {int(train['n_nonzero_coefs'].min())}--{int(train['n_nonzero_coefs'].max())}.")
    except FileNotFoundError:
        lines.append("\\textit{Training summary not available.}")

    lines.append("")

    # ── 3. Top 10 Phrases ─────────────────────────────────────────
    lines.append("\\subsection{Top 10 Partisan Phrases}")
    lines.append("")

    try:
        phrases = pd.read_csv(paths["phrases"])

        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\renewcommand{\\arraystretch}{1.05}")
        lines.append("\\begin{tabular}{@{} c l r l r l r @{}}")
        lines.append("\\toprule")
        lines.append("& \\multicolumn{2}{c}{Congress 100} & \\multicolumn{2}{c}{Congress 104} & \\multicolumn{2}{c}{Congress 108} \\\\")
        lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
        lines.append("\\addlinespace[4pt]")

        # Build data for 3 congresses
        data = {}
        for cong in [100, 104, 108]:
            sub = phrases[phrases["congress"] == cong]
            rep = sub[sub["coefficient"] > 0].nlargest(10, "coefficient")
            dem = sub[sub["coefficient"] < 0].nsmallest(10, "coefficient")
            data[cong] = {"rep": rep, "dem": dem}

        # Panel A: Republican
        lines.append("\\multicolumn{7}{@{}l}{\\emph{Panel A: Republican phrases ($\\hat{\\beta} > 0$)}} \\\\")
        lines.append("\\addlinespace[3pt]")
        for i in range(10):
            cells = [str(i+1)]
            for cong in [100, 104, 108]:
                rep = data[cong]["rep"]
                if i < len(rep):
                    row = rep.iloc[i]
                    cells.append(f"{esc(row['phrase'])} & {row['coefficient']:.1f}")
                else:
                    cells.append("& ")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\addlinespace[6pt]")

        # Panel B: Democratic
        lines.append("\\multicolumn{7}{@{}l}{\\emph{Panel B: Democratic phrases ($\\hat{\\beta} < 0$)}} \\\\")
        lines.append("\\addlinespace[3pt]")
        for i in range(10):
            cells = [str(i+1)]
            for cong in [100, 104, 108]:
                dem = data[cong]["dem"]
                if i < len(dem):
                    row = dem.iloc[i]
                    cells.append(f"{esc(row['phrase'])} & $-${abs(row['coefficient']):.1f}")
                else:
                    cells.append("& ")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    except FileNotFoundError:
        lines.append("\\textit{Phrases not available.}")

    lines.append("")

    # ── 4. Normalization Parameters ───────────────────────────────
    lines.append("\\subsection{Normalization Parameters}")
    lines.append("")

    try:
        norm = pd.read_csv(paths["norm"])
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\renewcommand{\\arraystretch}{1.1}")
        lines.append("\\begin{tabular}{@{} c l r r c r r @{}}")
        lines.append("\\toprule")
        lines.append("Congress & Window & $\\mu_R$ & $\\mu_D$ & $\\Delta_t$ & $N_{\\text{Rep}}$ & $N_{\\text{Dem}}$ \\\\")
        lines.append("\\midrule")
        for _, r in norm.iterrows():
            w = str(r["window"]).replace("+", "$+$")
            lines.append(
                f"{int(r['congress'])} & {w} & "
                f"{r['mu_R']:.4f} & {r['mu_D']:.4f} & "
                f"{r['gap']:.4f} & {int(r['n_rep']):,} & {int(r['n_dem']):,} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
        lines.append(f"Partisan gap range: {norm['gap'].min():.4f}--{norm['gap'].max():.4f} "
                      f"(mean {norm['gap'].mean():.4f}).")
    except FileNotFoundError:
        lines.append("\\textit{Normalization parameters not available.}")

    lines.append("")

    # ── 5. DiD Results ────────────────────────────────────────────
    lines.append("\\subsection{Difference-in-Differences Results}")
    lines.append("")

    try:
        did = pd.read_csv(paths["did"])
        has_raw = "net_slant" in did["depvar"].values

        # Build a combined table: Panel A (all) + Panel B (econ)
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\renewcommand{\\arraystretch}{1.05}")
        lines.append("\\begin{tabular}{@{} l c c c @{}}")
        lines.append("\\toprule")
        lines.append("& (1) & (2) & (3) \\\\")
        lines.append("\\midrule")

        if has_raw:
            # Panel A raw
            lines.append("\\addlinespace[4pt]")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{Panel A: All articles (raw)}} \\\\")
            lines.append("\\addlinespace[3pt]")
            for dv, label in [
                ("net_slant", "Net slant"),
                ("right_intensity", "Right intensity"),
                ("left_intensity", "Left intensity"),
                ("politicization", "Politicization"),
            ]:
                sub = did[did["depvar"] == dv]
                if sub.empty:
                    continue
                cells = [label]
                se_cells = [""]
                for spec in ["spec1", "spec2", "spec3"]:
                    row = sub[sub["spec"] == spec].iloc[0]
                    s = stars(row["pval"])
                    sign = "$-$" if row["coef"] < 0 else ""
                    cells.append(f"{sign}{abs(row['coef']):.3f}${s}$")
                    se_cells.append(f"({row['se']:.3f})")
                lines.append(" & ".join(cells) + " \\\\")
                lines.append(" & ".join(se_cells) + " \\\\")
                lines.append("\\addlinespace[2pt]")

            # Panel A normalized
            lines.append("\\addlinespace[4pt]")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{Panel B: All articles (normalized)}} \\\\")
            lines.append("\\addlinespace[3pt]")
        else:
            lines.append("\\addlinespace[4pt]")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{Panel A: All articles}} \\\\")
            lines.append("\\addlinespace[3pt]")

        for dv, label in [
            ("net_slant_norm", "Net slant ($\\tilde{S}$)"),
            ("right_norm", "Right intensity ($\\tilde{R}$)"),
            ("left_norm", "Left intensity ($\\tilde{L}$)"),
            ("politicization_norm", "Politicization ($\\tilde{P}$)"),
        ]:
            sub = did[did["depvar"] == dv]
            if sub.empty:
                continue
            cells = [label]
            se_cells = [""]
            for spec in ["spec1", "spec2", "spec3"]:
                row = sub[sub["spec"] == spec].iloc[0]
                s = stars(row["pval"])
                sign = "$-$" if row["coef"] < 0 else ""
                cells.append(f"{sign}{abs(row['coef']):.3f}${s}$")
                se_cells.append(f"({row['se']:.3f})")
            lines.append(" & ".join(cells) + " \\\\")
            lines.append(" & ".join(se_cells) + " \\\\")
            lines.append("\\addlinespace[2pt]")

        # Economic articles
        if has_raw:
            # Raw econ
            lines.append("\\addlinespace[4pt]")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{Panel C: Economic articles (raw)}} \\\\")
            lines.append("\\addlinespace[3pt]")
            for dv, label in [
                ("right_intensity_econ", "Right intensity (econ)"),
                ("politicization_econ", "Politicization (econ)"),
                ("econ_share", "Econ article share"),
            ]:
                sub = did[did["depvar"] == dv]
                if sub.empty:
                    continue
                cells = [label]
                se_cells = [""]
                for spec in ["spec1", "spec2", "spec3"]:
                    row = sub[sub["spec"] == spec].iloc[0]
                    s = stars(row["pval"])
                    sign = "$-$" if row["coef"] < 0 else ""
                    cells.append(f"{sign}{abs(row['coef']):.3f}${s}$")
                    se_cells.append(f"({row['se']:.3f})")
                lines.append(" & ".join(cells) + " \\\\")
                lines.append(" & ".join(se_cells) + " \\\\")
                lines.append("\\addlinespace[2pt]")

            # Normalized econ
            lines.append("\\addlinespace[4pt]")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{Panel D: Economic articles (normalized)}} \\\\")
            lines.append("\\addlinespace[3pt]")
        else:
            lines.append("\\addlinespace[4pt]")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{Panel B: Economic articles}} \\\\")
            lines.append("\\addlinespace[3pt]")

        for dv, label in [
            ("right_norm_econ", "Right intensity ($\\tilde{R}^{\\text{econ}}$)"),
            ("politicization_norm_econ", "Politicization ($\\tilde{P}^{\\text{econ}}$)"),
            ("econ_share", "Econ article share"),
        ]:
            sub = did[did["depvar"] == dv]
            if sub.empty:
                continue
            # Skip econ_share if already printed in raw panel
            if has_raw and dv == "econ_share":
                continue
            cells = [label]
            se_cells = [""]
            for spec in ["spec1", "spec2", "spec3"]:
                row = sub[sub["spec"] == spec].iloc[0]
                s = stars(row["pval"])
                sign = "$-$" if row["coef"] < 0 else ""
                cells.append(f"{sign}{abs(row['coef']):.3f}${s}$")
                se_cells.append(f"({row['se']:.3f})")
            lines.append(" & ".join(cells) + " \\\\")
            lines.append(" & ".join(se_cells) + " \\\\")
            lines.append("\\addlinespace[2pt]")

        # If not has_raw, also print econ_share in Panel B
        if not has_raw:
            dv = "econ_share"
            sub = did[did["depvar"] == dv]
            if not sub.empty:
                cells = ["Econ article share"]
                se_cells = [""]
                for spec in ["spec1", "spec2", "spec3"]:
                    row = sub[sub["spec"] == spec].iloc[0]
                    s = stars(row["pval"])
                    sign = "$-$" if row["coef"] < 0 else ""
                    cells.append(f"{sign}{abs(row['coef']):.3f}${s}$")
                    se_cells.append(f"({row['se']:.3f})")
                lines.append(" & ".join(cells) + " \\\\")
                lines.append(" & ".join(se_cells) + " \\\\")
                lines.append("\\addlinespace[2pt]")

        lines.append("\\midrule")
        lines.append("Paper FE & Yes & Yes & Yes \\\\")
        lines.append("Year FE & Yes & Yes & Yes \\\\")
        lines.append("Division $\\times$ Year FE & Yes & Yes & Yes \\\\")
        lines.append("Manuf.\\ share $\\times$ Year & & Yes & Yes \\\\")
        lines.append("China shock $\\times$ Year & & & Yes \\\\")
        lines.append("\\midrule")

        n_all = int(did[did["depvar"].str.contains("net_slant_norm") & ~did["depvar"].str.contains("econ")]["N"].iloc[0])
        lines.append(f"Observations & {n_all:,} & {n_all:,} & {n_all:,} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

    except FileNotFoundError:
        lines.append("\\textit{DiD results not available.}")

    lines.append("")

    # ── 6. Event Study Summary ────────────────────────────────────
    lines.append("\\subsection{Event Study Summary}")
    lines.append("")

    try:
        es = pd.read_csv(paths["es"])

        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append("\\footnotesize")
        lines.append("\\renewcommand{\\arraystretch}{1.05}")
        lines.append("\\begin{tabular}{@{} c r@{\\hspace{4pt}}c r@{\\hspace{4pt}}c r@{\\hspace{4pt}}c @{}}")
        lines.append("\\toprule")
        lines.append("Year & \\multicolumn{2}{c}{Net Slant} & \\multicolumn{2}{c}{Right Int.} & \\multicolumn{2}{c}{Left Int.} \\\\")
        lines.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")

        depvars = [("net_slant_norm", "controls"), ("right_norm", "controls"), ("left_norm", "controls")]
        # fallback to baseline if controls not present
        for i, (dv, sp) in enumerate(depvars):
            sub = es[(es["depvar"] == dv) & (es["spec"] == sp)]
            if sub.empty:
                depvars[i] = (dv, "baseline")

        years = sorted(es["year"].unique())
        for yr in years:
            yr = int(yr)
            cells = [str(yr)]
            for dv, sp in depvars:
                sub = es[(es["depvar"] == dv) & (es["spec"] == sp) & (es["year"] == yr)]
                if sub.empty:
                    cells.append("&")
                    continue
                row = sub.iloc[0]
                coef = row["coef"]
                se = row["se"]
                ci_lo = row["ci_lo"]
                ci_hi = row["ci_hi"]
                sig = ""
                if se > 0 and (ci_lo > 0 or ci_hi < 0):
                    from scipy.stats import norm as spnorm
                    p_approx = 2 * (1 - spnorm.cdf(abs(coef / se)))
                    sig = stars(p_approx)
                sign = "$-$" if coef < 0 else ""
                cells.append(f"{sign}{abs(coef):.4f}${sig}$ & ({se:.4f})")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        # Post-period summary
        lines.append("")
        for dv, sp in depvars:
            sub = es[(es["depvar"] == dv) & (es["spec"] == sp)]
            post = sub[sub["year"] > 1993]
            post_sig = post[(post["ci_lo"] > 0) | (post["ci_hi"] < 0)]
            dv_label = dv.replace("_norm", "").replace("_", " ").title()
            if not post_sig.empty:
                sig_yrs = sorted(post_sig["year"].astype(int).tolist())
                lines.append(f"{dv_label}: {len(post_sig)}/{len(post)} post-period years significant ({sig_yrs}).")
                lines.append("")
            else:
                lines.append(f"{dv_label}: 0/{len(post)} post-period years significant.")
                lines.append("")

    except (FileNotFoundError, Exception) as e:
        lines.append(f"\\textit{{Event study data not available.}}")

    lines.append("")

    # ── 7. Event Study Figures ─────────────────────────────────────
    fig_dir = BASE / "reports" / "figures"
    panel_fig = fig_dir / f"{run_name}_4panel.png"
    if panel_fig.exists():
        lines.append("\\subsection{Event Study Figures}")
        lines.append("")
        lines.append("\\begin{figure}[H]")
        lines.append("\\centering")
        lines.append(f"\\includegraphics[width=\\textwidth]{{figures/{run_name}_4panel.png}}")
        lines.append(f"\\caption{{{desc}: Event study coefficients for NAFTA vulnerability (controls specification). "
                     f"Base year = {1993}. Panels show net slant (raw and normalized), right intensity, and left intensity.}}")
        lines.append(f"\\label{{fig:{run_name}_es}}")
        lines.append("\\end{figure}")
    else:
        lines.append("\\textit{Event study figures not available.}")

    lines.append("")
    lines.append("\\clearpage")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    exp_names = sorted([d.name for d in RUNS.iterdir() if d.is_dir()])
    if "main" in exp_names:
        exp_names.remove("main")
        exp_names.insert(0, "main")

    doc = []
    doc.append(r"""\documentclass[11pt]{article}
\usepackage[top=0.8in, bottom=0.8in, left=0.7in, right=0.7in]{geometry}
\usepackage{booktabs}
\usepackage{float}
\usepackage{threeparttable}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}

\hypersetup{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\title{Experiment Reports:\\Trade Shocks and Media Slant}
\author{Auto-generated}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\clearpage
""")

    print(f"Generating LaTeX report for {len(exp_names)} runs...")

    for name in exp_names:
        print(f"  -> {name}...", end=" ")
        try:
            section = generate_section(name, is_main=(name == "main"))
            doc.append(section)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    doc.append(r"\end{document}")

    out_path = REPORT_DIR / "all_experiments.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    print(f"\nDone. Output: {out_path}")
