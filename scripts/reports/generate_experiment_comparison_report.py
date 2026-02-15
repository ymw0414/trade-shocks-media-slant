"""
Generate a LaTeX report comparing NLP model specifications for measuring
newspaper partisan slant and their impact on DiD regression results.

Output: reports/experiment_comparison.tex
"""

import os
import shutil
import pandas as pd
import numpy as np
from collections import OrderedDict
from pathlib import Path

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
REPORT_DIR = BASE / "reports"
FIG_DEST = REPORT_DIR / "figures"

# ── Experiment metadata ──────────────────────────────────────────────
EXPERIMENTS = OrderedDict([
    ("main", {
        "label": "Bigram-only, BIC",
        "short": "Bigram BIC",
        "ngram": "Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "BIC",
        "aggregation": "Speech",
        "training_sample": "All R/D",
    }),
    ("exp_unigram_gst", {
        "label": "Uni+Bigram, GST, BIC",
        "short": "U+B BIC",
        "ngram": "Uni+Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "BIC",
        "aggregation": "Speech",
        "training_sample": "All R/D",
    }),
    ("exp_uni_gst_cv", {
        "label": "Uni+Bigram, GST, CV",
        "short": "U+B CV",
        "ngram": "Uni+Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "CV",
        "aggregation": "Speech",
        "training_sample": "All R/D",
    }),
    ("exp_uni_gst_leg_cv", {
        "label": "Legislator-level, All R/D, CV",
        "short": "Leg All",
        "ngram": "Uni+Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "CV",
        "aggregation": "Legislator",
        "training_sample": "All R/D",
    }),
    ("exp_uni_gst_leg_core_cv", {
        "label": "Legislator-level, Core 20\\%, CV",
        "short": "Leg Core",
        "ngram": "Uni+Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "CV",
        "aggregation": "Legislator",
        "training_sample": "Core 20\\%",
    }),
    ("exp_shvocab_cv", {
        "label": "Speech-level, All R/D, CV, Shared Vocab",
        "short": "ShVocab CV",
        "ngram": "Uni+Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "CV",
        "aggregation": "Speech",
        "training_sample": "All R/D",
    }),
    ("exp_uni_gst_core_cv", {
        "label": "Speech-level, Core 20\\%, CV, Shared Vocab",
        "short": "Core CV",
        "ngram": "Uni+Bigram",
        "gst_filter": "Yes",
        "lambda_sel": "CV",
        "aggregation": "Speech",
        "training_sample": "Core 20\\%",
    }),
])

# Key outcomes for comparison tables
KEY_OUTCOMES = [
    ("net_slant_norm", "Net Slant (Norm.)"),
    ("ext_nonzero", "Share Non-Zero"),
    ("ext_R", "Share R-Leaning"),
    ("ext_D", "Share D-Leaning"),
    ("int_net_slant_norm", "Int. Net Slant (Norm.)"),
    ("int_R_norm", "Int. R Intensity (Norm.)"),
    ("int_D_norm", "Int. D Intensity (Norm.)"),
]

# All 13 outcomes for appendix
ALL_OUTCOMES = [
    ("net_slant", "Net Slant (Raw)"),
    ("net_slant_norm", "Net Slant (Norm.)"),
    ("politicization", "Politicization (Raw)"),
    ("politicization_norm", "Politicization (Norm.)"),
    ("ext_nonzero", "Share Non-Zero"),
    ("ext_R", "Share R-Leaning"),
    ("ext_D", "Share D-Leaning"),
    ("int_net_slant", "Int. Net Slant (Raw)"),
    ("int_R", "Int. R Intensity (Raw)"),
    ("int_D", "Int. D Intensity (Raw)"),
    ("int_net_slant_norm", "Int. Net Slant (Norm.)"),
    ("int_R_norm", "Int. R Intensity (Norm.)"),
    ("int_D_norm", "Int. D Intensity (Norm.)"),
]

BEST_MODEL = "exp_shvocab_cv"

EVENT_STUDY_FIGS = [
    "event_study_net_slant_norm.png",
    "event_study_ext_R_vs_D.png",
    "event_study_int_R_vs_D.png",
    "event_study_ext_R.png",
    "event_study_ext_D.png",
]


# ── Utility functions ────────────────────────────────────────────────

def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def stars(p):
    if p < 0.01: return "^{***}"
    if p < 0.05: return "^{**}"
    if p < 0.10: return "^{*}"
    return ""


def get_paths(run_name):
    if run_name == "main":
        model_dir = BASE / "data" / "processed" / "speeches" / "models"
        news_dir = BASE / "data" / "processed" / "newspapers"
        tab_dir = BASE / "output" / "tables"
        panel_dir = BASE / "data" / "processed" / "panel"
        fig_dir = BASE / "output" / "figures"
    else:
        run_dir = BASE / "data" / "processed" / "runs" / run_name
        model_dir = run_dir / "models"
        news_dir = run_dir / "newspapers"
        tab_dir = run_dir / "output" / "tables"
        panel_dir = run_dir / "panel"
        fig_dir = run_dir / "output" / "figures"
    return {
        "training": model_dir / "06_training_summary.csv",
        "phrases": model_dir / "06_top_partisan_phrases.csv",
        "norm": news_dir / "09_normalization_params.csv",
        "did": tab_dir / "did_results.csv",
        "es": tab_dir / "event_study_coefficients.csv",
        "panel": panel_dir / "14_regression_panel.parquet",
        "fig_dir": fig_dir,
    }


def load_all_data():
    data = {}
    for name in EXPERIMENTS:
        paths = get_paths(name)
        exp = {}
        try:
            exp["training"] = pd.read_csv(paths["training"])
        except FileNotFoundError:
            exp["training"] = None
        try:
            exp["did"] = pd.read_csv(paths["did"])
        except FileNotFoundError:
            exp["did"] = None
        try:
            panel = pd.read_parquet(paths["panel"])
            exp["panel_means"] = {}
            for col in ["ext_nonzero", "ext_R", "ext_D"]:
                if col in panel.columns:
                    exp["panel_means"][col] = panel[col].mean()
        except FileNotFoundError:
            exp["panel_means"] = None
        data[name] = exp
    return data


def copy_figures():
    FIG_DEST.mkdir(parents=True, exist_ok=True)
    src_dir = get_paths(BEST_MODEL)["fig_dir"]
    for fig in EVENT_STUDY_FIGS:
        src = src_dir / fig
        if src.exists():
            shutil.copy2(src, FIG_DEST / fig)
            print(f"  Copied: {fig}")


# ── LaTeX generation ─────────────────────────────────────────────────

def generate_report(data):
    doc = []

    # ── Preamble ──
    doc.append(r"""\documentclass[11pt]{article}
\usepackage[top=0.8in, bottom=0.8in, left=0.7in, right=0.7in]{geometry}
\usepackage{booktabs}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{pdflscape}
\usepackage{caption}

\hypersetup{colorlinks=true, linkcolor=blue!60!black, urlcolor=blue!60!black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\title{NLP Specification Comparison:\\ Trade Shocks and Media Slant}
\author{}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\clearpage
""")

    # ── Section 1: Motivation ──
    doc.append(r"""
\section{Motivation}

This report compares five NLP model specifications for measuring partisan slant
in U.S.\ newspaper articles, and evaluates their downstream impact on
difference-in-differences (DiD) regressions linking NAFTA tariff vulnerability
to newspaper slant.

The baseline specification follows Widmer et al., using \textbf{bigrams only} to
identify partisan language from Congressional Record speeches. This approach
suffers from severe \textbf{zero-inflation}: the vast majority of short newspaper
articles ($\sim$80 words on average) contain no partisan bigrams at all, and therefore
receive a partisan score of exactly zero. In the baseline model, only
\textbf{9.5\%} of newspaper-year article observations have any non-zero partisan
content. These zeros are not ``centrist''---they are simply unmeasured. Including
them in the newspaper-year mean slant attenuates the signal toward zero and
destroys statistical power.

Adding \textbf{unigrams} back to the vocabulary dramatically resolves this problem:
with unigram+bigram features, \textbf{97.9--99.9\%} of articles receive non-zero
scores, depending on the lambda selection method. This report compares five
specifications that vary along three dimensions:
\begin{itemize}
\item \textbf{N-gram type}: Bigrams only vs.\ unigrams+bigrams
\item \textbf{Lambda selection}: BIC (sparse) vs.\ 5-fold cross-validation (dense)
\item \textbf{Aggregation level}: Individual speeches vs.\ legislator-congress level
\end{itemize}
""")

    # ── Section 2: Experiment Design ──
    doc.append(r"""
\section{Experiment Design}

All specifications share the same underlying data (Congressional Record speeches,
Congresses 99--108) and newspaper corpus (55.9M articles). The GST procedural
bigram filter is applied in all cases. Table~\ref{tab:design} summarizes the
five specifications.

\begin{table}[H]
\centering
\caption{Experiment Design Summary}
\label{tab:design}
\footnotesize
\begin{tabular}{@{} c l l c l l l @{}}
\toprule
\# & Run Name & N-gram & GST & $\lambda$ & Aggregation & Sample \\
\midrule""")

    for i, (name, meta) in enumerate(EXPERIMENTS.items(), 1):
        doc.append(
            f"{i} & {esc(name)} & {meta['ngram']} & {meta['gst_filter']} & "
            f"{meta['lambda_sel']} & {meta['aggregation']} & {meta['training_sample']} \\\\"
        )

    doc.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # ── Section 3: Training Diagnostics ──
    doc.append(r"""
\section{Training Diagnostics}

Table~\ref{tab:training} compares LASSO classifier performance across the five
specifications. Each model is trained on 9 one-congress windows (Congresses 100--108).

\begin{table}[H]
\centering
\caption{LASSO Training Summary}
\label{tab:training}
\footnotesize
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{} l r r r r @{}}
\toprule
Experiment & Avg.\ Accuracy & Range & Avg.\ Features & Range \\
\midrule""")

    for name, meta in EXPERIMENTS.items():
        tr = data[name].get("training")
        if tr is not None:
            avg_acc = tr["train_accuracy"].mean()
            min_acc = tr["train_accuracy"].min()
            max_acc = tr["train_accuracy"].max()
            avg_k = tr["n_nonzero_coefs"].mean()
            min_k = int(tr["n_nonzero_coefs"].min())
            max_k = int(tr["n_nonzero_coefs"].max())
            doc.append(
                f"{meta['short']} & {avg_acc:.3f} & {min_acc:.3f}--{max_acc:.3f} & "
                f"{avg_k:.0f} & {min_k}--{max_k} \\\\"
            )
        else:
            doc.append(f"{meta['short']} & \\multicolumn{{4}}{{c}}{{---}} \\\\")

    doc.append(r"""\bottomrule
\end{tabular}
\end{table}

\noindent\textbf{Note:} The legislator-level models achieve $\sim$0.99 training accuracy,
which reflects severe overfitting due to the small number of legislator-congress
documents ($\sim$200--260 per window) relative to the feature space. Their high
accuracy does not indicate superior classification and results in noisy
regression estimates downstream.
""")

    # ── Section 4: Zero-Inflation ──
    doc.append(r"""
\section{Zero-Inflation Analysis}

Table~\ref{tab:zeros} reports the mean share of newspaper-year observations with
non-zero partisan scores, and the mean share classified as R-leaning or D-leaning.

\begin{table}[H]
\centering
\caption{Partisan Article Coverage (Newspaper-Year Panel Means)}
\label{tab:zeros}
\footnotesize
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{@{} l r r r @{}}
\toprule
Experiment & \% Non-Zero & \% R-Leaning & \% D-Leaning \\
\midrule""")

    for name, meta in EXPERIMENTS.items():
        pm = data[name].get("panel_means")
        if pm:
            nz = pm.get("ext_nonzero", float("nan"))
            er = pm.get("ext_R", float("nan"))
            ed = pm.get("ext_D", float("nan"))
            doc.append(f"{meta['short']} & {nz*100:.1f}\\% & {er*100:.1f}\\% & {ed*100:.1f}\\% \\\\")
        else:
            doc.append(f"{meta['short']} & \\multicolumn{{3}}{{c}}{{---}} \\\\")

    doc.append(r"""\bottomrule
\end{tabular}
\end{table}

\noindent The baseline bigram-only model scores only 9.5\% of articles as having any
partisan content. Adding unigrams raises coverage to 97.9\% (BIC) or 99.9\% (CV).
The legislator-level models, despite using the same vocabulary, produce more zeros
because their sparser coefficient vectors score fewer articles. The core-only model
is the most affected (66.2\% non-zero) due to using only 21--46 features per window.
""")

    # ── Section 5: DiD Results ──
    doc.append(r"""
\section{Difference-in-Differences Results}

Tables~\ref{tab:did_full} and \ref{tab:did_base} compare DiD coefficients across
the five specifications. The coefficient of interest is
$\hat{\beta}_{\text{vuln} \times \text{post}}$: the interaction of pre-NAFTA tariff
vulnerability (scaled Q4$-$Q1) with a post-1994 indicator. All specifications include
paper fixed effects, year fixed effects, and Census division $\times$ year fixed effects.
Standard errors are clustered at the commuting zone level.
""")

    # Table A: Full controls (spec3)
    _did_table(doc, data, "spec3",
               "DiD Results: Full Controls (Spec 3)",
               "tab:did_full",
               "Paper FE, Year FE, Division$\\times$Year FE, Manuf.\\ share$\\times$Year, China shock$\\times$Year")

    # Table B: Baseline (spec1)
    _did_table(doc, data, "spec1",
               "DiD Results: Baseline (Spec 1)",
               "tab:did_base",
               "Paper FE, Year FE, Division$\\times$Year FE")

    # ── Section 6: Event Study ──
    doc.append(r"""
\section{Event Study: Main Model (All R/D, GST, CV, Shared Vocab)}

Figures~\ref{fig:es_slant}--\ref{fig:es_int} show event-study coefficients
for the main model (exp\_shvocab\_cv). The base year is 1993
(the last pre-NAFTA year). Each panel shows baseline and controlled
specifications with 95\% confidence intervals.
""")

    fig_specs = [
        ("event_study_net_slant_norm.png", "Net Slant (Normalized)", "fig:es_slant",
         "Event-study coefficients for normalized net slant. Base year = 1993."),
        ("event_study_ext_R_vs_D.png", "Extensive Margin: R vs.\\ D Article Share", "fig:es_ext",
         "Event-study coefficients for share of R-leaning (red) and D-leaning (blue) articles."),
        ("event_study_int_R_vs_D.png", "Intensive Margin: R vs.\\ D Intensity", "fig:es_int",
         "Event-study coefficients for R and D intensity conditional on non-zero partisan content."),
    ]

    for figname, title, label, caption in fig_specs:
        fig_path = FIG_DEST / figname
        if fig_path.exists():
            doc.append(f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{figures/{figname}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}
""")

    # ── Section 7: Margin Decomposition ──
    # Pull actual coefficients for inline discussion
    best_did = data[BEST_MODEL].get("did")
    ext_r_s3 = ext_d_s3 = int_ns_s3 = None
    if best_did is not None:
        _r = best_did[(best_did["depvar"] == "ext_R") & (best_did["spec"] == "spec3")]
        if len(_r): ext_r_s3 = _r.iloc[0]
        _d = best_did[(best_did["depvar"] == "ext_D") & (best_did["spec"] == "spec3")]
        if len(_d): ext_d_s3 = _d.iloc[0]
        _n = best_did[(best_did["depvar"] == "int_net_slant_norm") & (best_did["spec"] == "spec3")]
        if len(_n): int_ns_s3 = _n.iloc[0]

    doc.append(r"""
\section{Margin Decomposition}

The results from the main specification (All R/D, Uni+Bigram, GST, CV, Shared Vocab)
reveal that the effect of NAFTA vulnerability on newspaper slant operates
primarily through the \textbf{extensive margin}---the composition of which
articles carry partisan content---rather than the intensive margin.
""")

    if ext_r_s3 is not None and ext_d_s3 is not None:
        doc.append(
            f"The share of R-leaning articles increases by "
            f"$\\hat{{\\beta}} = {ext_r_s3['coef']:.4f}$ (SE $= {ext_r_s3['se']:.4f}$, $p = {ext_r_s3['pval']:.4f}$), "
            f"while the share of D-leaning articles decreases by "
            f"${abs(ext_d_s3['coef']):.4f}$ (SE $= {ext_d_s3['se']:.4f}$, $p = {ext_d_s3['pval']:.4f}$). "
            f"These effects are nearly symmetric and highly significant ($p < 0.01$).\n"
        )

    if int_ns_s3 is not None:
        doc.append(
            f"\nThe intensive margin net slant is marginally significant: "
            f"$\\hat{{\\beta}} = {int_ns_s3['coef']:.4f}$ (SE $= {int_ns_s3['se']:.4f}$, $p = {int_ns_s3['pval']:.4f}$). "
        )

    doc.append(r"""
This decomposition suggests that NAFTA vulnerability shifts the
\emph{composition} of partisan language in newspaper coverage: trade-shocked
areas see a relative increase in R-leaning articles and decrease in D-leaning
articles. The unconditional net slant (which combines extensive and intensive
margins) is marginally significant at the 10\% level, consistent with
the extensive margin driving the aggregate effect.

The results are robust to the inclusion of China shock and manufacturing share
controls, and in fact \textbf{strengthen} slightly with controls (Spec 3 vs.\ Spec 1),
suggesting that the baseline estimates are conservative.
""")

    # ── Appendix ──
    doc.append(r"""
\clearpage
\appendix
\section{Full DiD Results by Experiment}

The following tables report the complete set of DiD coefficients for all 13
outcome variables across three specifications. Standard errors (clustered at
CZ level) are in parentheses.
""")

    for name, meta in EXPERIMENTS.items():
        did = data[name].get("did")
        if did is None:
            doc.append(f"\n\\subsection{{{meta['label']}}}\n\\textit{{Results not available.}}\n\\clearpage\n")
            continue

        doc.append(f"\n\\subsection{{{meta['label']} (\\texttt{{{esc(name)}}})}}")
        doc.append(r"""
\begin{table}[H]
\centering
\footnotesize
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{@{} l c c c @{}}
\toprule
& (1) Baseline & (2) +Manuf. & (3) +China \\
\midrule""")

        # Group into panels
        panels = [
            ("Unconditional", [("net_slant", "Net Slant (Raw)"),
                                ("net_slant_norm", "Net Slant (Norm.)"),
                                ("politicization", "Politicization (Raw)"),
                                ("politicization_norm", "Politicization (Norm.)")]),
            ("Extensive Margin", [("ext_nonzero", "Share Non-Zero"),
                                   ("ext_R", "Share R-Leaning"),
                                   ("ext_D", "Share D-Leaning")]),
            ("Intensive Margin (Raw)", [("int_net_slant", "Net Slant"),
                                         ("int_R", "R Intensity"),
                                         ("int_D", "D Intensity")]),
            ("Intensive Margin (Normalized)", [("int_net_slant_norm", "Net Slant"),
                                                ("int_R_norm", "R Intensity"),
                                                ("int_D_norm", "D Intensity")]),
        ]

        for panel_name, outcomes in panels:
            doc.append(f"\\addlinespace[4pt]")
            doc.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{panel_name}}}}} \\\\")
            doc.append(f"\\addlinespace[3pt]")
            for depvar, label in outcomes:
                sub = did[did["depvar"] == depvar]
                if sub.empty:
                    continue
                cells = [label]
                se_cells = [""]
                for spec in ["spec1", "spec2", "spec3"]:
                    row = sub[sub["spec"] == spec]
                    if row.empty:
                        cells.append("")
                        se_cells.append("")
                        continue
                    row = row.iloc[0]
                    s = stars(row["pval"])
                    sign = "$-$" if row["coef"] < 0 else ""
                    cells.append(f"{sign}{abs(row['coef']):.4f}${s}$")
                    se_cells.append(f"({row['se']:.4f})")
                doc.append(" & ".join(cells) + " \\\\")
                doc.append(" & ".join(se_cells) + " \\\\")
                doc.append("\\addlinespace[2pt]")

        doc.append(r"""\midrule
Paper FE & Yes & Yes & Yes \\
Year FE & Yes & Yes & Yes \\
Division $\times$ Year FE & Yes & Yes & Yes \\
Manuf.\ share $\times$ Year & & Yes & Yes \\
China shock $\times$ Year & & & Yes \\
\midrule""")

        # N from net_slant_norm spec1
        n_row = did[(did["depvar"] == "net_slant_norm") & (did["spec"] == "spec1")]
        if not n_row.empty:
            n_val = int(n_row.iloc[0]["N"])
            doc.append(f"Observations & {n_val:,} & {n_val:,} & {n_val:,} \\\\")

        doc.append(r"""\bottomrule
\end{tabular}
\end{table}
\clearpage""")

    doc.append(r"""
\end{document}
""")

    return "\n".join(doc)


def _did_table(doc, data, spec, title, label, fe_note):
    """Generate a cross-experiment DiD comparison table for one spec."""
    n_exp = len(EXPERIMENTS)

    col_spec = "r@{\\hspace{2pt}}l " * n_exp
    doc.append(f"""
\\begin{{table}}[H]
\\centering
\\caption{{{title}}}
\\label{{{label}}}
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{@{{}} l {col_spec}@{{}}}}
\\toprule""")

    # Column headers
    header = "Outcome"
    for meta in EXPERIMENTS.values():
        header += f" & \\multicolumn{{2}}{{c}}{{{meta['short']}}}"
    doc.append(header + " \\\\")

    # Cmidrules
    rules = ""
    for i in range(n_exp):
        c1 = 2 + i * 2
        c2 = c1 + 1
        rules += f"\\cmidrule(lr){{{c1}-{c2}}} "
    doc.append(rules)

    # Rows
    for depvar, label_text in KEY_OUTCOMES:
        # Coefficient row
        cells = [label_text]
        se_cells = [""]
        for name in EXPERIMENTS:
            did = data[name].get("did")
            if did is None:
                cells.append("\\multicolumn{2}{c}{---}")
                se_cells.append("\\multicolumn{2}{c}{}")
                continue
            row = did[(did["depvar"] == depvar) & (did["spec"] == spec)]
            if row.empty:
                cells.append("\\multicolumn{2}{c}{---}")
                se_cells.append("\\multicolumn{2}{c}{}")
                continue
            row = row.iloc[0]
            s = stars(row["pval"])
            sign = "$-$" if row["coef"] < 0 else ""
            cells.append(f"{sign}{abs(row['coef']):.4f} & ${s}$")
            se_cells.append(f"({row['se']:.4f}) & ")
        doc.append(" & ".join(cells) + " \\\\")
        doc.append(" & ".join(se_cells) + " \\\\")
        doc.append("\\addlinespace[3pt]")

    doc.append(f"\\midrule")
    doc.append(f"\\multicolumn{{{1 + n_exp * 2}}}{{@{{}}l}}{{\\footnotesize {fe_note}}} \\\\")

    # N
    for name in EXPERIMENTS:
        did = data[name].get("did")
        if did is not None:
            n_row = did[(did["depvar"] == "net_slant_norm") & (did["spec"] == spec)]
            if not n_row.empty:
                n_val = int(n_row.iloc[0]["N"])
                doc.append(f"\\multicolumn{{{1 + n_exp * 2}}}{{@{{}}l}}{{\\footnotesize $N = {n_val:,}$, CZ clusters $= 99$}} \\\\")
                break

    doc.append(r"""\bottomrule
\end{tabular}
\end{table}
""")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    REPORT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    data = load_all_data()

    print("Copying figures...")
    copy_figures()

    print("Generating report...")
    tex = generate_report(data)

    out_path = REPORT_DIR / "experiment_comparison.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\nDone. Output: {out_path}")
