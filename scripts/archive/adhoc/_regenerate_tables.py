"""
Regenerate descriptive tables and figures after NLP pipeline update.

Outputs:
  - draft/tab_model_diagnostics.tex
  - draft/tab_partisan_phrases.tex
  - draft/tab_variance_decomp.tex
  - output/figures/newspaper_slant_ranking.pdf  (.png)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

BASE = os.environ.get("SHIFTING_SLANT_DIR", "C:/Users/ymw04/Dropbox/shifting_slant")
sys.path.insert(0, os.path.join(BASE, "scripts", "nlp"))
import pipeline_config as cfg

DRAFT = os.path.join(BASE, "draft")
FIGURES = os.path.join(BASE, "output", "figures")
MODELS = str(cfg.MODEL_DIR)
NEWS   = str(cfg.NEWS_DIR)

os.makedirs(DRAFT, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── 1. Model Diagnostics Table ─────────────────────────────────────────────

print("=== 1. Model Diagnostics Table ===")

ts = pd.read_csv(os.path.join(MODELS, "06_training_summary.csv"))
np_params = pd.read_csv(os.path.join(NEWS, "09_normalization_params.csv"))

# Merge gap onto training summary by window
ts = ts.merge(np_params[["window", "gap"]], on="window", how="left")

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"    \centering")
lines.append(r"    \begin{threeparttable}")
lines.append(r"    \caption{LASSO Model Diagnostics by Rolling Window}")
lines.append(r"    \label{tab:diagnostics}")
lines.append(r"    \small")
lines.append(r"    \renewcommand{\arraystretch}{1.15}")
lines.append(r"    \begin{tabular}{@{} l c c c c c c c @{}}")
lines.append(r"        \toprule")
lines.append(r"        Window & Congress & \multicolumn{2}{c}{Partisan core} & CV accuracy & \multicolumn{2}{c}{Selected phrases} & Partisan gap \\")
lines.append(r"        \cmidrule(lr){3-4} \cmidrule(lr){6-7}")
lines.append(r"        & scored & Rep. & Dem. & & $\hat{\beta}>0$ & $\hat{\beta}<0$ & $\Delta_t$ \\")
lines.append(r"        \midrule")

for _, r in ts.iterrows():
    window = r["window"]
    cong   = int(r["scores_congress"])
    nrep   = int(r["n_rep_core"])
    ndem   = int(r["n_dem_core"])
    cvacc  = f"{r['cv_accuracy']:.3f}"
    npos   = int(r["n_positive_coefs"])
    nneg   = int(r["n_negative_coefs"])
    gap    = f"{r['gap']:.2f}"
    lines.append(f"        {window:<8s} & {cong} & {nrep} & {ndem} & {cvacc} & {npos} & {nneg} & {gap} \\\\")

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"    \vspace{6pt}")
lines.append(r"    \begin{tablenotes}[flushleft]")
lines.append(r"        \setlength{\labelsep}{0pt}")
lines.append(r"        \scriptsize")
lines.append(r"        \item \textit{Notes:} Each row corresponds to a two-Congress rolling window. ``Partisan core'' reports the number of legislators in the top (Republican) and bottom (Democratic) 20th percentile of Nokken--Poole scores. CV accuracy is the five-fold cross-validation score. Selected phrases are the number of non-zero LASSO coefficients. The partisan gap $\Delta_t = \mu_{R,t} - \mu_{D,t}$ is the difference in mean net slant between all Republican and all Democratic legislators (not just the core), used for normalization.")
lines.append(r"    \end{tablenotes}")
lines.append(r"    \end{threeparttable}")
lines.append(r"\end{table}")
lines.append("")

tex = "\n".join(lines)
out_path = os.path.join(DRAFT, "tab_model_diagnostics.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"  Written: {out_path}")

# ── 2. Top Partisan Phrases Table ──────────────────────────────────────────

print("=== 2. Top Partisan Phrases Table ===")

sys.path.insert(0, os.path.join(BASE, "scripts", "utils"))
import joblib

vectorizer = joblib.load(str(cfg.SPEECH_DIR / "05_feature_vectorizer.joblib"))
intersection_cols = np.load(os.path.join(MODELS, "06_intersection_cols.npy"))
feature_names = vectorizer.get_feature_names_out()[intersection_cols]

# Three representative windows
windows = [("99", "100"), ("103", "104"), ("107", "108")]
window_labels = ["99+100 (1985--88)", "103+104 (1993--96)", "107+108 (2001--04)"]

top_n = 10
all_pos = []
all_neg = []

for prev, curr in windows:
    model_path = os.path.join(MODELS, f"06_lasso_window_{prev}_{curr}.joblib")
    model = joblib.load(model_path)
    coefs = model.coef_.ravel()

    # Top positive (Republican)
    pos_idx = np.argsort(coefs)[::-1][:top_n]
    pos_phrases = [(feature_names[i], coefs[i]) for i in pos_idx]
    all_pos.append(pos_phrases)

    # Top negative (Democratic)
    neg_idx = np.argsort(coefs)[:top_n]
    neg_phrases = [(feature_names[i], coefs[i]) for i in neg_idx]
    all_neg.append(neg_phrases)

def fmt_coef(c):
    """Format coefficient: positive as plain number, negative with $-$ prefix."""
    if c >= 0:
        return f"{c:.1f}"
    else:
        return f"$-${abs(c):.1f}"

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"    \centering")
lines.append(r"    \begin{threeparttable}")
lines.append(r"    \caption{Top Partisan Phrases by LASSO Window}")
lines.append(r"    \label{tab:phrases}")
lines.append(r"    \small")
lines.append(r"    \renewcommand{\arraystretch}{1.1}")
lines.append(r"    \begin{tabular}{@{} c l r l r l r @{}}")
lines.append(r"        \toprule")
lines.append(r"        & \multicolumn{2}{c}{" + window_labels[0] + r"} & \multicolumn{2}{c}{" + window_labels[1] + r"} & \multicolumn{2}{c}{" + window_labels[2] + r"} \\")
lines.append(r"        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")

# Panel A: Republican phrases
lines.append(r"        \addlinespace[4pt]")
lines.append(r"        \multicolumn{7}{@{}l}{\emph{Panel A: Republican phrases ($\hat{\beta} > 0$)}} \\")
lines.append(r"        \addlinespace[3pt]")

for rank in range(top_n):
    parts = [f"        {rank+1}"]
    for w in range(3):
        phrase, coef = all_pos[w][rank]
        parts.append(f" & {phrase} & {fmt_coef(coef)}")
    parts.append(r" \\")
    lines.append("".join(parts))

# Panel B: Democratic phrases
lines.append(r"        \addlinespace[6pt]")
lines.append(r"        \multicolumn{7}{@{}l}{\emph{Panel B: Democratic phrases ($\hat{\beta} < 0$)}} \\")
lines.append(r"        \addlinespace[3pt]")

for rank in range(top_n):
    parts = [f"        {rank+1}"]
    for w in range(3):
        phrase, coef = all_neg[w][rank]
        parts.append(f" & {phrase} & {fmt_coef(coef)}")
    parts.append(r" \\")
    lines.append("".join(parts))

lines.append(r"        \addlinespace[2pt]")
lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"    \vspace{6pt}")
lines.append(r"    \begin{tablenotes}[flushleft]")
lines.append(r"        \setlength{\labelsep}{0pt}")
lines.append(r"        \scriptsize")
lines.append(r"        \item \textit{Notes:} Each column shows the 10 phrases with the largest positive (Republican) and most negative (Democratic) LASSO coefficients for the indicated rolling window. Phrases are Porter-stemmed bigrams and unigrams. Coefficients are from a logistic regression with L1 penalty trained on the partisan core of each party. The three windows span the beginning, middle, and end of the sample period.")
lines.append(r"    \end{tablenotes}")
lines.append(r"    \end{threeparttable}")
lines.append(r"\end{table}")
lines.append("")

tex = "\n".join(lines)
out_path = os.path.join(DRAFT, "tab_partisan_phrases.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"  Written: {out_path}")

# ── 3. Variance Decomposition Table ────────────────────────────────────────

print("=== 3. Variance Decomposition Table ===")

panel = pd.read_parquet(os.path.join(NEWS, "10_newspaper_year_panel.parquet"))

outcomes = [
    ("net_slant_norm",      r"Net slant ($\tilde{S}$)"),
    ("politicization_norm", r"Politicization ($\tilde{P}$)"),
    ("net_slant_norm_econ", r"Net slant, econ ($\tilde{S}^{\text{econ}}$)"),
    ("econ_share",          r"Economic article share"),
]

decomp_rows = []
for col, label in outcomes:
    y = panel[["paper", "year", col]].dropna()
    total_var = y[col].var(ddof=0)

    # Newspaper FE: demeaned by newspaper
    paper_means = y.groupby("paper")[col].transform("mean")
    paper_fe_var = paper_means.var(ddof=0)

    # Year FE: demeaned by year
    year_means = y.groupby("year")[col].transform("mean")
    year_fe_var = year_means.var(ddof=0)

    # Grand mean
    grand_mean = y[col].mean()

    # Proper decomposition: Var(y) = Var(paper_FE) + Var(year_FE) + Var(residual)
    # where paper_FE = paper_mean - grand_mean, year_FE = year_mean - grand_mean
    # residual = y - paper_mean - year_mean + grand_mean
    residual = y[col] - paper_means - year_means + grand_mean
    resid_var = residual.var(ddof=0)

    # Cross terms -- compute shares from actual FE components
    paper_fe = paper_means - grand_mean
    year_fe  = year_means  - grand_mean
    paper_share = paper_fe.var(ddof=0) / total_var
    year_share  = year_fe.var(ddof=0)  / total_var
    resid_share = resid_var / total_var

    decomp_rows.append((label, total_var, paper_share, year_share, resid_share))

lines = []
lines.append(r"\begin{table}[H]")
lines.append(r"    \centering")
lines.append(r"    \begin{threeparttable}")
lines.append(r"    \caption{Variance Decomposition of Slant Measures}")
lines.append(r"    \label{tab:variance}")
lines.append(r"    \small")
lines.append(r"    \renewcommand{\arraystretch}{1.15}")
lines.append(r"    \begin{tabular}{@{} l r r r r @{}}")
lines.append(r"        \toprule")
lines.append(r"        & Total variance & Newspaper FE & Year FE & Residual \\")
lines.append(r"        \midrule")

for label, tvar, pshare, yshare, rshare in decomp_rows:
    tvar_str = f"{tvar:.6f}"
    pshare_str = f"{pshare*100:.1f}\\%"
    yshare_str = f"{yshare*100:.1f}\\%"
    rshare_str = f"{rshare*100:.1f}\\%"
    lines.append(f"        {label} & {tvar_str} & {pshare_str} & {yshare_str} & {rshare_str} \\\\")

lines.append(r"        \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"    \vspace{6pt}")
lines.append(r"    \begin{tablenotes}[flushleft]")
lines.append(r"        \setlength{\labelsep}{0pt}")
lines.append(r"        \scriptsize")
lines.append(r"        \item \textit{Notes:} Decomposition of variance in each outcome variable into the share explained by newspaper fixed effects (cross-sectional), year fixed effects (common time trends), and the residual. The residual is the within-newspaper, within-year variation that identifies the event study coefficients. For slant variables, year effects dominate (64--69\%), reflecting common national political dynamics. For economic article share, newspaper effects dominate (45\%), reflecting persistent differences in coverage focus.")
lines.append(r"    \end{tablenotes}")
lines.append(r"    \end{threeparttable}")
lines.append(r"\end{table}")
lines.append("")

tex = "\n".join(lines)
out_path = os.path.join(DRAFT, "tab_variance_decomp.tex")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"  Written: {out_path}")

# ── 4. Newspaper Ranking Figure ────────────────────────────────────────────

print("=== 4. Newspaper Ranking Figure ===")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

paper_means = panel.groupby("paper")["net_slant_norm"].mean()

# Top 10 right (most positive) and top 10 left (most negative)
top_right = paper_means.nlargest(10).sort_values()  # ascending for horizontal bars
top_left  = paper_means.nsmallest(10).sort_values(ascending=False)  # descending for horizontal bars

# Combine: left papers on bottom, right papers on top
combined = pd.concat([top_left, top_right])

fig, ax = plt.subplots(figsize=(8, 7))

colors = ["#4c72b0" if v < 0 else "#c44e52" for v in combined.values]
y_pos = range(len(combined))

ax.barh(y_pos, combined.values, color=colors, edgecolor="none", height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(combined.index, fontsize=9)
ax.set_xlabel("Mean normalized net slant", fontsize=11)
ax.axvline(0, color="black", linewidth=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_pdf = os.path.join(FIGURES, "newspaper_slant_ranking.pdf")
out_png = os.path.join(FIGURES, "newspaper_slant_ranking.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"  Written: {out_pdf}")
print(f"  Written: {out_png}")

print("\n=== All outputs generated successfully ===")
