"""
19_balance_test.py

Balance test: compare pre-NAFTA characteristics of high- vs low-vulnerability CZs.
Splits at the median of vulnerability1990_scaled and tests equality of means.

Also runs a joint F-test (OLS of vulnerability on all pre-treatment covariates).

Inputs:
  - data/processed/runs/exp_shvocab_cv/panel/14_regression_panel.parquet

Outputs:
  - output/tables/tab_balance.tex
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
sys.path.insert(0, str(BASE_DIR / "scripts" / "nlp"))
import pipeline_config as cfg

PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
TAB_DIR = BASE_DIR / "output" / "tables"


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)

    # --- Pre-NAFTA (1987-1993) newspaper-year averages collapsed to CZ ---
    pre = panel[panel["year"] <= 1993].copy()
    pre_cz = pre.groupby("cz").agg(
        pre_net_slant=("net_slant_norm", "mean"),
        pre_politicization=("politicization_norm", "mean"),
        pre_right_intensity=("right_norm", "mean"),
        pre_left_intensity=("left_norm", "mean"),
        pre_ext_R=("ext_R", "mean"),
        pre_ext_D=("ext_D", "mean"),
        pre_n_articles=("n_articles", "mean"),
    ).reset_index()

    # --- CZ-level cross-section (time-invariant) ---
    cz_xs = panel.drop_duplicates(subset="cz")[
        ["cz", "vulnerability1990_scaled", "manushare1990",
         "pop1990_total", "less_highschool1990", "bachelor_higher1990",
         "income1989", "china_shock"]
    ].copy()

    df = cz_xs.merge(pre_cz, on="cz", how="inner")
    df = df.dropna(subset=["vulnerability1990_scaled"])

    # Median split
    med = df["vulnerability1990_scaled"].median()
    df["high_vuln"] = (df["vulnerability1990_scaled"] >= med).astype(int)
    n_low = (df["high_vuln"] == 0).sum()
    n_high = (df["high_vuln"] == 1).sum()
    print(f"Median vulnerability: {med:.4f}")
    print(f"Low-vulnerability CZs: {n_low}, High-vulnerability CZs: {n_high}")

    # Variables to test
    variables = [
        ("pre_net_slant", "Pre-NAFTA net slant ($\\tilde{S}$)"),
        ("pre_politicization", "Pre-NAFTA politicization ($\\tilde{P}$)"),
        ("pre_ext_R", "Pre-NAFTA share R-leaning"),
        ("pre_ext_D", "Pre-NAFTA share D-leaning"),
        ("pre_right_intensity", "Pre-NAFTA right intensity ($\\tilde{R}$)"),
        ("pre_left_intensity", "Pre-NAFTA left intensity ($\\tilde{L}$)"),
        ("manushare1990", "Manuf.\\ empl.\\ share, 1990"),
        ("less_highschool1990", "Share less than high school, 1990"),
        ("bachelor_higher1990", "Share bachelor's or higher, 1990"),
        ("income1989", "Per capita income, 1989 (\\$1000s)"),
        ("pop1990_total", "Population, 1990 (1000s)"),
        ("china_shock", "China import shock"),
    ]

    rows = []
    for var, label in variables:
        low = df.loc[df["high_vuln"] == 0, var].dropna()
        high = df.loc[df["high_vuln"] == 1, var].dropna()

        # Scale for readability
        if var == "income1989":
            low = low / 1000
            high = high / 1000
        elif var == "pop1990_total":
            low = low / 1000
            high = high / 1000

        m_low, m_high = low.mean(), high.mean()
        diff = m_high - m_low
        t_stat, p_val = stats.ttest_ind(low, high, equal_var=False)

        rows.append({
            "label": label,
            "mean_low": m_low,
            "mean_high": m_high,
            "diff": diff,
            "p_val": p_val,
        })
        print(f"  {var:30s}  low={m_low:8.3f}  high={m_high:8.3f}  "
              f"diff={diff:+8.3f}  p={p_val:.3f}")

    # --- Joint F-test: regress vulnerability on all pre-treatment covariates ---
    from sklearn.linear_model import LinearRegression

    X_cols = ["pre_net_slant", "pre_politicization", "pre_ext_R",
              "pre_right_intensity", "pre_left_intensity",
              "manushare1990",
              "less_highschool1990", "bachelor_higher1990",
              "income1989", "pop1990_total", "china_shock"]
    reg_df = df[["vulnerability1990_scaled"] + X_cols].dropna()
    y = reg_df["vulnerability1990_scaled"].values
    X = reg_df[X_cols].values

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    n, k = X.shape
    f_stat = (r2 / k) / ((1 - r2) / (n - k - 1))
    f_pval = 1 - stats.f.cdf(f_stat, k, n - k - 1)
    print(f"\nJoint F-test: R²={r2:.3f}, F({k},{n-k-1})={f_stat:.2f}, p={f_pval:.3f}")

    # --- Generate LaTeX table ---
    def fmt_num(x, var_name=""):
        if abs(x) >= 100:
            return f"{x:,.0f}"
        elif abs(x) >= 1:
            return f"{x:.2f}"
        else:
            return f"{x:.3f}"

    def fmt_pval(p):
        if p < 0.01:
            return "$<$0.01"
        elif p < 0.05:
            return f"{p:.2f}"
        elif p < 0.10:
            return f"{p:.2f}"
        else:
            return f"{p:.2f}"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    lines.append(r"    \singlespacing")
    lines.append(r"    \begin{threeparttable}")
    lines.append(r"    \caption{Balance Test: Pre-NAFTA Characteristics by Vulnerability}")
    lines.append(r"    \label{tab:balance}")
    lines.append(r"    \footnotesize")
    lines.append(r"    \renewcommand{\arraystretch}{1.05}")
    lines.append(r"    \begin{tabular}{@{} l c c c c @{}}")
    lines.append(r"        \toprule")
    lines.append(f"        & Low vulnerability & High vulnerability & Difference & $p$-value \\\\")
    lines.append(f"        & ($N = {n_low}$) & ($N = {n_high}$) & & \\\\")
    lines.append(r"        \midrule")

    # Pre-NAFTA media outcomes
    lines.append(r"        \addlinespace[3pt]")
    lines.append(r"        \multicolumn{5}{@{}l}{\emph{Panel A: Pre-NAFTA media outcomes (1987--1993)}} \\")
    lines.append(r"        \addlinespace[2pt]")

    for i, row in enumerate(rows):
        if i == 6:  # Switch to Panel B
            lines.append(r"        \addlinespace[4pt]")
            lines.append(r"        \multicolumn{5}{@{}l}{\emph{Panel B: Socioeconomic characteristics}} \\")
            lines.append(r"        \addlinespace[2pt]")

        m_l = fmt_num(row["mean_low"])
        m_h = fmt_num(row["mean_high"])
        d = fmt_num(row["diff"])
        if row["diff"] > 0:
            d = f"+{d}" if not d.startswith("+") else d
        p = fmt_pval(row["p_val"])
        lines.append(f"        {row['label']} & {m_l} & {m_h} & {d} & {p} \\\\")

    lines.append(r"        \addlinespace[3pt]")
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \vspace{4pt}")
    lines.append(r"    \begin{tablenotes}[flushleft]")
    lines.append(r"        \setlength{\labelsep}{0pt}")
    lines.append(r"        \scriptsize")
    lines.append(r"        \item \textit{Notes:} Commuting zones split at the median of NAFTA vulnerability (1990). Panel~A reports pre-NAFTA (1987--1993) means of newspaper-level outcomes averaged within each CZ. Panel~B reports 1990 Census characteristics. The ``Difference'' column reports High $-$ Low. $p$-values are from two-sided Welch $t$-tests.")
    lines.append(r"    \end{tablenotes}")
    lines.append(r"    \end{threeparttable}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    out = TAB_DIR / "tab_balance.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
