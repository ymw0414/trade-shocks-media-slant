"""
Generate LaTeX longtable listing all newspapers in the estimation sample.
Outputs draft/tab_newspaper_list.tex
"""

import pandas as pd
from pathlib import Path
import os

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
OUT = BASE / "draft" / "tab_newspaper_list.tex"

stats = pd.read_csv(BASE / "output" / "tables" / "newspaper_list.csv")


def clean_name(name):
    short = name.split(",")[0].strip()
    if short == short.upper() and len(short) > 5:
        KEEP = {"USA"}
        words = [w if w in KEEP else w.title() for w in short.split()]
        short = " ".join(words)
    short = short.replace("The ", "").strip()
    short = short.replace("&", r"\&")
    return short


stats["short_name"] = stats["paper"].apply(clean_name)
stats["mean_articles_int"] = stats["mean_articles"].round(0).astype(int)
stats = stats.sort_values(["state", "short_name"])

NL = "\\\\"  # LaTeX newline

lines = [
    r"\begin{longtable}{llrr}",
    r"\caption{Newspapers in the Estimation Sample}",
    r"\label{tab:newspaper_list}" + NL,
    r"\toprule",
    r"Newspaper & State & Avg.\ Articles/Yr & Mean Slant" + NL,
    r"\midrule",
    r"\endfirsthead",
    r"\multicolumn{4}{l}{\small\textit{Table~\ref{tab:newspaper_list} continued}}" + NL,
    r"\toprule",
    r"Newspaper & State & Avg.\ Articles/Yr & Mean Slant" + NL,
    r"\midrule",
    r"\endhead",
    r"\midrule",
    r"\multicolumn{4}{r}{\small\textit{Continued on next page}}" + NL,
    r"\endfoot",
    r"\bottomrule",
    r"\endlastfoot",
]

for _, row in stats.iterrows():
    name = row["short_name"]
    st = row["state"] if pd.notna(row["state"]) else "--"
    arts = f"{row['mean_articles_int']:,}"
    slant = f"{row['mean_slant']:.3f}"
    lines.append(f"{name} & {st} & {arts} & {slant}" + NL)

lines.append(r"\end{longtable}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {OUT}")
print(f"Total newspapers: {len(stats)}")
