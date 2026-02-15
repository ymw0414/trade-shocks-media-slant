"""
05_analysis.py -- Event study and DiD with fine-tuned BERT slant.

Uses raw logit difference (logit_R - logit_D) as the slant measure.
Same regression specification as SBERT and TF-IDF pipelines.
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

BERT_DIR = BASE_DIR / "data" / "processed" / "models" / "finetuned_bert"
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel" / "14_regression_panel.parquet"

FIG_DIR = BERT_DIR / "figures"
TAB_DIR = BERT_DIR / "tables"

BASE_YEAR = 1993
NAFTA_YEAR = 1994

STATE_TO_DIVISION = {
    9: 1, 23: 1, 25: 1, 33: 1, 44: 1, 50: 1,
    34: 2, 36: 2, 42: 2,
    17: 3, 18: 3, 26: 3, 39: 3, 55: 3,
    19: 4, 20: 4, 27: 4, 29: 4, 31: 4, 38: 4, 46: 4,
    10: 5, 11: 5, 12: 5, 13: 5, 24: 5, 37: 5, 45: 5, 51: 5, 54: 5,
    1: 6, 21: 6, 28: 6, 47: 6,
    5: 7, 22: 7, 40: 7, 48: 7,
    4: 8, 8: 8, 16: 8, 30: 8, 32: 8, 35: 8, 49: 8, 56: 8,
    2: 9, 6: 9, 15: 9, 41: 9, 53: 9,
}


def build_panel():
    bert = pd.read_parquet(BERT_DIR / "article_slant_all.parquet")
    print(f"BERT articles: {len(bert):,}")

    panel = bert.groupby(["paper", "year"]).agg(
        bert_slant=("slant", "mean"),
        bert_prob_R=("prob_R", "mean"),
        bert_n_articles=("slant", "count"),
    ).reset_index()

    econ = pd.read_parquet(PANEL_PATH)
    econ_vars = econ[["paper", "year", "fips", "cz",
                      "vulnerability1990_scaled", "manushare1990",
                      "china_shock"]].copy()

    merged = panel.merge(econ_vars, on=["paper", "year"], how="inner")
    merged = merged[merged["cz"].notna() & merged["vulnerability1990_scaled"].notna()].copy()

    merged["state_fips"] = (merged["fips"] // 1000).astype(int)
    merged["division"] = merged["state_fips"].map(STATE_TO_DIVISION)
    merged["paper_id"] = merged["paper"].astype("category").cat.codes

    print(f"Merged panel: {len(merged):,} obs, {merged['paper'].nunique()} papers, "
          f"{merged['cz'].nunique()} CZs")
    return merged


def setup_interactions(df):
    years = sorted(df["year"].unique())
    base_yr = years[0]

    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]

    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    return df, years


def run_event_study(df, depvar, years):
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    vul_rhs = " + ".join(vul_vars)

    base_yr = years[0]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    control_rhs = " + ".join(china_vars + manu_vars)

    results = {}
    for spec_name, rhs in [("baseline", vul_rhs),
                            ("controls", f"{vul_rhs} + {control_rhs}")]:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        print(f"  {spec_name}: N={m._N}, R2={m._r2:.4f}")

        rows = []
        t = m.tidy()
        for yr in years:
            if yr == BASE_YEAR:
                rows.append({"year": yr, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
                continue
            vname = f"vul_{yr}"
            if vname in t.index:
                r = t.loc[vname]
                rows.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                             "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
            else:
                rows.append({"year": yr, "coef": np.nan, "se": np.nan,
                             "ci_lo": np.nan, "ci_hi": np.nan})
        results[spec_name] = pd.DataFrame(rows)

    return results


def run_did(df, depvar, years):
    base_yr = years[0]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_rhs = " + ".join(manu_vars)
    china_rhs = " + ".join(china_vars)

    specs = [
        ("spec1", "vuln_x_post", "Paper + Year + Division x Year FE"),
        ("spec2", f"vuln_x_post + {manu_rhs}", "+ Manushare x Year"),
        ("spec3", f"vuln_x_post + {china_rhs} + {manu_rhs}", "+ China + Manu x Year"),
    ]

    results = {}
    for spec_name, rhs, desc in specs:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy().loc["vuln_x_post"]
        results[spec_name] = {
            "coef": t["Estimate"], "se": t["Std. Error"],
            "pval": t["Pr(>|t|)"], "N": m._N, "R2": m._r2,
        }
        stars = ""
        if t["Pr(>|t|)"] < 0.01: stars = "***"
        elif t["Pr(>|t|)"] < 0.05: stars = "**"
        elif t["Pr(>|t|)"] < 0.1: stars = "*"
        print(f"  {desc}")
        print(f"    vuln x post = {t['Estimate']:.6f}{stars} (SE={t['Std. Error']:.6f}), "
              f"p={t['Pr(>|t|)']:.4f}, N={m._N}, R2={m._r2:.4f}")

    return results


def plot_event_study(coefs_base, coefs_ctrl, depvar_label, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    yrs = coefs_base["year"].values
    offset = 0.15

    ax.errorbar(yrs - offset, coefs_base["coef"],
                yerr=[coefs_base["coef"] - coefs_base["ci_lo"],
                      coefs_base["ci_hi"] - coefs_base["coef"]],
                fmt="o", color="#cb181d", markersize=5, capsize=3,
                linewidth=1.2, label="Baseline")

    ax.errorbar(yrs + offset, coefs_ctrl["coef"],
                yerr=[coefs_ctrl["coef"] - coefs_ctrl["ci_lo"],
                      coefs_ctrl["ci_hi"] - coefs_ctrl["coef"]],
                fmt="s", color="#2171b5", markersize=5, capsize=3,
                linewidth=1.2, label="+ China shock, manushare")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7, label="NAFTA (1994)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability x Year", fontsize=12)
    ax.set_title(f"Event Study: {depvar_label}", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks(yrs)
    ax.set_xticklabels([str(int(yr)) for yr in yrs], fontsize=9, rotation=45)
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Fine-Tuned BERT Analysis: Event Study + DiD")
    print("=" * 60)

    df = build_panel()
    df, years = setup_interactions(df)

    print(f"\nPanel: {len(df):,} obs, years {min(years)}-{max(years)}")
    print(f"Pre-NAFTA: {(df['post']==0).sum():,}, Post-NAFTA: {(df['post']==1).sum():,}")

    print(f"\nBERT slant summary:")
    print(f"  Mean:   {df['bert_slant'].mean():.4f}")
    print(f"  Std:    {df['bert_slant'].std():.4f}")
    print(f"  Median: {df['bert_slant'].median():.4f}")

    # Event Study
    print(f"\n{'='*60}")
    print("EVENT STUDY: BERT Slant")
    print(f"{'='*60}")

    es_results = run_event_study(df, "bert_slant", years)
    plot_event_study(
        es_results["baseline"], es_results["controls"],
        "Fine-Tuned BERT Slant (logit_R - logit_D)",
        FIG_DIR / "bert_event_study_slant.png",
    )

    all_es = []
    for spec, coefs in es_results.items():
        c = coefs.copy()
        c["spec"] = spec
        all_es.append(c)
    pd.concat(all_es, ignore_index=True).to_csv(
        TAB_DIR / "bert_event_study_coefficients.csv", index=False, float_format="%.6f")

    # DiD
    print(f"\n{'='*60}")
    print("DiD: BERT Slant")
    print(f"{'='*60}")

    did_results = run_did(df, "bert_slant", years)

    did_rows = [{"depvar": "bert_slant", "spec": spec, **vals}
                for spec, vals in did_results.items()]
    pd.DataFrame(did_rows).to_csv(
        TAB_DIR / "bert_did_results.csv", index=False, float_format="%.6f")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Panel: {len(df):,} obs, {df['paper'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs")
    print(f"\nDiD (vuln x post) coefficients:")
    for spec, vals in did_results.items():
        stars = ""
        if vals["pval"] < 0.01: stars = "***"
        elif vals["pval"] < 0.05: stars = "**"
        elif vals["pval"] < 0.1: stars = "*"
        print(f"  {spec}: {vals['coef']:.6f}{stars} (SE={vals['se']:.6f}, p={vals['pval']:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
