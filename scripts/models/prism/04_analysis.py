"""
04_analysis.py -- Event study and DiD regressions with PRISM slant.

Uses raw cross-encoder score (sim_R - sim_D, not normalized).

Pipeline:
  1. Aggregate PRISM article slant to newspaper-year panel
  2. Merge with geography / NAFTA / China shock / controls
  3. Event study (baseline + controls)
  4. DiD (3 specs)
  5. Topic-level DiD (PRISM-unique: which policy topics drive NAFTA slant?)
  6. Generate plots

Specification:
  FE: paper + year + division x year
  Cluster: commuting zone (CZ)
"""

import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.prism.config import (
    OUTPUT_DIR, FIG_DIR, TAB_DIR, PANEL_PATH,
    SCORE_CONGRESSES, STATE_TO_DIVISION, congress_to_years,
)

BASE_YEAR = 1993
NAFTA_YEAR = 1994


def build_prism_panel():
    """Aggregate article-level PRISM slant to newspaper-year."""
    prism = pd.read_parquet(OUTPUT_DIR / "article_slant_all.parquet")
    print(f"PRISM articles: {len(prism):,}")

    panel = prism.groupby(["paper", "year"]).agg(
        prism_slant=("prism_slant", "mean"),
        prism_n_articles=("prism_slant", "count"),
        prism_slant_std=("prism_slant", "std"),
        prism_n_topics=("prism_n_topics", "mean"),
    ).reset_index()

    print(f"PRISM newspaper-year obs: {len(panel):,} "
          f"({panel['paper'].nunique()} papers, years {panel['year'].min()}-{panel['year'].max()})")
    return panel


def merge_with_econ():
    """Merge PRISM panel with geography/econ variables."""
    prism_panel = build_prism_panel()

    econ = pd.read_parquet(PANEL_PATH)
    econ_vars = econ[["paper", "year", "fips", "cz",
                      "vulnerability1990_scaled", "manushare1990",
                      "china_shock"]].copy()

    merged = prism_panel.merge(econ_vars, on=["paper", "year"], how="inner")
    merged = merged[merged["cz"].notna() & merged["vulnerability1990_scaled"].notna()].copy()

    merged["state_fips"] = (merged["fips"] // 1000).astype(int)
    merged["division"] = merged["state_fips"].map(STATE_TO_DIVISION)
    merged["paper_id"] = merged["paper"].astype("category").cat.codes

    print(f"Merged panel: {len(merged):,} obs, {merged['paper'].nunique()} papers, "
          f"{merged['cz'].nunique()} CZs")
    return merged


def setup_interactions(df):
    """Create interaction variables for event study and DiD."""
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
    """Run event study: baseline and with controls."""
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
                rows.append({"year": yr, "coef": 0, "se": 0,
                             "ci_lo": 0, "ci_hi": 0})
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
    """Run DiD with 3 specifications."""
    base_yr = years[0]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_rhs = " + ".join(manu_vars)
    china_rhs = " + ".join(china_vars)

    specs = [
        ("spec1", "vuln_x_post",
         "Paper + Year + Division x Year FE"),
        ("spec2", f"vuln_x_post + {manu_rhs}",
         "+ Manushare x Year"),
        ("spec3", f"vuln_x_post + {china_rhs} + {manu_rhs}",
         "+ China shock x Year + Manushare x Year"),
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
    """Plot event study with baseline and controlled specifications."""
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

    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
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


def run_topic_did(df, years):
    """Run DiD for each topic separately (PRISM-unique analysis).

    Uses article-topic scores to identify which policy topics
    drive NAFTA slant changes.
    """
    print(f"\n{'='*60}")
    print("TOPIC-LEVEL DiD ANALYSIS")
    print(f"{'='*60}")

    # Load topic scores + article metadata per-congress, then merge
    all_topic_with_meta = []
    for cong in SCORE_CONGRESSES:
        ts_path = OUTPUT_DIR / f"article_topic_scores_cong_{cong}.parquet"
        art_path = OUTPUT_DIR / f"article_slant_cong_{cong}.parquet"
        if not ts_path.exists() or not art_path.exists():
            continue
        ts = pd.read_parquet(ts_path)
        art = pd.read_parquet(art_path)
        # article_idx is per-congress (0-9999), matches positional index in art
        art_meta = art[["paper", "year"]].reset_index().rename(columns={"index": "article_idx"})
        ts_merged = ts.merge(art_meta, on="article_idx", how="inner")
        all_topic_with_meta.append(ts_merged)

    if not all_topic_with_meta:
        print("  No topic scores found, skipping topic DiD")
        return None

    topic_scores = pd.concat(all_topic_with_meta, ignore_index=True)
    print(f"  Article-topic pairs with metadata: {len(topic_scores):,}")

    # Load econ data once
    econ = pd.read_parquet(PANEL_PATH)
    econ_vars = econ[["paper", "year", "fips", "cz",
                      "vulnerability1990_scaled"]].copy()

    unique_topics = topic_scores["topic_label"].unique()
    print(f"  Topics to test: {len(unique_topics)}")

    topic_results = []
    for topic_label in unique_topics:
        ts_topic = topic_scores[topic_scores["topic_label"] == topic_label].copy()

        # Aggregate to newspaper-year
        topic_panel = ts_topic.groupby(["paper", "year"]).agg(
            topic_slant=("topic_slant", "mean"),
            n_articles=("topic_slant", "count"),
        ).reset_index()

        # Merge with econ
        merged = topic_panel.merge(econ_vars, on=["paper", "year"], how="inner")
        merged = merged[merged["cz"].notna() & merged["vulnerability1990_scaled"].notna()].copy()

        if len(merged) < 50:
            continue

        merged["state_fips"] = (merged["fips"] // 1000).astype(int)
        merged["division"] = merged["state_fips"].map(STATE_TO_DIVISION)
        merged["paper_id"] = merged["paper"].astype("category").cat.codes
        merged["post"] = (merged["year"] >= NAFTA_YEAR).astype(int)
        merged["vuln_x_post"] = merged["vulnerability1990_scaled"] * merged["post"]

        try:
            fml = "topic_slant ~ vuln_x_post | paper_id + year + division^year"
            m = pf.feols(fml, data=merged, vcov={"CRV1": "cz"})
            t = m.tidy().loc["vuln_x_post"]
            topic_results.append({
                "topic_label": topic_label,
                "coef": t["Estimate"],
                "se": t["Std. Error"],
                "pval": t["Pr(>|t|)"],
                "N": m._N,
            })
        except Exception:
            continue

    if not topic_results:
        print("  No valid topic regressions")
        return None

    results_df = pd.DataFrame(topic_results).sort_values("coef", ascending=False)

    # Stars
    results_df["stars"] = ""
    results_df.loc[results_df["pval"] < 0.1, "stars"] = "*"
    results_df.loc[results_df["pval"] < 0.05, "stars"] = "**"
    results_df.loc[results_df["pval"] < 0.01, "stars"] = "***"

    results_df.to_csv(TAB_DIR / "prism_topic_did_results.csv", index=False, float_format="%.6f")

    print(f"\n  Topic DiD results ({len(results_df)} topics):")
    print(f"  {'Topic':<50} {'Coef':>10} {'SE':>8} {'p':>8}")
    print("  " + "-" * 80)
    for _, r in results_df.head(15).iterrows():
        print(f"  {r['topic_label'][:50]:<50} "
              f"{r['coef']:+.6f} {r['se']:.4f} {r['pval']:.4f}{r['stars']}")

    # Plot: topic DiD bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_data = results_df.head(20)
    colors = ["#cb181d" if x > 0 else "#2171b5" for x in plot_data["coef"]]
    y_pos = range(len(plot_data))
    ax.barh(y_pos, plot_data["coef"], xerr=plot_data["se"] * 1.96,
            color=colors, alpha=0.7, capsize=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['topic_label'][:40]}{r['stars']}" for _, r in plot_data.iterrows()],
        fontsize=7
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("DiD Coefficient (vuln x post)")
    ax.set_title("PRISM: Topic-Level NAFTA Effect on Newspaper Slant")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prism_topic_did_bar.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n  Saved: prism_topic_did_bar.png")

    return results_df


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRISM Slant Analysis: Event Study + DiD")
    print("=" * 60)

    # Build and merge panel
    df = merge_with_econ()
    df, years = setup_interactions(df)

    print(f"\nPanel: {len(df):,} obs, years {min(years)}-{max(years)}")
    print(f"Pre-NAFTA: {(df['post']==0).sum():,}, Post-NAFTA: {(df['post']==1).sum():,}")

    print(f"\nPRISM slant summary:")
    print(f"  Mean:   {df['prism_slant'].mean():.6f}")
    print(f"  Std:    {df['prism_slant'].std():.6f}")
    print(f"  Median: {df['prism_slant'].median():.6f}")

    # ── Event Study ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVENT STUDY: PRISM Slant")
    print(f"{'='*60}")

    es_results = run_event_study(df, "prism_slant", years)

    plot_event_study(
        es_results["baseline"], es_results["controls"],
        "PRISM Cross-Encoder Slant",
        FIG_DIR / "prism_event_study.png",
    )

    # Save coefficients
    all_es = []
    for spec, coefs in es_results.items():
        c = coefs.copy()
        c["spec"] = spec
        all_es.append(c)
    es_df = pd.concat(all_es, ignore_index=True)
    es_df.to_csv(TAB_DIR / "prism_event_study_coefficients.csv", index=False, float_format="%.6f")

    # ── DiD ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DiD: PRISM Slant")
    print(f"{'='*60}")

    did_results = run_did(df, "prism_slant", years)

    did_rows = []
    for spec, vals in did_results.items():
        did_rows.append({"depvar": "prism_slant", "spec": spec, **vals})
    did_df = pd.DataFrame(did_rows)
    did_df.to_csv(TAB_DIR / "prism_did_results.csv", index=False, float_format="%.6f")

    # ── Topic-level DiD ─────────────────────────────────────
    topic_did = run_topic_did(df, years)

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Panel: {len(df):,} obs, {df['paper'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs, years {min(years)}-{max(years)}")
    print(f"\nDiD (vuln x post) coefficients:")
    for spec, vals in did_results.items():
        stars = ""
        if vals["pval"] < 0.01: stars = "***"
        elif vals["pval"] < 0.05: stars = "**"
        elif vals["pval"] < 0.1: stars = "*"
        print(f"  {spec}: {vals['coef']:.6f}{stars} (SE={vals['se']:.6f}, p={vals['pval']:.4f})")

    if topic_did is not None:
        sig_topics = topic_did[topic_did["pval"] < 0.1]
        print(f"\nSignificant topics (p<0.1): {len(sig_topics)} / {len(topic_did)}")
        for _, r in sig_topics.iterrows():
            print(f"  {r['topic_label'][:50]}: {r['coef']:+.6f}{r['stars']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
