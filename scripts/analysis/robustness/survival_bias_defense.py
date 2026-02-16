"""
Survival bias defense for the newspaper sample.

Tests whether the newspaper panel composition is systematically related
to trade vulnerability, which would bias our results if high-vulnerability
newspapers exit the sample differentially.

Analyses:
  1. Panel composition: entry/exit timing of newspapers
  2. Vulnerability comparison: full-span vs partial-span newspapers (t-test)
  3. Correlation: vulnerability vs entry year and survival length
  4. Balanced-panel-only DiD: restrict to newspapers present in all years
  5. Entry/exit event study: are entering/exiting papers differentially vulnerable?

Outputs:
  - output/tables/survival_bias_tests.csv
  - output/figures/survival_bias_defense.pdf
"""

import os, sys
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
from scipy import stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

END_YEAR = 2004

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


def run_did(df, depvar, years):
    """Run spec 3 DiD."""
    base_yr = years[0]
    manu_vars = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_vars = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])
    fml = f"{depvar} ~ vuln_x_post + {china_vars} + {manu_vars} | paper_id + year + division^year"
    m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
    t = m.tidy().loc["vuln_x_post"]
    return t["Estimate"], t["Std. Error"], t["t value"], t["Pr(>|t|)"], m._N


def prepare_panel(df):
    """Add regression variables."""
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()
    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= 1994).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]
    years = sorted(df["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)
    return df, years


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel ...")
    full_panel = pd.read_parquet(PANEL_PATH)
    full_panel = full_panel[full_panel["year"] <= END_YEAR].copy()

    # ===================================================================
    # 1. Panel composition
    # ===================================================================
    print("\n" + "=" * 60)
    print("  PANEL COMPOSITION")
    print("=" * 60)

    paper_stats = full_panel.groupby("paper").agg(
        min_year=("year", "min"),
        max_year=("year", "max"),
        n_years=("year", "nunique"),
        vuln=("vulnerability1990_scaled", "first"),
        cz=("cz", "first"),
    ).reset_index()

    all_years = sorted(full_panel["year"].unique())
    min_yr, max_yr = min(all_years), max(all_years)
    n_total_years = len(all_years)

    # Full-span = present from min to max year
    paper_stats["full_span"] = (
        (paper_stats["min_year"] == min_yr) & (paper_stats["max_year"] == max_yr)
    )
    paper_stats["early_exit"] = paper_stats["max_year"] < max_yr
    paper_stats["late_entry"] = paper_stats["min_year"] > min_yr

    n_papers = len(paper_stats)
    n_full = paper_stats["full_span"].sum()
    n_exit = paper_stats["early_exit"].sum()
    n_enter = paper_stats["late_entry"].sum()

    print(f"  Total newspapers: {n_papers}")
    print(f"  Year range: {min_yr}-{max_yr} ({n_total_years} years)")
    print(f"  Full-span (all years): {n_full} ({n_full/n_papers*100:.0f}%)")
    print(f"  Late entry (after {min_yr}): {n_enter}")
    print(f"  Early exit (before {max_yr}): {n_exit}")
    print(f"  Mean years per paper: {paper_stats['n_years'].mean():.1f}")

    # Year-by-year counts
    yearly_counts = full_panel.groupby("year")["paper"].nunique()
    print(f"\n  Papers per year:")
    for yr, cnt in yearly_counts.items():
        print(f"    {yr}: {cnt}")

    # ===================================================================
    # 2. Vulnerability comparison
    # ===================================================================
    print("\n" + "=" * 60)
    print("  VULNERABILITY vs SURVIVAL")
    print("=" * 60)

    has_vuln = paper_stats[paper_stats["vuln"].notna()]
    full_vuln = has_vuln[has_vuln["full_span"]]["vuln"]
    part_vuln = has_vuln[~has_vuln["full_span"]]["vuln"]

    print(f"\n  Full-span papers (n={len(full_vuln)}):")
    print(f"    Mean vulnerability: {full_vuln.mean():.4f}")
    print(f"    Median: {full_vuln.median():.4f}")
    print(f"    Std: {full_vuln.std():.4f}")

    print(f"\n  Partial-span papers (n={len(part_vuln)}):")
    print(f"    Mean vulnerability: {part_vuln.mean():.4f}")
    print(f"    Median: {part_vuln.median():.4f}")
    print(f"    Std: {part_vuln.std():.4f}")

    t_stat, p_val = stats.ttest_ind(full_vuln.dropna(), part_vuln.dropna())
    print(f"\n  t-test (full vs partial): t={t_stat:.3f}, p={p_val:.4f}")

    # Correlation: vulnerability vs entry year
    r_entry, p_entry = stats.pearsonr(
        has_vuln["min_year"], has_vuln["vuln"]
    )
    print(f"  Corr(vulnerability, entry year): r={r_entry:.4f}, p={p_entry:.4f}")

    # Correlation: vulnerability vs number of years
    r_nyears, p_nyears = stats.pearsonr(
        has_vuln["n_years"], has_vuln["vuln"]
    )
    print(f"  Corr(vulnerability, n_years): r={r_nyears:.4f}, p={p_nyears:.4f}")

    # ===================================================================
    # 3. Balanced panel DiD
    # ===================================================================
    print("\n" + "=" * 60)
    print("  BALANCED PANEL DiD")
    print("=" * 60)

    # Full sample
    df_full, years = prepare_panel(full_panel)

    # Balanced: only full-span papers
    full_span_papers = paper_stats[paper_stats["full_span"]]["paper"].tolist()
    df_balanced = df_full[df_full["paper"].isin(full_span_papers)].copy()
    df_balanced["paper_id"] = df_balanced["paper"].astype("category").cat.codes

    outcomes = [
        ("net_slant_norm", "Net Slant"),
        ("ext_R", "Share R-Leaning"),
        ("ext_D", "Share D-Leaning"),
    ]

    comparison_rows = []
    print(f"\n  {'Outcome':<20s}  {'Full (N)':>8s}  {'Full coef':>10s}  "
          f"{'Bal (N)':>8s}  {'Bal coef':>10s}  {'% diff':>7s}")
    print("-" * 75)

    for depvar, label in outcomes:
        c_f, se_f, t_f, p_f, n_f = run_did(df_full, depvar, years)
        c_b, se_b, t_b, p_b, n_b = run_did(df_balanced, depvar, years)
        pct_diff = (c_b - c_f) / abs(c_f) * 100 if c_f != 0 else 0

        print(f"  {label:<20s}  {n_f:>8}  {c_f:>10.4f}  {n_b:>8}  {c_b:>10.4f}  {pct_diff:>6.1f}%")

        comparison_rows.append({
            "depvar": depvar, "label": label,
            "full_N": n_f, "full_coef": c_f, "full_se": se_f, "full_p": p_f,
            "balanced_N": n_b, "balanced_coef": c_b, "balanced_se": se_b, "balanced_p": p_b,
            "pct_diff": pct_diff,
        })

    # ===================================================================
    # Save results
    # ===================================================================
    # Summary table
    test_results = [{
        "test": "T-test (full vs partial vulnerability)",
        "statistic": t_stat, "p_value": p_val,
    }, {
        "test": "Corr(vulnerability, entry year)",
        "statistic": r_entry, "p_value": p_entry,
    }, {
        "test": "Corr(vulnerability, n_years observed)",
        "statistic": r_nyears, "p_value": p_nyears,
    }]
    test_df = pd.DataFrame(test_results)

    comp_df = pd.DataFrame(comparison_rows)
    all_df = pd.concat([test_df, comp_df], axis=1)

    csv_path = TAB_DIR / "survival_bias_tests.csv"
    comp_df.to_csv(csv_path, index=False, float_format="%.6f")
    test_df.to_csv(TAB_DIR / "survival_bias_statistical_tests.csv",
                   index=False, float_format="%.6f")
    print(f"\n  Results saved: {csv_path}")

    # ===================================================================
    # Figure: 2x2 panel
    # ===================================================================
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel a: Vulnerability distribution by survival status
    ax = axes[0, 0]
    ax.hist(full_vuln.dropna(), bins=15, alpha=0.6, color="#2d2d2d",
            label=f"Full span (n={len(full_vuln)})", density=True)
    ax.hist(part_vuln.dropna(), bins=15, alpha=0.6, color="#dd8452",
            label=f"Partial span (n={len(part_vuln)})", density=True)
    ax.axvline(full_vuln.mean(), color="#2d2d2d", linewidth=1.5, linestyle="--")
    ax.axvline(part_vuln.mean(), color="#dd8452", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Vulnerability (scaled)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("(a) Vulnerability by Panel Status", fontsize=10)
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95, f"t-test p = {p_val:.3f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel b: Entry year vs vulnerability scatter
    ax = axes[0, 1]
    ax.scatter(has_vuln["min_year"], has_vuln["vuln"], alpha=0.4, s=20, color="#333")
    z = np.polyfit(has_vuln["min_year"], has_vuln["vuln"], 1)
    x_line = np.array([min_yr, max_yr])
    ax.plot(x_line, np.polyval(z, x_line), color="#bf6b63", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Entry year", fontsize=9)
    ax.set_ylabel("Vulnerability (scaled)", fontsize=9)
    ax.set_title("(b) Vulnerability vs Entry Timing", fontsize=10)
    ax.text(0.03, 0.95, f"r = {r_entry:.3f}, p = {p_entry:.3f}",
            transform=ax.transAxes, fontsize=8, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel c: Papers per year
    ax = axes[1, 0]
    ax.bar(yearly_counts.index, yearly_counts.values, color="#2d2d2d",
           edgecolor="white", linewidth=0.5)
    ax.axvline(1993.5, color="#7a7a7a", linewidth=1.0, linestyle="--",
               alpha=0.7, label="NAFTA (1994)")
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Number of newspapers", fontsize=9)
    ax.set_title("(c) Panel Size Over Time", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel d: DiD comparison (full vs balanced)
    ax = axes[1, 1]
    x_pos = np.arange(len(outcomes))
    width = 0.35
    full_coefs = [comparison_rows[i]["full_coef"] for i in range(len(outcomes))]
    full_ses = [comparison_rows[i]["full_se"] for i in range(len(outcomes))]
    bal_coefs = [comparison_rows[i]["balanced_coef"] for i in range(len(outcomes))]
    bal_ses = [comparison_rows[i]["balanced_se"] for i in range(len(outcomes))]

    ax.bar(x_pos - width/2, full_coefs, width, label="Full panel",
           color="#2d2d2d", edgecolor="white")
    ax.errorbar(x_pos - width/2, full_coefs, yerr=[1.96*s for s in full_ses],
                fmt="none", color="black", capsize=3)
    ax.bar(x_pos + width/2, bal_coefs, width, label="Balanced panel",
           color="#dd8452", edgecolor="white")
    ax.errorbar(x_pos + width/2, bal_coefs, yerr=[1.96*s for s in bal_ses],
                fmt="none", color="black", capsize=3)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([o[1] for o in outcomes], fontsize=8)
    ax.set_ylabel("DiD coefficient", fontsize=9)
    ax.set_title("(d) Full vs Balanced Panel DiD", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig_path = FIG_DIR / "survival_bias_defense.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
