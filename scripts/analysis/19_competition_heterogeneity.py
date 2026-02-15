"""
19_competition_heterogeneity.py

Newspaper market competition heterogeneity analysis.
Tests whether NAFTA slant effect differs by local newspaper market structure
using number of daily newspapers per CZ from Gentzkow, Shapiro & Sinkinson
(2011, ICPSR 30261).

Approach:
  1. Interaction model: vuln_x_post + vuln_x_post x HighCompetition
  2. Split-sample: separate regressions for high vs low competition CZs
  3. Continuous interaction: vuln_x_post x n_dailies (log)

Inputs:
  - data/processed/panel/14_regression_panel.parquet
  - data/processed/panel/cz_newspaper_competition.parquet

Outputs:
  - Console results
  - output/tables/competition_heterogeneity.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import pyfixest as pf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
COMP_PATH = BASE_DIR / "data" / "processed" / "panel" / "cz_newspaper_competition.parquet"
TAB_DIR = cfg.TAB_DIR

NAFTA_YEAR = 1994
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


def stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def load_panel():
    """Load regression panel with competition variable."""
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    # Controls
    years = sorted(df["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    # Merge competition
    comp = pd.read_parquet(COMP_PATH)
    df = df.merge(comp, on="cz", how="left")
    n_matched = df["n_dailies_1988"].notna().sum()
    print(f"  Competition merge: {n_matched}/{len(df)} obs matched")

    # Competition variables
    median_dailies = df.groupby("cz")["n_dailies_1988"].first().median()
    df["high_competition"] = (df["n_dailies_1988"] >= median_dailies).astype(int)
    df["log_n_dailies"] = np.log(df["n_dailies_1988"].clip(lower=1))

    # Interaction terms
    df["vuln_x_post_x_hicomp"] = df["vuln_x_post"] * df["high_competition"]
    df["vuln_x_post_x_logdailies"] = df["vuln_x_post"] * df["log_n_dailies"]

    return df, years, median_dailies


def part1_interaction(df, years, median_dailies):
    """Interaction model: vuln_x_post + vuln_x_post x HighCompetition."""
    print("\n" + "=" * 80)
    print("  PART 1: Competition Heterogeneity (Interaction Model)")
    print(f"  HighCompetition = 1 if n_dailies_1988 >= {median_dailies:.0f} (median)")
    print("=" * 80)

    base_yr = years[0]
    manu_vars = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_vars = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])
    controls = f"{china_vars} + {manu_vars}"

    n_hi = df[df["high_competition"] == 1]["paper"].nunique()
    n_lo = df[df["high_competition"] == 0]["paper"].nunique()
    cz_hi = df[df["high_competition"] == 1]["cz"].nunique()
    cz_lo = df[df["high_competition"] == 0]["cz"].nunique()
    print(f"\n  High competition: {n_hi} papers, {cz_hi} CZs")
    print(f"  Low competition:  {n_lo} papers, {cz_lo} CZs")

    outcomes = [
        ("net_slant_norm",       "Net Slant"),
        ("right_norm",           "R Intensity (Uncond)"),
        ("left_norm",            "L Intensity (Uncond)"),
        ("politicization_norm",  "Politicization"),
        ("ext_R",                "Share R-Leaning"),
        ("ext_D",                "Share D-Leaning"),
        ("int_R_norm",           "R Intensity (Intens)"),
        ("int_D_norm",           "D Intensity (Intens)"),
    ]

    print(f"\n  {'Outcome':<24} {'Vuln*Post':>14} {'Vuln*Post*HiComp':>19} "
          f"{'Low comp':>12} {'High comp':>12}")
    print(f"  {'':<24} {'coef (se)':>14} {'coef (se)':>19} "
          f"{'(implied)':>12} {'(implied)':>12}")
    print("  " + "-" * 85)

    results = []
    for depvar, label in outcomes:
        fml = (f"{depvar} ~ vuln_x_post + vuln_x_post_x_hicomp + {controls} "
               f"| paper_id + year + division^year")
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy()

        base_coef = t.loc["vuln_x_post", "Estimate"]
        base_se = t.loc["vuln_x_post", "Std. Error"]
        base_p = t.loc["vuln_x_post", "Pr(>|t|)"]

        inter_coef = t.loc["vuln_x_post_x_hicomp", "Estimate"]
        inter_se = t.loc["vuln_x_post_x_hicomp", "Std. Error"]
        inter_p = t.loc["vuln_x_post_x_hicomp", "Pr(>|t|)"]

        lo_effect = base_coef
        hi_effect = base_coef + inter_coef

        print(f"  {label:<24} {base_coef:>8.4f}{stars(base_p):3s}({base_se:.4f}) "
              f"{inter_coef:>11.4f}{stars(inter_p):3s}({inter_se:.4f}) "
              f"{lo_effect:>12.4f} {hi_effect:>12.4f}")

        results.append({
            'depvar': depvar, 'label': label, 'model': 'interaction',
            'base_coef': base_coef, 'base_se': base_se, 'base_p': base_p,
            'inter_coef': inter_coef, 'inter_se': inter_se, 'inter_p': inter_p,
            'lo_implied': lo_effect, 'hi_implied': hi_effect,
            'N': m._N,
        })

    print(f"\n  * p<0.1, ** p<0.05, *** p<0.01")
    print(f"  Vuln*Post = effect in low-competition CZs")
    print(f"  Vuln*Post*HiComp = differential for high-competition CZs")
    return results


def part2_split_sample(df, years, median_dailies):
    """Separate regressions for high vs low competition."""
    print("\n" + "=" * 80)
    print("  PART 2: Split-Sample DiD (High vs Low Competition)")
    print(f"  Split at median n_dailies = {median_dailies:.0f}")
    print("=" * 80)

    base_yr = years[0]
    manu_vars = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_vars = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])
    controls = f"{china_vars} + {manu_vars}"

    outcomes = [
        ("net_slant_norm",       "Net Slant"),
        ("right_norm",           "R Intensity (Uncond)"),
        ("left_norm",            "L Intensity (Uncond)"),
        ("politicization_norm",  "Politicization"),
        ("ext_R",                "Share R-Leaning"),
        ("ext_D",                "Share D-Leaning"),
    ]

    results = []
    for group_name, mask in [("High competition", df["high_competition"] == 1),
                              ("Low competition", df["high_competition"] == 0)]:
        sub = df[mask].copy()
        sub["paper_id"] = sub["paper"].astype("category").cat.codes
        print(f"\n  --- {group_name}: {len(sub)} obs, {sub['paper'].nunique()} papers, "
              f"{sub['cz'].nunique()} CZs ---")

        for depvar, label in outcomes:
            fml = f"{depvar} ~ vuln_x_post + {controls} | paper_id + year + division^year"
            try:
                m = pf.feols(fml, data=sub, vcov={"CRV1": "cz"})
                t = m.tidy().loc["vuln_x_post"]
                print(f"    {label:<24} {t['Estimate']:>8.4f}{stars(t['Pr(>|t|)']):3s} "
                      f"(SE={t['Std. Error']:.4f}), N={m._N}")

                results.append({
                    'depvar': depvar, 'label': label, 'model': f'split_{group_name}',
                    'coef': t['Estimate'], 'se': t['Std. Error'], 'p': t['Pr(>|t|)'],
                    'N': m._N,
                })
            except Exception as e:
                print(f"    {label:<24} FAILED: {e}")

    return results


def part3_continuous(df, years):
    """Continuous interaction: vuln_x_post x log(n_dailies)."""
    print("\n" + "=" * 80)
    print("  PART 3: Continuous Interaction (log n_dailies)")
    print("=" * 80)

    base_yr = years[0]
    manu_vars = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_vars = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])
    controls = f"{china_vars} + {manu_vars}"

    outcomes = [
        ("net_slant_norm",       "Net Slant"),
        ("right_norm",           "R Intensity (Uncond)"),
        ("left_norm",            "L Intensity (Uncond)"),
        ("politicization_norm",  "Politicization"),
        ("ext_R",                "Share R-Leaning"),
        ("ext_D",                "Share D-Leaning"),
    ]

    print(f"\n  log(n_dailies): mean={df['log_n_dailies'].mean():.2f}, "
          f"sd={df['log_n_dailies'].std():.2f}, "
          f"range=[{df['log_n_dailies'].min():.2f}, {df['log_n_dailies'].max():.2f}]")

    print(f"\n  {'Outcome':<24} {'Vuln*Post':>14} {'Vuln*Post*lnDaily':>20}")
    print(f"  {'':<24} {'coef (se)':>14} {'coef (se)':>20}")
    print("  " + "-" * 60)

    results = []
    for depvar, label in outcomes:
        fml = (f"{depvar} ~ vuln_x_post + vuln_x_post_x_logdailies + {controls} "
               f"| paper_id + year + division^year")
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy()

        base_coef = t.loc["vuln_x_post", "Estimate"]
        base_se = t.loc["vuln_x_post", "Std. Error"]
        base_p = t.loc["vuln_x_post", "Pr(>|t|)"]

        inter_coef = t.loc["vuln_x_post_x_logdailies", "Estimate"]
        inter_se = t.loc["vuln_x_post_x_logdailies", "Std. Error"]
        inter_p = t.loc["vuln_x_post_x_logdailies", "Pr(>|t|)"]

        print(f"  {label:<24} {base_coef:>8.4f}{stars(base_p):3s}({base_se:.4f}) "
              f"{inter_coef:>12.4f}{stars(inter_p):3s}({inter_se:.4f})")

        results.append({
            'depvar': depvar, 'label': label, 'model': 'continuous',
            'base_coef': base_coef, 'base_se': base_se, 'base_p': base_p,
            'inter_coef': inter_coef, 'inter_se': inter_se, 'inter_p': inter_p,
            'N': m._N,
        })

    print(f"\n  * p<0.1, ** p<0.05, *** p<0.01")
    return results


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  NEWSPAPER COMPETITION HETEROGENEITY ANALYSIS")
    print("  (Gentzkow, Shapiro & Sinkinson 2011 — ICPSR 30261)")
    print("=" * 80)

    print("\nLoading panel ...")
    df, years, median_dailies = load_panel()
    print(f"  {len(df):,} obs, {df['paper'].nunique()} papers, "
          f"{df['cz'].nunique()} CZs, years {df['year'].min()}-{df['year'].max()}")

    # Distribution summary
    cz_comp = df.groupby("cz")["n_dailies_1988"].first()
    print(f"\n  n_dailies_1988 distribution (in sample):")
    print(f"    median={cz_comp.median():.0f}, mean={cz_comp.mean():.1f}, "
          f"min={cz_comp.min():.0f}, max={cz_comp.max():.0f}")
    print(f"    25th pctl={cz_comp.quantile(0.25):.0f}, "
          f"75th pctl={cz_comp.quantile(0.75):.0f}")

    res1 = part1_interaction(df, years, median_dailies)
    res2 = part2_split_sample(df, years, median_dailies)
    res3 = part3_continuous(df, years)

    # Save all results
    all_res = pd.DataFrame(res1 + res2 + res3)
    csv_path = TAB_DIR / "competition_heterogeneity.csv"
    all_res.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Results saved: {csv_path}")

    print("\n" + "=" * 80)
    print("  Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()
