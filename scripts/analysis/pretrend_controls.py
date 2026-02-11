"""
Try additional controls to address pre-trends in the event study.

Specs (all include paper FE + year FE + division×year FE, cluster CZ):
  1. Baseline:    vul×year
  2. Current:     + china×year + manushare×year
  3. + Income:    + income1989×year
  4. + Education: + college1990×year
  5. + Pop:       + pop1990_total×year
  6. Kitchen sink: all of the above
"""

import os
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = BASE_DIR / "data" / "processed" / "panel" / "14_regression_panel.parquet"
FIG_DIR = BASE_DIR / "output" / "figures"

BASE_YEAR = 1993
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


def load_panel():
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes

    years = sorted(df["year"].unique())
    base_yr = years[0]

    # Standardize continuous controls for comparability
    for col in ["income1989", "college1990", "pop1990_total"]:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    # Vulnerability × year (omit BASE_YEAR)
    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]

    # China × year (omit first year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    # Manushare × year
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

    # Income × year
    for yr in years:
        if yr == base_yr:
            continue
        df[f"inc_{yr}"] = (df["year"] == yr).astype(float) * df["income1989_z"].fillna(0)

    # College × year
    for yr in years:
        if yr == base_yr:
            continue
        df[f"col_{yr}"] = (df["year"] == yr).astype(float) * df["college1990_z"].fillna(0)

    # Pop × year
    for yr in years:
        if yr == base_yr:
            continue
        df[f"pop_{yr}"] = (df["year"] == yr).astype(float) * df["pop1990_total_z"].fillna(0)

    return df, years


def extract_vul_coefs(model, years):
    t = model.tidy()
    rows = []
    for yr in years:
        if yr == BASE_YEAR:
            rows.append({"year": yr, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
            continue
        r = t.loc[f"vul_{yr}"]
        rows.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                      "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
    return pd.DataFrame(rows)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading panel ...")
    df, years = load_panel()
    base_yr = years[0]
    print(f"  {len(df):,} obs, {df['paper_id'].nunique()} papers, {df['cz'].nunique()} CZs")

    # Check correlations
    print("\n=== Correlations with vulnerability1990_scaled ===")
    for col in ["china_shock", "manushare1990", "income1989", "college1990", "pop1990_total"]:
        r = df.drop_duplicates("cz")[[col, "vulnerability1990_scaled"]].dropna().corr().iloc[0, 1]
        print(f"  {col:25s}: {r:.3f}")

    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    vul_rhs = " + ".join(vul_vars)

    def make_vars(prefix):
        return [f"{prefix}_{yr}" for yr in years if yr != base_yr]

    china_rhs = " + ".join(make_vars("china"))
    manu_rhs = " + ".join(make_vars("manu"))
    inc_rhs = " + ".join(make_vars("inc"))
    col_rhs = " + ".join(make_vars("col"))
    pop_rhs = " + ".join(make_vars("pop"))

    specs = {
        "1. Baseline":      vul_rhs,
        "2. +China,Manu":   f"{vul_rhs} + {china_rhs} + {manu_rhs}",
        "3. +Income":       f"{vul_rhs} + {china_rhs} + {manu_rhs} + {inc_rhs}",
        "4. +Education":    f"{vul_rhs} + {china_rhs} + {manu_rhs} + {col_rhs}",
        "5. +Population":   f"{vul_rhs} + {china_rhs} + {manu_rhs} + {pop_rhs}",
        "6. Kitchen sink":  f"{vul_rhs} + {china_rhs} + {manu_rhs} + {inc_rhs} + {col_rhs} + {pop_rhs}",
    }

    for depvar, dep_label in [("net_slant_norm", "Net Slant"),
                               ("net_slant_norm_econ", "Net Slant (Econ)")]:
        print(f"\n{'='*60}")
        print(f"  {dep_label}")
        print(f"{'='*60}")

        all_coefs = {}
        for spec_name, rhs in specs.items():
            fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
            m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
            coefs = extract_vul_coefs(m, years)
            all_coefs[spec_name] = coefs

            # Pre-trend summary: avg absolute coef in pre-period
            pre = coefs[coefs["year"] < BASE_YEAR]
            avg_pre = pre["coef"].abs().mean()
            max_pre = pre["coef"].abs().max()
            # How many pre-period CIs include zero?
            pre_zero = ((pre["ci_lo"] <= 0) & (pre["ci_hi"] >= 0)).sum()
            print(f"  {spec_name:20s}  R2={m._r2:.4f}  "
                  f"pre|avg|={avg_pre:.4f}  pre|max|={max_pre:.4f}  "
                  f"pre CIs incl 0: {pre_zero}/{len(pre)}")

        # Plot: baseline vs kitchen sink
        fig, ax = plt.subplots(figsize=(12, 6))
        yrs = all_coefs["1. Baseline"]["year"].values

        colors = ["#cb181d", "#2171b5", "#238b45", "#6a3d9a", "#ff7f00", "#000000"]
        markers = ["o", "s", "^", "D", "v", "P"]
        offsets = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]

        for i, (spec_name, coefs) in enumerate(all_coefs.items()):
            ax.errorbar(yrs + offsets[i], coefs["coef"],
                        yerr=[coefs["coef"] - coefs["ci_lo"],
                              coefs["ci_hi"] - coefs["coef"]],
                        fmt=markers[i], color=colors[i], markersize=3.5, capsize=1.5,
                        linewidth=0.8, label=spec_name, alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Coefficient on Vulnerability x Year", fontsize=12)
        ax.set_title(f"{dep_label}: Pre-trend Robustness", fontsize=13)
        ax.legend(fontsize=8, framealpha=0.9, ncol=2)
        ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
        ax.set_xticklabels([str(yr) for yr in yrs if yr % 2 == 1 or yr == yrs[0]], fontsize=10)
        ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        suffix = "net_slant" if "econ" not in depvar else "net_slant_econ"
        out = FIG_DIR / f"pretrend_controls_{suffix}.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
