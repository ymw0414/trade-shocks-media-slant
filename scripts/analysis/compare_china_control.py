"""
Compare event study results:
  Spec A: china_shock × ALL years (current approach)
  Spec B: china_shock × post-2000 years only

For net_slant_norm (main outcome).
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

    # Vulnerability × year (omit BASE_YEAR=1993)
    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]

    # China shock × ALL years (omit first year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    # China shock × post-2000 only
    for yr in years:
        if yr <= 2000:
            continue
        df[f"chinapost_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    # Manushare × ALL years (omit first year)
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

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
    print(f"  {len(df):,} obs, {df['paper_id'].nunique()} papers, {df['cz'].nunique()} CZs")

    base_yr = years[0]
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    vul_rhs = " + ".join(vul_vars)

    china_all_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    china_post_vars = [f"chinapost_{yr}" for yr in years if yr > 2000]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]

    depvars = [
        ("net_slant_norm", "Net Slant (Normalized)"),
        ("net_slant_norm_econ", "Net Slant — Econ Articles (Normalized)"),
    ]

    for depvar, label in depvars:
        print(f"\n=== {label} ===")

        # Spec A: china × all years
        rhs_a = f"{vul_rhs} + {' + '.join(china_all_vars + manu_vars)}"
        fml_a = f"{depvar} ~ {rhs_a} | paper_id + year + division^year"
        m_a = pf.feols(fml_a, data=df, vcov={"CRV1": "cz"})
        coefs_a = extract_vul_coefs(m_a, years)
        print(f"  China x all years:  N={m_a._N}, R2={m_a._r2:.4f}")

        # Spec B: china × post-2000 only
        rhs_b = f"{vul_rhs} + {' + '.join(china_post_vars + manu_vars)}"
        fml_b = f"{depvar} ~ {rhs_b} | paper_id + year + division^year"
        m_b = pf.feols(fml_b, data=df, vcov={"CRV1": "cz"})
        coefs_b = extract_vul_coefs(m_b, years)
        print(f"  China x post-2000:  N={m_b._N}, R2={m_b._r2:.4f}")

        # Print comparison
        comp = coefs_a[["year", "coef"]].rename(columns={"coef": "all_years"}).merge(
            coefs_b[["year", "coef"]].rename(columns={"coef": "post2000"}), on="year")
        comp["diff"] = comp["all_years"] - comp["post2000"]
        print(f"\n  Year-by-year vulnerability coefficients:")
        print(comp.to_string(index=False, float_format="%.4f"))

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        yrs = coefs_a["year"].values
        offset = 0.15

        ax.errorbar(yrs - offset, coefs_a["coef"],
                    yerr=[coefs_a["coef"] - coefs_a["ci_lo"],
                          coefs_a["ci_hi"] - coefs_a["coef"]],
                    fmt="o", color="#cb181d", markersize=4, capsize=2.5,
                    linewidth=1.0, label="China x all years")

        ax.errorbar(yrs + offset, coefs_b["coef"],
                    yerr=[coefs_b["coef"] - coefs_b["ci_lo"],
                          coefs_b["ci_hi"] - coefs_b["coef"]],
                    fmt="s", color="#2171b5", markersize=4, capsize=2.5,
                    linewidth=1.0, label="China x post-2000 only")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.7,
                   label="NAFTA (1994)")
        ax.axvline(2000.5, color="orange", linewidth=0.8, linestyle=":", alpha=0.7,
                   label="China WTO (2001)")

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Coefficient on Vulnerability x Year", fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
        ax.set_xticklabels([str(yr) for yr in yrs if yr % 2 == 1 or yr == yrs[0]], fontsize=10)
        ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        suffix = depvar.replace("_norm", "").replace("_econ", "_econ")
        out_path = FIG_DIR / f"compare_china_control_{suffix}.png"
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
