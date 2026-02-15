"""
Short/Long DiD split: adds column (4) to DiD table.
Single regression with Vul×Short (1994-1998) + Vul×Long (1999-2004),
full controls (manushare×year + china shock×year).
"""

import os, sys, json
import numpy as np
import pandas as pd
import pyfixest as pf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
NAFTA_YEAR = 1994
SHORT_END = 1998  # Short: 1994-1998, Long: 1999-2004
END_YEAR = 2004

# State FIPS -> Census Division mapping
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


def main():
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes

    # Short and Long post indicators
    df["short_post"] = ((df["year"] >= NAFTA_YEAR) & (df["year"] <= SHORT_END)).astype(int)
    df["long_post"] = (df["year"] > SHORT_END).astype(int)
    df["vuln_x_short"] = df["vulnerability1990_scaled"] * df["short_post"]
    df["vuln_x_long"] = df["vulnerability1990_scaled"] * df["long_post"]

    years = sorted(df["year"].unique())
    base_yr = years[0]

    # Build control variable interactions
    manu_vars = []
    china_vars = []
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
        manu_vars.append(f"manu_{yr}")
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)
        china_vars.append(f"china_{yr}")

    manu_rhs = " + ".join(manu_vars)
    china_rhs = " + ".join(china_vars)

    # Outcomes matching DiD table
    outcomes = [
        # Panel A: Aggregate
        ("net_slant_norm",   "Net slant"),
        ("right_norm",       "Republican slant (R~)"),
        ("left_norm",        "Democratic slant (L~)"),
        # Panel B: Extensive
        ("ext_R",            "Share R-leaning"),
        ("ext_D",            "Share D-leaning"),
        # Panel C: Intensive
        ("int_right_norm",   "Republican slant (R~, intensive)"),
        ("int_left_norm",    "Democratic slant (L~, intensive)"),
    ]

    results = {}
    for depvar, label in outcomes:
        fml = f"{depvar} ~ vuln_x_short + vuln_x_long + {china_rhs} + {manu_rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        tidy = m.tidy()

        short_row = tidy.loc["vuln_x_short"]
        long_row = tidy.loc["vuln_x_long"]

        # Wald test: short == long
        try:
            wald = m.wald_test(R="vuln_x_short = vuln_x_long")
            wald_p = wald.pvalue
        except Exception:
            wald_p = np.nan

        results[depvar] = {
            "label": label,
            "short_coef": short_row["Estimate"],
            "short_se": short_row["Std. Error"],
            "short_pval": short_row["Pr(>|t|)"],
            "long_coef": long_row["Estimate"],
            "long_se": long_row["Std. Error"],
            "long_pval": long_row["Pr(>|t|)"],
            "wald_p": wald_p,
            "N": int(m._N),
        }

        stars_s = "***" if short_row["Pr(>|t|)"] < 0.01 else "**" if short_row["Pr(>|t|)"] < 0.05 else "*" if short_row["Pr(>|t|)"] < 0.1 else ""
        stars_l = "***" if long_row["Pr(>|t|)"] < 0.01 else "**" if long_row["Pr(>|t|)"] < 0.05 else "*" if long_row["Pr(>|t|)"] < 0.1 else ""

        print(f"\n{label} ({depvar}):")
        print(f"  Short (94-98): {short_row['Estimate']:.4f}{stars_s} (SE={short_row['Std. Error']:.4f})")
        print(f"  Long  (99-04): {long_row['Estimate']:.4f}{stars_l} (SE={long_row['Std. Error']:.4f})")
        if not np.isnan(wald_p):
            print(f"  Wald p(Short=Long): {wald_p:.4f}")

    # Save results as JSON for easy parsing
    output_path = BASE_DIR / "output" / "tables" / "did_shortlong.json"
    # Convert numpy types for JSON serialization
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
