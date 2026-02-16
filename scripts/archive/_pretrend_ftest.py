"""
Compute joint F-test for pre-trend coefficients = 0
in the controlled event study specification.

Tests H0: vul_1987 = vul_1988 = ... = vul_1992 = 0
for key outcomes.
"""

import os, sys
import numpy as np
import pandas as pd
import pyfixest as pf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
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

def main():
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["ext_net"] = df["ext_R"] - df["ext_D"]

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
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)

    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    rhs = " + ".join(vul_vars + china_vars + manu_vars)

    # Pre-period vul variables (before BASE_YEAR=1993)
    pre_vul_vars = [f"vul_{yr}" for yr in years if yr < BASE_YEAR]
    print(f"Pre-period variables: {pre_vul_vars}")

    outcomes = [
        ("net_slant_norm", "Net Slant (Normalized)"),
        ("ext_R", "Share R-Leaning"),
        ("ext_D", "Share D-Leaning"),
        ("ext_net", "Net Extensive Margin"),
        ("right_norm", "R Component (Unconditional)"),
        ("left_norm", "L Component (Unconditional)"),
        ("int_right_norm", "R Component (Intensive)"),
        ("int_left_norm", "L Component (Intensive)"),
    ]

    print(f"\n{'Outcome':<30} {'F-stat':>10} {'p-value':>10} {'df_num':>8} {'df_den':>8}")
    print("-" * 70)

    for depvar, label in outcomes:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})

        # Build R matrix: test only pre-period vul coefficients = 0
        coef_names = list(m._coefnames)
        n_coefs = len(coef_names)
        pre_indices = [coef_names.index(v) for v in pre_vul_vars if v in coef_names]

        R = np.zeros((len(pre_indices), n_coefs))
        for i, idx in enumerate(pre_indices):
            R[i, idx] = 1.0
        q = np.zeros(len(pre_indices))

        result = m.wald_test(R=R, q=q, distribution="F")
        f_stat = m._f_statistic
        p_val = m._p_value

        print(f"{label:<30} {f_stat:>10.3f} {p_val:>10.4f} {m._dfn:>8} {m._dfd:>8}")

    # Also test baseline spec (without controls) for comparison
    print("\n\n--- BASELINE SPEC (no controls) ---")
    vul_rhs = " + ".join(vul_vars)

    print(f"\n{'Outcome':<30} {'F-stat':>10} {'p-value':>10} {'df_num':>8} {'df_den':>8}")
    print("-" * 70)

    for depvar, label in outcomes:
        fml = f"{depvar} ~ {vul_rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})

        coef_names = list(m._coefnames)
        n_coefs = len(coef_names)
        pre_indices = [coef_names.index(v) for v in pre_vul_vars if v in coef_names]

        R = np.zeros((len(pre_indices), n_coefs))
        for i, idx in enumerate(pre_indices):
            R[i, idx] = 1.0
        q = np.zeros(len(pre_indices))

        result = m.wald_test(R=R, q=q, distribution="F")
        f_stat = m._f_statistic
        p_val = m._p_value

        print(f"{label:<30} {f_stat:>10.3f} {p_val:>10.4f} {m._dfn:>8} {m._dfd:>8}")


if __name__ == "__main__":
    main()
