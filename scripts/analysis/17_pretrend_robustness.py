"""
17_pretrend_robustness.py

Pre-trend robustness checks for the event study (main model: exp_shvocab_cv).

Tests:
  1. Joint F-test: H0 = all pre-NAFTA vulnerability × year coefficients = 0
  2. Linear trend adjustment: add vulnerability × linear_time to absorb pre-trend
  3. Alternative base years: 1990 and 1991 (anticipation)
  4. Visual: extrapolate pre-trend line onto post-NAFTA period

Outputs:
  - reports/figures/pretrend_*.png
  - Printed F-test results
"""

import os
import sys
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

DEPVARS = [
    ("net_slant_norm", "Net Slant (Norm.)"),
    ("share_R", "Share R-Leaning"),
    ("share_D", "Share D-Leaning"),
]


def load_panel():
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes

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

    # Linear time trend interaction
    df["time_trend"] = df["year"] - BASE_YEAR
    df["vul_trend"] = df["vulnerability1990_scaled"] * df["time_trend"]

    return df, years


def extract_coefs(model, years, base_year):
    """Extract event study coefficients from a fitted model."""
    t = model.tidy()
    rows = []
    for yr in years:
        if yr == base_year:
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
    return pd.DataFrame(rows)


# ── Test 1: Joint F-test for pre-trends ─────────────────────────────

def test_pretrends(df, years):
    """Joint F-test: H0 = all pre-NAFTA vul×year coefficients = 0."""
    print("\n" + "=" * 70)
    print("TEST 1: Joint F-test for pre-NAFTA coefficients = 0")
    print("=" * 70)

    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    pre_vars = [f"vul_{yr}" for yr in years if yr < BASE_YEAR]
    base_yr = years[0]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    rhs = " + ".join(vul_vars + china_vars + manu_vars)

    results = []
    for depvar, label in DEPVARS:
        fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy()

        # Extract pre-NAFTA coefficients and their variance-covariance
        pre_coefs = []
        pre_ses = []
        for v in pre_vars:
            if v in t.index:
                pre_coefs.append(t.loc[v, "Estimate"])
                pre_ses.append(t.loc[v, "Std. Error"])

        n_pre = len(pre_coefs)
        if n_pre == 0:
            continue

        # Wald test: sum(beta_i^2 / se_i^2) ~ chi2(n_pre) as approximation
        # More properly: use the vcov matrix from pyfixest
        try:
            vcov = m._vcov
            param_names = m._coefnames
            pre_idx = [i for i, name in enumerate(param_names) if name in pre_vars]
            beta_pre = np.array([m.coef().iloc[i] for i in pre_idx])
            V_pre = vcov[np.ix_(pre_idx, pre_idx)]
            # Wald statistic: beta' V^{-1} beta / k ~ F(k, G-1) where G = clusters
            wald = beta_pre @ np.linalg.solve(V_pre, beta_pre) / n_pre
            n_clusters = df["cz"].nunique()
            p_val = 1 - stats.f.cdf(wald, n_pre, n_clusters - 1)
        except Exception as e:
            print(f"  {label}: vcov extraction failed ({e}), using naive test")
            wald = sum((c / s) ** 2 for c, s in zip(pre_coefs, pre_ses)) / n_pre
            p_val = 1 - stats.f.cdf(wald, n_pre, n_pre)

        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        results.append({"depvar": label, "F_stat": wald, "df1": n_pre,
                        "df2": n_clusters - 1, "p_value": p_val, "sig": sig})
        print(f"  {label:>25s}: F({n_pre},{n_clusters-1}) = {wald:.3f}, p = {p_val:.4f} {sig}")

    return pd.DataFrame(results)


# ── Test 2: Detrended event study ───────────────────────────────────

def trend_adjusted_es(df, years):
    """Detrend event study: fit linear trend to pre-NAFTA coefficients, subtract."""
    print("\n" + "=" * 70)
    print("TEST 2: Detrended event study (subtract pre-trend line)")
    print("=" * 70)

    # Load original controlled-spec coefficients
    tab_dir = cfg.TAB_DIR
    es_df = pd.read_csv(tab_dir / "event_study_coefficients.csv")

    all_coefs = {}
    for depvar, label in DEPVARS:
        ctrl = es_df[(es_df["depvar"] == depvar) & (es_df["spec"] == "controls")].copy()
        ctrl = ctrl.sort_values("year").reset_index(drop=True)

        # Fit linear trend through pre-NAFTA coefficients (including base year)
        pre = ctrl[ctrl["year"] <= BASE_YEAR]
        x_pre = pre["year"].values.astype(float)
        y_pre = pre["coef"].values
        slope, intercept = np.polyfit(x_pre, y_pre, 1)

        # Detrend: subtract fitted trend from all coefficients
        detrended = ctrl.copy()
        trend_vals = slope * ctrl["year"].values + intercept
        detrended["coef"] = ctrl["coef"] - trend_vals
        detrended["ci_lo"] = ctrl["ci_lo"] - trend_vals
        detrended["ci_hi"] = ctrl["ci_hi"] - trend_vals

        sig = "***" if abs(slope) > 0.02 else ""
        print(f"  {label:>25s}: pre-trend slope = {slope:.5f}/yr, "
              f"detrended post-mean = {detrended[detrended['year'] > BASE_YEAR]['coef'].mean():.4f}")

        all_coefs[depvar] = {"original": ctrl, "detrended": detrended, "slope": slope}
        plot_trend_comparison(ctrl, detrended, label, depvar, slope)

    return all_coefs


def plot_trend_comparison(coefs_orig, coefs_detrended, label, depvar, slope):
    """Plot original vs detrended event study."""
    fig, ax = plt.subplots(figsize=(12, 6))
    yrs = coefs_orig["year"].values
    offset = 0.15

    ax.errorbar(yrs - offset, coefs_orig["coef"],
                yerr=[coefs_orig["coef"] - coefs_orig["ci_lo"],
                      coefs_orig["ci_hi"] - coefs_orig["coef"]],
                fmt="o", color="#2d2d2d", markersize=4, capsize=2.5,
                linewidth=1.0, label="Original", alpha=0.5)

    ax.errorbar(yrs + offset, coefs_detrended["coef"],
                yerr=[coefs_detrended["coef"] - coefs_detrended["ci_lo"],
                      coefs_detrended["ci_hi"] - coefs_detrended["coef"]],
                fmt="s", color="#7a7a7a", markersize=4, capsize=2.5,
                linewidth=1.0, label=f"Detrended (slope={slope:.4f}/yr)")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7, label="NAFTA (1994)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title(f"{label}: Original vs. Detrended", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / f"pretrend_trend_adj_{depvar}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"    Saved: {out.name}")


# ── Test 3: Alternative base years ──────────────────────────────────

def alternative_base_years(df, years):
    """Re-run event study with base year = 1990 and 1991."""
    print("\n" + "=" * 70)
    print("TEST 3: Alternative base years (anticipation)")
    print("=" * 70)

    for alt_base in [1990, 1991]:
        print(f"\n  --- Base year = {alt_base} ---")
        vul_vars = [f"vul_{yr}" for yr in years if yr != alt_base]
        base_yr = years[0]
        china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
        manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]

        # Need to create vul vars relative to alt_base
        # The existing vul_1993 still exists but now we INCLUDE it; we EXCLUDE vul_{alt_base}
        # Actually, we need to rebuild: vul_{yr} for yr != alt_base
        # The existing vul_ columns omit 1993. We need vul_1993 and exclude vul_{alt_base}.
        for yr in years:
            col = f"vul_alt_{yr}"
            if yr == alt_base:
                continue
            df[col] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]

        alt_vul_vars = [f"vul_alt_{yr}" for yr in years if yr != alt_base]
        rhs = " + ".join(alt_vul_vars + china_vars + manu_vars)

        for depvar, label in DEPVARS:
            fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
            m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
            coefs = extract_coefs_alt(m, years, alt_base)

            # Plot
            plot_alt_base(coefs, label, depvar, alt_base)

        # Cleanup
        for yr in years:
            col = f"vul_alt_{yr}"
            if col in df.columns:
                df.drop(columns=[col], inplace=True)


def extract_coefs_alt(model, years, base_year):
    t = model.tidy()
    rows = []
    for yr in years:
        if yr == base_year:
            rows.append({"year": yr, "coef": 0, "se": 0, "ci_lo": 0, "ci_hi": 0})
            continue
        vname = f"vul_alt_{yr}"
        if vname in t.index:
            r = t.loc[vname]
            rows.append({"year": yr, "coef": r["Estimate"], "se": r["Std. Error"],
                         "ci_lo": r["2.5%"], "ci_hi": r["97.5%"]})
        else:
            rows.append({"year": yr, "coef": np.nan, "se": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan})
    return pd.DataFrame(rows)


def plot_alt_base(coefs, label, depvar, alt_base):
    fig, ax = plt.subplots(figsize=(12, 6))
    yrs = coefs["year"].values

    color = "#2d2d2d"
    ax.errorbar(yrs, coefs["coef"],
                yerr=[coefs["coef"] - coefs["ci_lo"],
                      coefs["ci_hi"] - coefs["coef"]],
                fmt="o", color=color, markersize=5, capsize=3, linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(alt_base + 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.7, label=f"Base = {alt_base}")
    ax.axvline(1993.5, color="orange", linewidth=0.8,
               linestyle=":", alpha=0.7, label="NAFTA (1994)")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
    ax.set_title(f"{label} (base = {alt_base})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
    ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.3)
    out = FIG_DIR / f"pretrend_base{alt_base}_{depvar}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"    {label}: saved {out.name}")


# ── Test 4: Pre-trend extrapolation overlay ─────────────────────────

def extrapolation_overlay(df, years):
    """Fit linear line through pre-NAFTA coefficients, overlay on full plot."""
    print("\n" + "=" * 70)
    print("TEST 4: Pre-trend extrapolation overlay")
    print("=" * 70)

    # Load original controlled-spec coefficients
    tab_dir = cfg.TAB_DIR
    es_df = pd.read_csv(tab_dir / "event_study_coefficients.csv")

    for depvar, label in DEPVARS:
        ctrl = es_df[(es_df["depvar"] == depvar) & (es_df["spec"] == "controls")].copy()
        ctrl = ctrl.sort_values("year")

        # Pre-NAFTA points (excluding base year which is 0 by construction)
        pre = ctrl[(ctrl["year"] < BASE_YEAR)].copy()
        # Include base year in the fit
        pre_with_base = ctrl[ctrl["year"] <= BASE_YEAR].copy()

        # Fit linear trend through pre-NAFTA coefficients
        x_pre = pre_with_base["year"].values.astype(float)
        y_pre = pre_with_base["coef"].values
        if len(x_pre) < 2:
            continue

        slope, intercept = np.polyfit(x_pre, y_pre, 1)
        print(f"  {label:>25s}: pre-trend slope = {slope:.5f}/year")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        yrs = ctrl["year"].values

        ax.errorbar(yrs, ctrl["coef"],
                    yerr=[ctrl["coef"] - ctrl["ci_lo"],
                          ctrl["ci_hi"] - ctrl["coef"]],
                    fmt="o", color="#2d2d2d", markersize=5, capsize=3,
                    linewidth=1.2, label="Event study (with controls)")

        # Extrapolated trend line
        all_yrs = np.arange(years[0], years[-1] + 1)
        trend_line = slope * all_yrs + intercept
        ax.plot(all_yrs, trend_line, color="#bf6b63", linewidth=2,
                linestyle="--", alpha=0.8, label=f"Pre-trend line (slope={slope:.4f}/yr)")

        # Shade pre vs post
        ax.axvspan(years[0] - 1, BASE_YEAR + 0.5, alpha=0.05, color="#e8e8e8")
        ax.axvspan(BASE_YEAR + 0.5, years[-1] + 1, alpha=0.05, color="#e8e8e8")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(BASE_YEAR + 0.5, color="gray", linewidth=0.8,
                   linestyle="--", alpha=0.7)

        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Coefficient on Vulnerability \u00d7 Year", fontsize=12)
        ax.set_title(f"{label}: Pre-trend Extrapolation", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_xticks([yr for yr in yrs if yr % 2 == 1 or yr == yrs[0]])
        ax.set_xlim(yrs[0] - 0.7, yrs[-1] + 0.7)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout(pad=0.3)
        out = FIG_DIR / f"pretrend_extrap_{depvar}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"    Saved: {out.name}")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading panel ...")
    df, years = load_panel()
    print(f"  {len(df):,} obs, years {years[0]}-{years[-1]}")

    ftest_results = test_pretrends(df, years)
    trend_coefs = trend_adjusted_es(df, years)
    alternative_base_years(df, years)
    extrapolation_overlay(df, years)

    print("\n" + "=" * 70)
    print("All pre-trend robustness checks complete.")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 70)
