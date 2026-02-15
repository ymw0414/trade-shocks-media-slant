"""
HonestDiD sensitivity analysis (Rambachan & Roth 2023).

Implements both:
  1. Smoothness restriction (DeltaSD): |delta_{t+1} - 2*delta_t + delta_{t-1}| <= Mbar
     - Conditions on observed pre-treatment coefficients (plug-in approach)
     - LP over post-treatment delta only
  2. Relative magnitudes (DeltaRM): max post |first diff| <= Mbar * max pre |first diff|

For each Mbar value, computes worst-case bias for the average post-treatment effect
and constructs robust confidence intervals.

Outputs:
  - output/figures/honestdid_sensitivity.pdf  (combined 2-panel figure)
  - output/tables/honestdid_sensitivity.csv
"""

import os, sys
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

BASE_YEAR = 1993
END_YEAR = 2004
Z_CRIT = 1.96
N_GRID = 80

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
    for yr in years:
        if yr == BASE_YEAR:
            continue
        df[f"vul_{yr}"] = (df["year"] == yr).astype(float) * df["vulnerability1990_scaled"]
    for yr in years:
        if yr == base_yr:
            continue
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
    return df, years


def run_event_study(df, depvar, years):
    """Run controlled event study, return vuln coefs, vcov, year list."""
    base_yr = years[0]
    vul_vars = [f"vul_{yr}" for yr in years if yr != BASE_YEAR]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    rhs = " + ".join(vul_vars + china_vars + manu_vars)
    fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
    m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
    all_names = [str(c) for c in m._coefnames]
    vul_idx = [i for i, nm in enumerate(all_names) if nm.startswith("vul_")]
    beta_vul = m._beta_hat[vul_idx]
    vcov_vul = m._vcov[np.ix_(vul_idx, vul_idx)]
    vul_years = [int(all_names[i].replace("vul_", "")) for i in vul_idx]
    return beta_vul, vcov_vul, vul_years


def _get_pre_post_betas(beta_vul, vul_years):
    """Organize betas into full vector with 0 at base year."""
    all_years = sorted(set(vul_years) | {BASE_YEAR})
    beta_full = {}
    for yr in all_years:
        if yr == BASE_YEAR:
            beta_full[yr] = 0.0
        else:
            j = vul_years.index(yr)
            beta_full[yr] = beta_vul[j]

    pre_years = sorted([yr for yr in all_years if yr < BASE_YEAR])
    post_years = sorted([yr for yr in all_years if yr > BASE_YEAR])
    return beta_full, all_years, pre_years, post_years


# ---------------------------------------------------------------------------
# Smoothness restriction (DeltaSD) — conditional on pre-treatment estimates
# ---------------------------------------------------------------------------

def smoothness_sensitivity(beta_vul, vcov_vul, vul_years):
    """
    Conditional smoothness approach.

    Fix delta_pre = beta_hat_pre, delta_base = 0.
    LP over delta_post subject to second-difference constraints.
    """
    beta_full, all_years, pre_years, post_years = _get_pre_post_betas(beta_vul, vul_years)
    n_post = len(post_years)

    # Compute pre-treatment second differences to find Mbar_min
    pre_second_diffs = []
    all_sorted = sorted(all_years)
    for i in range(1, len(all_sorted) - 1):
        yr_prev, yr_cur, yr_next = all_sorted[i-1], all_sorted[i], all_sorted[i+1]
        # Only include second differences that are fully determined by pre + base
        if yr_next <= BASE_YEAR or (yr_cur <= BASE_YEAR and yr_next <= post_years[0]):
            sd = beta_full[yr_next] - 2 * beta_full[yr_cur] + beta_full[yr_prev]
            pre_second_diffs.append(abs(sd))

    # Include the boundary second difference at t = last_pre
    last_pre = pre_years[-1]
    # At t = last_pre: |delta_base - 2*delta_{last_pre} + delta_{last_pre-1}|
    if len(pre_years) >= 2:
        sd_boundary = abs(beta_full[BASE_YEAR] - 2 * beta_full[last_pre] + beta_full[pre_years[-2]])
        pre_second_diffs.append(sd_boundary)

    Mbar_min = max(pre_second_diffs) if pre_second_diffs else 0.0
    print(f"    Mbar_min (pre-trend curvature): {Mbar_min:.6f}")

    # SE of average post-treatment effect
    n_post_vul = sum(1 for yr in vul_years if yr > BASE_YEAR)
    l_vul = np.array([1.0 / n_post_vul if yr > BASE_YEAR else 0.0 for yr in vul_years])
    se_theta = np.sqrt(l_vul @ vcov_vul @ l_vul)
    theta_hat = np.mean([beta_full[yr] for yr in post_years])

    # Grid: from 0 to ~3x Mbar_min (or enough to find breakdown)
    # Start from 0 — at Mbar < Mbar_min, LP is infeasible (which is informative)
    mbar_max = max(Mbar_min * 4, 0.01)
    mbar_grid = np.linspace(0, mbar_max, N_GRID)

    # Ensure Mbar_min is in the grid
    mbar_grid = np.sort(np.unique(np.append(mbar_grid, [Mbar_min, Mbar_min * 1.001])))

    results = []
    for Mbar in mbar_grid:
        max_bias = _solve_smoothness_lp(beta_full, pre_years, post_years, Mbar, "max")
        min_bias = _solve_smoothness_lp(beta_full, pre_years, post_years, Mbar, "min")

        if np.isnan(max_bias) or np.isnan(min_bias):
            # Infeasible (Mbar too small for pre-trend curvature)
            results.append({
                "Mbar": Mbar, "theta_hat": theta_hat, "se_theta": se_theta,
                "max_bias": np.nan, "min_bias": np.nan,
                "robust_ci_lo": np.nan, "robust_ci_hi": np.nan,
                "covers_zero": np.nan, "feasible": False,
            })
            continue

        robust_lo = theta_hat - max_bias - Z_CRIT * se_theta
        robust_hi = theta_hat - min_bias + Z_CRIT * se_theta
        covers_zero = (robust_lo <= 0 <= robust_hi)

        results.append({
            "Mbar": Mbar, "theta_hat": theta_hat, "se_theta": se_theta,
            "max_bias": max_bias, "min_bias": min_bias,
            "robust_ci_lo": robust_lo, "robust_ci_hi": robust_hi,
            "covers_zero": covers_zero, "feasible": True,
        })

    df_res = pd.DataFrame(results)

    # If breakdown not found, expand grid
    feasible = df_res[df_res["feasible"]]
    if len(feasible) > 0 and not feasible["covers_zero"].any():
        # Need larger Mbar — estimate from linear extrapolation
        last_lo = feasible["robust_ci_lo"].iloc[-1]
        first_lo = feasible["robust_ci_lo"].dropna().iloc[0]
        last_mbar = feasible["Mbar"].iloc[-1]
        first_mbar = feasible["Mbar"].dropna().iloc[0]
        if last_lo != first_lo:
            slope = (last_lo - first_lo) / (last_mbar - first_mbar + 1e-10)
            needed_mbar = last_mbar + (0 - last_lo) / (slope - 1e-10) if slope < 0 else last_mbar * 3
            needed_mbar = min(needed_mbar * 1.5, last_mbar * 5)
        else:
            needed_mbar = last_mbar * 3

        extra_grid = np.linspace(mbar_max, needed_mbar, N_GRID // 2)
        extra_results = []
        for Mbar in extra_grid:
            max_bias = _solve_smoothness_lp(beta_full, pre_years, post_years, Mbar, "max")
            min_bias = _solve_smoothness_lp(beta_full, pre_years, post_years, Mbar, "min")
            if np.isnan(max_bias):
                continue
            robust_lo = theta_hat - max_bias - Z_CRIT * se_theta
            robust_hi = theta_hat - min_bias + Z_CRIT * se_theta
            covers_zero = (robust_lo <= 0 <= robust_hi)
            extra_results.append({
                "Mbar": Mbar, "theta_hat": theta_hat, "se_theta": se_theta,
                "max_bias": max_bias, "min_bias": min_bias,
                "robust_ci_lo": robust_lo, "robust_ci_hi": robust_hi,
                "covers_zero": covers_zero, "feasible": True,
            })
        if extra_results:
            df_res = pd.concat([df_res, pd.DataFrame(extra_results)], ignore_index=True)
            df_res = df_res.sort_values("Mbar").reset_index(drop=True)

    return df_res, Mbar_min


def _solve_smoothness_lp(beta_full, pre_years, post_years, Mbar, direction):
    """
    Solve LP for worst-case post-treatment bias under smoothness.

    Decision variables: x[0..n_post-1] = delta for each post-treatment year.
    Fixed: delta_pre = beta_full[pre years], delta_base = 0.
    """
    n_post = len(post_years)
    last_pre_beta = beta_full[pre_years[-1]]

    # Build inequality constraints
    A_rows = []
    b_vals = []

    # C1: Boundary at t = BASE_YEAR
    # |x[0] - 2*0 + beta_{last_pre}| <= Mbar
    # x[0] + last_pre_beta <= Mbar  AND  -x[0] - last_pre_beta <= Mbar
    row = np.zeros(n_post)
    row[0] = 1.0
    A_rows.append(row.copy())
    b_vals.append(Mbar - last_pre_beta)

    row = np.zeros(n_post)
    row[0] = -1.0
    A_rows.append(row.copy())
    b_vals.append(Mbar + last_pre_beta)

    # C2: At t = post_years[0] (1994)
    # |x[1] - 2*x[0] + 0| <= Mbar  (delta_base = 0)
    if n_post >= 2:
        row = np.zeros(n_post)
        row[1] = 1.0
        row[0] = -2.0
        A_rows.append(row.copy())
        b_vals.append(Mbar)

        row = np.zeros(n_post)
        row[1] = -1.0
        row[0] = 2.0
        A_rows.append(row.copy())
        b_vals.append(Mbar)

    # C3+: Post-treatment second differences (t = post_years[1], ..., post_years[-2])
    for k in range(2, n_post):
        # |x[k] - 2*x[k-1] + x[k-2]| <= Mbar
        row = np.zeros(n_post)
        row[k] = 1.0
        row[k-1] = -2.0
        row[k-2] = 1.0
        A_rows.append(row.copy())
        b_vals.append(Mbar)

        row = np.zeros(n_post)
        row[k] = -1.0
        row[k-1] = 2.0
        row[k-2] = -1.0
        A_rows.append(row.copy())
        b_vals.append(Mbar)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)

    # Objective: average post-treatment delta
    c = np.full(n_post, 1.0 / n_post)
    if direction == "max":
        c_obj = -c  # negate for maximization
    else:
        c_obj = c

    bounds = [(None, None)] * n_post

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        return -res.fun if direction == "max" else res.fun
    else:
        return np.nan


# ---------------------------------------------------------------------------
# Relative magnitudes (DeltaRM) — conditional on pre-treatment estimates
# ---------------------------------------------------------------------------

def relative_magnitudes_sensitivity(beta_vul, vcov_vul, vul_years):
    """
    Relative magnitudes approach.

    max post |delta_t - delta_{t-1}| <= Mbar * max_pre_first_diff
    Fix delta_pre = beta_hat_pre, delta_base = 0.
    """
    beta_full, all_years, pre_years, post_years = _get_pre_post_betas(beta_vul, vul_years)
    n_post = len(post_years)

    # Compute max pre-treatment first difference
    pre_first_diffs = []
    for i in range(1, len(pre_years)):
        fd = abs(beta_full[pre_years[i]] - beta_full[pre_years[i-1]])
        pre_first_diffs.append(fd)
    # Include base year boundary: |delta_base - delta_{last_pre}| = |0 - beta_{last_pre}|
    pre_first_diffs.append(abs(beta_full[pre_years[-1]]))
    max_pre_fd = max(pre_first_diffs) if pre_first_diffs else 0.001
    print(f"    Max pre-treatment first diff: {max_pre_fd:.6f}")

    # SE and theta
    n_post_vul = sum(1 for yr in vul_years if yr > BASE_YEAR)
    l_vul = np.array([1.0 / n_post_vul if yr > BASE_YEAR else 0.0 for yr in vul_years])
    se_theta = np.sqrt(l_vul @ vcov_vul @ l_vul)
    theta_hat = np.mean([beta_full[yr] for yr in post_years])

    # Grid from 0 to ~5
    mbar_grid = np.linspace(0, 5.0, N_GRID)

    results = []
    for Mbar in mbar_grid:
        bound = Mbar * max_pre_fd
        max_bias = _solve_relmag_lp(n_post, bound, "max")
        min_bias = _solve_relmag_lp(n_post, bound, "min")

        robust_lo = theta_hat - max_bias - Z_CRIT * se_theta
        robust_hi = theta_hat - min_bias + Z_CRIT * se_theta
        covers_zero = (robust_lo <= 0 <= robust_hi)

        results.append({
            "Mbar": Mbar, "theta_hat": theta_hat, "se_theta": se_theta,
            "max_bias": max_bias, "min_bias": min_bias,
            "robust_ci_lo": robust_lo, "robust_ci_hi": robust_hi,
            "covers_zero": covers_zero, "feasible": True,
            "max_pre_fd": max_pre_fd,
        })

    return pd.DataFrame(results), max_pre_fd


def _solve_relmag_lp(n_post, bound, direction):
    """
    Solve LP for relative magnitudes.

    Decision variables: x[0..n_post-1] = post-treatment delta.
    x[-1] := 0 (base year).
    |x[k] - x[k-1]| <= bound for k=0,...,n_post-1 (x[-1] = 0)
    """
    A_rows = []
    b_vals = []

    # x[0] - 0 <= bound  AND  -(x[0]) <= bound
    row = np.zeros(n_post)
    row[0] = 1.0
    A_rows.append(row.copy())
    b_vals.append(bound)

    row = np.zeros(n_post)
    row[0] = -1.0
    A_rows.append(row.copy())
    b_vals.append(bound)

    for k in range(1, n_post):
        row = np.zeros(n_post)
        row[k] = 1.0
        row[k-1] = -1.0
        A_rows.append(row.copy())
        b_vals.append(bound)

        row = np.zeros(n_post)
        row[k] = -1.0
        row[k-1] = 1.0
        A_rows.append(row.copy())
        b_vals.append(bound)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)

    c = np.full(n_post, 1.0 / n_post)
    if direction == "max":
        c_obj = -c
    else:
        c_obj = c

    bounds = [(None, None)] * n_post
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.success:
        return -res.fun if direction == "max" else res.fun
    else:
        return np.nan


# ---------------------------------------------------------------------------
# Breakdown value
# ---------------------------------------------------------------------------

def find_breakdown(sens_df):
    """Smallest Mbar where the feasible robust CI first covers zero."""
    feasible = sens_df[sens_df["feasible"] == True]
    if len(feasible) == 0:
        return np.inf
    covers = feasible[feasible["covers_zero"] == True]
    if len(covers) == 0:
        return np.inf
    return covers["Mbar"].iloc[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel ...")
    df, years = load_panel()
    df["ext_net"] = df["ext_R"] - df["ext_D"]
    print(f"  {len(df):,} obs, {df['paper_id'].nunique()} papers, {df['cz'].nunique()} CZs")

    outcomes = [
        ("net_slant_norm",  "Net Slant"),
        ("ext_R",           "Share R-Leaning"),
        ("ext_D",           "Share D-Leaning"),
        ("right_norm",      "R Component"),
        ("left_norm",       "L Component"),
    ]

    smooth_data = {}
    relmag_data = {}

    # ===== Smoothness approach =====
    print(f"\n{'='*60}")
    print("  Approach: Smoothness (DeltaSD)")
    print(f"{'='*60}")

    for depvar, label in outcomes:
        print(f"\n  {depvar} ({label})")
        beta_vul, vcov_vul, vul_years = run_event_study(df, depvar, years)
        sens_df, mbar_min = smoothness_sensitivity(beta_vul, vcov_vul, vul_years)
        bd = find_breakdown(sens_df)
        theta = sens_df["theta_hat"].iloc[0]
        se = sens_df["se_theta"].iloc[0]
        print(f"    theta = {theta:.4f}, SE = {se:.4f}, "
              f"Breakdown = {bd:.4f}, Mbar_min = {mbar_min:.4f}")
        if np.isfinite(bd):
            print(f"    Breakdown / Mbar_min = {bd / mbar_min:.2f}x")
        smooth_data[depvar] = {
            "label": label, "sens_df": sens_df, "mbar_min": mbar_min,
            "breakdown": bd, "theta": theta, "se": se,
        }

    # ===== Relative magnitudes approach =====
    print(f"\n{'='*60}")
    print("  Approach: Relative Magnitudes (DeltaRM)")
    print(f"{'='*60}")

    for depvar, label in outcomes:
        print(f"\n  {depvar} ({label})")
        beta_vul, vcov_vul, vul_years = run_event_study(df, depvar, years)
        sens_df, max_pre_fd = relative_magnitudes_sensitivity(beta_vul, vcov_vul, vul_years)
        bd = find_breakdown(sens_df)
        theta = sens_df["theta_hat"].iloc[0]
        se = sens_df["se_theta"].iloc[0]
        print(f"    theta = {theta:.4f}, SE = {se:.4f}, "
              f"Breakdown Mbar = {bd:.2f}")
        relmag_data[depvar] = {
            "label": label, "sens_df": sens_df, "max_pre_fd": max_pre_fd,
            "breakdown": bd, "theta": theta, "se": se,
        }

    # ===================================================================
    # Summary CSV
    # ===================================================================
    rows = []
    for depvar, label in outcomes:
        sd = smooth_data[depvar]
        rm = relmag_data[depvar]
        rows.append({
            "depvar": depvar, "label": label,
            "theta_hat": sd["theta"], "se_theta": sd["se"],
            "std_ci_lo": sd["theta"] - Z_CRIT * sd["se"],
            "std_ci_hi": sd["theta"] + Z_CRIT * sd["se"],
            "smooth_Mbar_min": sd["mbar_min"],
            "smooth_breakdown": sd["breakdown"],
            "smooth_breakdown_ratio": sd["breakdown"] / sd["mbar_min"] if sd["mbar_min"] > 0 else np.inf,
            "relmag_breakdown": rm["breakdown"],
            "relmag_max_pre_fd": rm["max_pre_fd"],
        })
    summary_df = pd.DataFrame(rows)
    csv_path = TAB_DIR / "honestdid_sensitivity.csv"
    summary_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Summary saved: {csv_path}")

    # ===================================================================
    # Figure: combined 2-row plot (smoothness top, relative magnitudes bottom)
    # ===================================================================
    n_out = len(outcomes)
    fig, axes = plt.subplots(2, n_out, figsize=(3.8 * n_out, 7))

    # Top row: Smoothness
    for j, (depvar, label) in enumerate(outcomes):
        ax = axes[0, j]
        sd = smooth_data[depvar]
        sdf = sd["sens_df"]
        feas = sdf[sdf["feasible"] == True]
        if len(feas) == 0:
            ax.text(0.5, 0.5, "No feasible\nMbar", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_title(label, fontsize=9)
            continue

        mbar = feas["Mbar"].values
        ci_lo = feas["robust_ci_lo"].values
        ci_hi = feas["robust_ci_hi"].values

        ax.fill_between(mbar, ci_lo, ci_hi, alpha=0.25, color="#4c72b0")
        ax.plot(mbar, ci_lo, color="#4c72b0", linewidth=0.8)
        ax.plot(mbar, ci_hi, color="#4c72b0", linewidth=0.8)
        ax.axhline(sd["theta"], color="#333", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.5)

        # Mbar_min line
        ax.axvline(sd["mbar_min"], color="#999999", linewidth=0.8, linestyle=":",
                   alpha=0.7)

        # Breakdown
        bd = sd["breakdown"]
        if np.isfinite(bd):
            ax.axvline(bd, color="#c44e52", linewidth=1.2, linestyle="--", alpha=0.8)
            # Position label
            ypos = ax.get_ylim()[0] + 0.85 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(bd * 1.03, ypos, f"$\\bar{{M}}^*$={bd:.3f}",
                    fontsize=6.5, color="#c44e52", ha="left", va="top")

        ax.set_title(label, fontsize=9)
        if j == 0:
            ax.set_ylabel("Robust CI\n(Smoothness)", fontsize=8)
        ax.set_xlabel("$\\bar{M}$", fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Bottom row: Relative magnitudes
    for j, (depvar, label) in enumerate(outcomes):
        ax = axes[1, j]
        rm = relmag_data[depvar]
        sdf = rm["sens_df"]

        mbar = sdf["Mbar"].values
        ci_lo = sdf["robust_ci_lo"].values
        ci_hi = sdf["robust_ci_hi"].values

        ax.fill_between(mbar, ci_lo, ci_hi, alpha=0.25, color="#dd8452")
        ax.plot(mbar, ci_lo, color="#dd8452", linewidth=0.8)
        ax.plot(mbar, ci_hi, color="#dd8452", linewidth=0.8)
        ax.axhline(rm["theta"], color="#333", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.5)

        bd = rm["breakdown"]
        if np.isfinite(bd):
            ax.axvline(bd, color="#c44e52", linewidth=1.2, linestyle="--", alpha=0.8)
            ypos = ax.get_ylim()[0] + 0.85 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(bd * 1.03, ypos, f"$\\bar{{M}}^*$={bd:.1f}",
                    fontsize=6.5, color="#c44e52", ha="left", va="top")

        ax.set_title(label, fontsize=9)
        if j == 0:
            ax.set_ylabel("Robust CI\n(Rel. Magnitudes)", fontsize=8)
        ax.set_xlabel("$\\bar{M}$", fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.8)
    fig_path = FIG_DIR / "honestdid_sensitivity.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # ===================================================================
    # Net-slant-only figure for the paper (cleaner, single panel per approach)
    # ===================================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Smoothness
    ax = axes[0]
    sd = smooth_data["net_slant_norm"]
    feas = sd["sens_df"][sd["sens_df"]["feasible"] == True]
    if len(feas) > 0:
        mbar = feas["Mbar"].values
        ax.fill_between(mbar, feas["robust_ci_lo"].values, feas["robust_ci_hi"].values,
                        alpha=0.3, color="#4c72b0", label="95% Robust CI")
        ax.plot(mbar, feas["robust_ci_lo"].values, color="#4c72b0", linewidth=1.0)
        ax.plot(mbar, feas["robust_ci_hi"].values, color="#4c72b0", linewidth=1.0)
    ax.axhline(sd["theta"], color="#333", linewidth=0.8, linestyle="--", alpha=0.5,
               label=f"Point est. = {sd['theta']:.3f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(sd["mbar_min"], color="#999", linewidth=0.8, linestyle=":",
               label=f"$\\bar{{M}}_{{min}}$ = {sd['mbar_min']:.4f}")
    bd = sd["breakdown"]
    if np.isfinite(bd):
        ax.axvline(bd, color="#c44e52", linewidth=1.5, linestyle="--",
                   label=f"Breakdown $\\bar{{M}}^*$ = {bd:.4f}")
    ax.set_xlabel("$\\bar{M}$ (smoothness bound on $\\Delta^2\\delta$)", fontsize=10)
    ax.set_ylabel("Avg. post-treatment effect", fontsize=10)
    ax.set_title("(a) Smoothness Restriction ($\\Delta$SD)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Relative magnitudes
    ax = axes[1]
    rm = relmag_data["net_slant_norm"]
    sdf = rm["sens_df"]
    mbar = sdf["Mbar"].values
    ax.fill_between(mbar, sdf["robust_ci_lo"].values, sdf["robust_ci_hi"].values,
                    alpha=0.3, color="#dd8452", label="95% Robust CI")
    ax.plot(mbar, sdf["robust_ci_lo"].values, color="#dd8452", linewidth=1.0)
    ax.plot(mbar, sdf["robust_ci_hi"].values, color="#dd8452", linewidth=1.0)
    ax.axhline(rm["theta"], color="#333", linewidth=0.8, linestyle="--", alpha=0.5,
               label=f"Point est. = {rm['theta']:.3f}")
    ax.axhline(0, color="black", linewidth=0.5)
    bd = rm["breakdown"]
    if np.isfinite(bd):
        ax.axvline(bd, color="#c44e52", linewidth=1.5, linestyle="--",
                   label=f"Breakdown $\\bar{{M}}^*$ = {bd:.1f}")
    ax.set_xlabel("$\\bar{M}$ (ratio to max pre-trend change)", fontsize=10)
    ax.set_ylabel("Avg. post-treatment effect", fontsize=10)
    ax.set_title("(b) Relative Magnitudes ($\\Delta$RM)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig_path2 = FIG_DIR / "honestdid_net_slant.pdf"
    fig.savefig(fig_path2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {fig_path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
