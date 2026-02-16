"""
Randomization inference for the DiD specification.

Uses Frisch-Waugh-Lovell theorem for speed:
1. Demean all variables by FEs (paper, year, division x year) via alternating projections
2. Partial out time-varying controls from demeaned outcomes (fixed across permutations)
3. For each permutation: demean permuted treatment, partial out controls, compute OLS coef

5,000 CZ-level permutations of vulnerability.

Outputs:
  - output/tables/permutation_test.csv
  - output/figures/permutation_distribution.pdf
"""

import os, sys, time
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nlp"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

NAFTA_YEAR = 1994
END_YEAR = 2004
N_PERMS = 5000
SEED = 42

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


def demean_by_fes(arr, fe_keys, tol=1e-10, maxiter=1000):
    """Demean array by multiple FE groups using alternating projections (Gauss-Seidel)."""
    resid = arr.copy().astype(np.float64)
    for it in range(maxiter):
        max_delta = 0.0
        for key in fe_keys:
            # Compute group means via pandas groupby (fast for categorical)
            s = pd.Series(resid)
            means = s.groupby(key).transform("mean").values
            delta = resid - means
            max_delta = max(max_delta, np.max(np.abs(resid - delta)))
            resid = delta
        if max_delta < tol:
            break
    return resid


def partial_out(y, X):
    """Partial out X from y via OLS. Returns residual."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def load_panel():
    df = pd.read_parquet(PANEL_PATH)
    df = df[df["cz"].notna() & df["vulnerability1990_scaled"].notna()].copy()
    df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    # Division x year interaction
    df["div_year"] = df["division"].astype(str) + "_" + df["year"].astype(str)

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
    df, years = load_panel()
    N = len(df)
    czs = df["cz"].unique()
    n_cz = len(czs)
    print(f"  {N:,} obs, {n_cz} CZs")

    outcomes = [
        ("net_slant_norm", "Net Slant (Norm)"),
        ("ext_R", "Share R-Leaning"),
        ("ext_D", "Share D-Leaning"),
        ("right_norm", "R Component"),
        ("left_norm", "L Component"),
    ]

    # --- FE group keys (as categorical for fast groupby) ---
    fe_paper = pd.Categorical(df["paper_id"].values)
    fe_year = pd.Categorical(df["year"].values)
    fe_div_year = pd.Categorical(df["div_year"].values)
    fe_keys = [fe_paper, fe_year, fe_div_year]

    # --- Build control matrix ---
    base_yr = years[0]
    control_cols = ([f"china_{yr}" for yr in years if yr != base_yr] +
                    [f"manu_{yr}" for yr in years if yr != base_yr])
    X_raw = df[control_cols].values.astype(np.float64)
    K = X_raw.shape[1]

    print(f"  Demeaning {K} control variables by FEs ...")
    t0 = time.time()
    X_dm = np.column_stack([demean_by_fes(X_raw[:, k], fe_keys) for k in range(K)])
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Demean outcomes and partial out controls ---
    print("  Demeaning outcomes and partialling out controls ...")
    y_resid = {}
    for depvar, label in outcomes:
        y = df[depvar].values.astype(np.float64)
        y_dm = demean_by_fes(y, fe_keys)
        y_resid[depvar] = partial_out(y_dm, X_dm)

    # --- Verify FWL matches pyfixest ---
    print("\nVerifying FWL against pyfixest ...")
    x_actual = df["vuln_x_post"].values.astype(np.float64)
    x_dm = demean_by_fes(x_actual, fe_keys)
    x_resid = partial_out(x_dm, X_dm)

    manu_str = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_str = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])

    actual = {}
    actual_se = {}
    for depvar, label in outcomes:
        fml = f"{depvar} ~ vuln_x_post + {china_str} + {manu_str} | paper_id + year + division^year"
        m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
        t = m.tidy().loc["vuln_x_post"]
        actual[depvar] = t["Estimate"]
        actual_se[depvar] = t["Std. Error"]

        beta_fwl = np.dot(x_resid, y_resid[depvar]) / np.dot(x_resid, x_resid)
        diff = abs(actual[depvar] - beta_fwl)
        print(f"  {label:<25s}  pyfixest={actual[depvar]:.6f}  FWL={beta_fwl:.6f}  diff={diff:.2e}")
        assert diff < 1e-4, f"FWL mismatch for {depvar}: {diff:.6e}"

    print("  FWL verification passed!")

    # --- Permutation loop ---
    cz_vuln = df.groupby("cz")["vulnerability1990_scaled"].first()
    cz_index = cz_vuln.index.values
    cz_vals = cz_vuln.values.copy()
    post_arr = df["post"].values.astype(np.float64)
    cz_arr = df["cz"].values

    # Pre-build CZ -> row indices map for fast mapping
    cz_to_rows = {}
    for i, cz in enumerate(cz_arr):
        cz_to_rows.setdefault(cz, []).append(i)

    rng = np.random.default_rng(SEED)
    perm_coefs = {depvar: np.zeros(N_PERMS) for depvar, _ in outcomes}

    print(f"\nRunning {N_PERMS} permutations ...")
    t0 = time.time()
    for i in range(N_PERMS):
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_PERMS - i - 1) / rate
            print(f"  {i+1}/{N_PERMS}  ({rate:.0f} iter/s, ETA {eta:.0f}s)")

        # Permute vulnerability at CZ level
        shuffled = cz_vals.copy()
        rng.shuffle(shuffled)

        # Map permuted vulnerability to observation level
        x_perm = np.zeros(N)
        for j, cz in enumerate(cz_index):
            for row in cz_to_rows.get(cz, []):
                x_perm[row] = shuffled[j] * post_arr[row]

        # Demean by FEs and partial out controls
        x_perm_dm = demean_by_fes(x_perm, fe_keys)
        x_perm_resid = partial_out(x_perm_dm, X_dm)

        # OLS coefficient for each outcome
        denom = np.dot(x_perm_resid, x_perm_resid)
        if denom < 1e-12:
            for depvar, _ in outcomes:
                perm_coefs[depvar][i] = np.nan
            continue

        for depvar, _ in outcomes:
            perm_coefs[depvar][i] = np.dot(x_perm_resid, y_resid[depvar]) / denom

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({N_PERMS/elapsed:.0f} iter/s)")

    # --- Compute empirical p-values ---
    print(f"\n{'Outcome':<25s} {'Actual':>10s} {'SE':>10s} {'Perm Mean':>10s} {'Perm SD':>10s} {'p-value':>10s}")
    print("-" * 80)

    results = []
    for depvar, label in outcomes:
        perm_arr = perm_coefs[depvar]
        perm_arr = perm_arr[~np.isnan(perm_arr)]
        act = actual[depvar]
        se = actual_se[depvar]
        p_val = np.mean(np.abs(perm_arr) >= np.abs(act))
        print(f"  {label:<25s} {act:>10.4f} {se:>10.4f} {perm_arr.mean():>10.4f} {perm_arr.std():>10.4f} {p_val:>10.4f}")
        results.append({
            "depvar": depvar, "label": label,
            "actual_coef": act, "actual_se": se,
            "perm_mean": perm_arr.mean(), "perm_sd": perm_arr.std(),
            "p_value_twosided": p_val, "n_perms": len(perm_arr),
        })

    # Save CSV
    res_df = pd.DataFrame(results)
    csv_path = TAB_DIR / "permutation_test.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nResults saved: {csv_path}")

    # --- Plot (Share R-Leaning only) ---
    plot_var, plot_label = "ext_R", "Share R-Leaning"
    perm_arr = perm_coefs[plot_var]
    perm_arr = perm_arr[~np.isnan(perm_arr)]
    act = actual[plot_var]
    p_val = np.mean(np.abs(perm_arr) >= np.abs(act))

    # Critical values from the permutation distribution (two-sided)
    abs_perm = np.abs(perm_arr)
    cv_levels = [(0.10, "10%", "#888888"), (0.05, "5%", "#555555"), (0.01, "1%", "#222222")]
    cv_vals = {alpha: np.quantile(abs_perm, 1 - alpha) for alpha, _, _ in cv_levels}

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(perm_arr, bins=60, color="#cccccc", edgecolor="#999999",
            linewidth=0.3, density=True, zorder=1)

    # Critical value lines (positive side only, with rotated labels)
    ymax = ax.get_ylim()[1]
    for alpha, label, color in cv_levels:
        cv = cv_vals[alpha]
        ax.axvline(cv, color=color, linewidth=1.0, linestyle="--", zorder=2, alpha=0.7)
        ax.text(cv + 0.002, ymax * 0.5, label, fontsize=7, ha="left", va="center",
                color=color, rotation=90)

    ax.axvline(act, color="#c0392b", linewidth=1.8, linestyle="-",
               zorder=3)
    ax.set_xlabel("Coefficient", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.text(0.97, 0.95, f"$p$ = {p_val:.3f}",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig_path = FIG_DIR / "permutation_distribution.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")
    print("Done.")


if __name__ == "__main__":
    main()
