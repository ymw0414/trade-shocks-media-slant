"""
robustness_3window.py

Robustness check: 3-congress rolling windows for LASSO training.

Replaces the main pipeline's 2-congress window slant scores with
3-congress window scores, then re-runs DiD regressions.

Steps:
  1. Project 3-window LASSO coefficients onto newspaper features
  2. Normalize by 3-window partisan gap (mu_R - mu_D)
  3. Aggregate to newspaper-year panel
  4. Swap slant columns into the existing regression panel
  5. Run DiD regressions and compare with main results

Requires: 06_train_lasso_3window.py to have been run first.

Inputs:
  - data/processed/speeches/models/06_lasso_3w_*.joblib
  - data/processed/speeches/models/06_intersection_cols.npy
  - data/processed/speeches/05_feature_matrix.npz / meta.parquet
  - data/processed/newspapers/07_newspaper_features_cong_{cong}.npz
  - data/processed/newspapers/04_newspaper_labeled_cong_{cong}.parquet
  - data/processed/panel/14_regression_panel.parquet

Outputs:
  - output/tables/robustness_3window_results.csv
"""

import gc
import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyfixest as pf
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

MODEL_DIR = BASE_DIR / "data" / "processed" / "speeches" / "models"
SPEECH_DIR = BASE_DIR / "data" / "processed" / "speeches"
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"
PANEL_PATH = BASE_DIR / "data" / "processed" / "runs" / "exp_shvocab_cv" / "panel" / "14_regression_panel.parquet"
TAB_DIR = BASE_DIR / "output" / "tables"

ECON_PERCENTILE = 90
NAFTA_YEAR = 1994
END_YEAR = 2004

SLANT_COLS_RAW = ["right_intensity", "left_intensity", "net_slant", "politicization"]
SLANT_COLS_NORM = ["right_norm", "left_norm", "net_slant_norm", "politicization_norm"]
SLANT_COLS = SLANT_COLS_RAW + SLANT_COLS_NORM

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


def compute_raw_scores(X, coef):
    """Compute raw slant decomposition from feature matrix and coefficients."""
    pos_mask = coef > 0
    neg_mask = coef < 0
    right = X[:, pos_mask].dot(coef[pos_mask])
    left = X[:, neg_mask].dot(np.abs(coef[neg_mask]))
    if hasattr(right, 'A1'):
        right = right.A1
    if hasattr(left, 'A1'):
        left = left.A1
    return right, left


def main():
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    congresses = list(range(100, 109))
    windows = [(t - 2, t - 1, t) for t in congresses]

    # ------------------------------------------------------------------
    # 1. Load speech data (for normalization gap computation)
    # ------------------------------------------------------------------
    print("Loading speech feature matrix and metadata ...")
    X_speech = sp.load_npz(SPEECH_DIR / "05_feature_matrix.npz")
    speech_meta = pd.read_parquet(SPEECH_DIR / "05_feature_meta.parquet")

    intersection_cols = np.load(MODEL_DIR / "06_intersection_cols_3w.npy")
    X_speech = X_speech[:, intersection_cols]
    print(f"  Speeches: {X_speech.shape[0]:,} x {X_speech.shape[1]:,} features")

    # ------------------------------------------------------------------
    # 2. Project + Normalize + Aggregate (per congress)
    # ------------------------------------------------------------------
    print("\nProjecting, normalizing, and aggregating ...\n")

    general_chunks = []
    econ_chunks = []
    norm_params = []

    for i, window_congs in enumerate(windows, 1):
        cong_curr = window_congs[-1]
        step_start = time.time()

        # Load 3-window model
        window_file = "_".join(str(c) for c in window_congs)
        model_path = MODEL_DIR / f"06_lasso_3w_{window_file}.joblib"
        if not model_path.exists():
            print(f"  WARNING: {model_path.name} not found, skipping")
            continue
        model = joblib.load(model_path)
        coef = model.coef_[0]
        n_pos = int((coef > 0).sum())
        n_neg = int((coef < 0).sum())

        # --- Compute normalization gap from ALL legislators in 3-congress window ---
        mask_window = speech_meta["congress_int"].isin(window_congs)
        mask_rep = mask_window & (speech_meta["party_code"] == "R")
        mask_dem = mask_window & (speech_meta["party_code"] == "D")

        idx_rep = speech_meta.index[mask_rep].values
        idx_dem = speech_meta.index[mask_dem].values

        right_rep, left_rep = compute_raw_scores(X_speech[idx_rep], coef)
        right_dem, left_dem = compute_raw_scores(X_speech[idx_dem], coef)

        mu_R = np.mean(right_rep - left_rep)
        mu_D = np.mean(right_dem - left_dem)
        gap = mu_R - mu_D

        norm_params.append({
            "congress": cong_curr,
            "window": "+".join(str(c) for c in window_congs),
            "n_rep": len(idx_rep), "n_dem": len(idx_dem),
            "mu_R": mu_R, "mu_D": mu_D, "gap": gap,
        })

        # --- Project onto newspaper features ---
        features_path = NEWSPAPER_DIR / f"07_newspaper_features_cong_{cong_curr}.npz"
        if not features_path.exists():
            print(f"  WARNING: {features_path.name} not found, skipping")
            continue

        X_news = sp.load_npz(features_path)
        X_news = X_news[:, intersection_cols]

        pos_mask = coef > 0
        neg_mask = coef < 0
        right_intensity = X_news[:, pos_mask].dot(coef[pos_mask])
        left_intensity = X_news[:, neg_mask].dot(np.abs(coef[neg_mask]))
        if hasattr(right_intensity, 'A1'):
            right_intensity = right_intensity.A1
        if hasattr(left_intensity, 'A1'):
            left_intensity = left_intensity.A1

        net_slant = right_intensity - left_intensity
        politicization = right_intensity + left_intensity

        # Normalize
        right_norm = right_intensity / gap
        left_norm = left_intensity / gap
        net_slant_norm = net_slant / gap
        politicization_norm = politicization / gap

        # --- Load labels and aggregate ---
        label_path = NEWSPAPER_DIR / f"04_newspaper_labeled_cong_{cong_curr}.parquet"
        meta_news = pd.read_parquet(label_path)
        n_articles = len(meta_news)

        assert n_articles == len(right_intensity), \
            f"Congress {cong_curr}: labels ({n_articles}) != slant ({len(right_intensity)})"

        df = meta_news.copy()
        df["right_intensity"] = right_intensity
        df["left_intensity"] = left_intensity
        df["net_slant"] = net_slant
        df["politicization"] = politicization
        df["right_norm"] = right_norm
        df["left_norm"] = left_norm
        df["net_slant_norm"] = net_slant_norm
        df["politicization_norm"] = politicization_norm

        # Filter: news only
        df = df[df["is_news"]].copy()
        n_news = len(df)

        # Econ threshold: per-congress p90
        econ_threshold = np.percentile(df["econ_score"], ECON_PERCENTILE)
        econ_mask = df["econ_score"] >= econ_threshold
        df["is_econ"] = df["econ_score"] > 0

        # Track A: General
        general_agg = (
            df.groupby(["paper", "year"])
            .agg(
                n_articles=("net_slant", "count"),
                n_econ_any=("is_econ", "sum"),
                **{f"{col}": (col, "mean") for col in SLANT_COLS},
            )
            .reset_index()
        )
        general_agg["econ_share"] = general_agg["n_econ_any"] / general_agg["n_articles"]
        general_chunks.append(general_agg)

        # Track B: Economic (top 10%)
        df_econ = df[econ_mask]
        econ_agg = (
            df_econ.groupby(["paper", "year"])
            .agg(
                n_articles_econ=("net_slant", "count"),
                **{f"{col}_econ": (col, "mean") for col in SLANT_COLS},
            )
            .reset_index()
        )
        econ_chunks.append(econ_agg)

        elapsed = time.time() - step_start
        print(f"  [{i}/{len(windows)}] Congress {cong_curr}: "
              f"{n_news:,} news articles  |  coefs +{n_pos}/-{n_neg}  |  "
              f"gap={gap:.3f}  |  slant_norm mean={df['net_slant_norm'].mean():.3f}  |  "
              f"{elapsed:.1f}s")

        del model, coef, X_news, df, df_econ, meta_news
        gc.collect()

    # ------------------------------------------------------------------
    # 3. Combine into newspaper-year panel
    # ------------------------------------------------------------------
    print("\nCombining newspaper-year panel ...")
    general = pd.concat(general_chunks, ignore_index=True)
    econ = pd.concat(econ_chunks, ignore_index=True)
    new_panel = general.merge(econ, on=["paper", "year"], how="left")
    print(f"  3-window panel: {len(new_panel):,} paper-years")

    # ------------------------------------------------------------------
    # 4. Swap slant columns into existing regression panel
    # ------------------------------------------------------------------
    print("\nSwapping slant columns into regression panel ...")
    reg_panel = pd.read_parquet(PANEL_PATH)

    # Columns to replace
    slant_related = [
        "n_articles", "n_econ_any",
        "right_intensity", "left_intensity", "net_slant", "politicization",
        "right_norm", "left_norm", "net_slant_norm", "politicization_norm",
        "econ_share",
        "n_articles_econ",
        "right_intensity_econ", "left_intensity_econ", "net_slant_econ", "politicization_econ",
        "right_norm_econ", "left_norm_econ", "net_slant_norm_econ", "politicization_norm_econ",
    ]

    # Keep only non-slant columns from existing panel
    keep_cols = [c for c in reg_panel.columns if c not in slant_related]
    reg_base = reg_panel[keep_cols].copy()

    # Merge new slant columns
    panel_3w = reg_base.merge(new_panel, on=["paper", "year"], how="inner")
    panel_3w = panel_3w.sort_values(["paper", "year"]).reset_index(drop=True)

    # Re-create interaction terms
    panel_3w["post_nafta"] = (panel_3w["year"] >= NAFTA_YEAR).astype(int)
    panel_3w["vuln_x_post"] = panel_3w["vulnerability1990_scaled"] * panel_3w["post_nafta"]

    print(f"  Final panel: {len(panel_3w):,} obs "
          f"(main: {len(reg_panel):,})")

    # ------------------------------------------------------------------
    # 5. Run DiD regressions
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DiD REGRESSIONS: 3-congress window robustness")
    print("=" * 72)

    df = panel_3w[
        panel_3w["cz"].notna() & panel_3w["vulnerability1990_scaled"].notna()
    ].copy()
    if END_YEAR:
        df = df[df["year"] <= END_YEAR].copy()

    df["state_fips"] = (df["fips"] // 1000).astype(int)
    df["division"] = df["state_fips"].map(STATE_TO_DIVISION)
    df["paper_id"] = df["paper"].astype("category").cat.codes
    df["post"] = (df["year"] >= NAFTA_YEAR).astype(int)
    df["vuln_x_post"] = df["vulnerability1990_scaled"] * df["post"]

    years = sorted(df["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == base_yr:
            continue
        df[f"manu_{yr}"] = (df["year"] == yr).astype(float) * df["manushare1990"].fillna(0)
        df[f"china_{yr}"] = (df["year"] == yr).astype(float) * df["china_shock"].fillna(0)

    manu_vars = [f"manu_{yr}" for yr in years if yr != base_yr]
    china_vars = [f"china_{yr}" for yr in years if yr != base_yr]
    manu_rhs = " + ".join(manu_vars)
    china_rhs = " + ".join(china_vars)

    outcomes = [
        ("net_slant_norm",           "Net Slant"),
        ("right_norm",               "Right Intensity"),
        ("left_norm",                "Left Intensity"),
        ("politicization_norm",      "Politicization"),
        ("net_slant_norm_econ",      "Net Slant (Econ)"),
        ("right_norm_econ",          "Right Intensity (Econ)"),
        ("left_norm_econ",           "Left Intensity (Econ)"),
        ("politicization_norm_econ", "Politicization (Econ)"),
        ("econ_share",               "Econ Article Share"),
    ]

    specs = [
        ("spec1", "vuln_x_post",
         "Paper + Year + Div×Year FE"),
        ("spec2", f"vuln_x_post + {manu_rhs}",
         "+ Manushare×Year"),
        ("spec3", f"vuln_x_post + {china_rhs} + {manu_rhs}",
         "+ China + Manu"),
    ]

    all_results = []
    for depvar, label in outcomes:
        print(f"\n  {label} ({depvar})")
        for spec_name, rhs, desc in specs:
            fml = f"{depvar} ~ {rhs} | paper_id + year + division^year"
            m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
            t = m.tidy().loc["vuln_x_post"]
            stars = ""
            if t["Pr(>|t|)"] < 0.01: stars = "***"
            elif t["Pr(>|t|)"] < 0.05: stars = "**"
            elif t["Pr(>|t|)"] < 0.1: stars = "*"
            print(f"    {desc}: {t['Estimate']:.6f}{stars} (SE={t['Std. Error']:.6f})")
            all_results.append({
                "depvar": depvar, "label": label, "spec": spec_name,
                "coef": t["Estimate"], "se": t["Std. Error"],
                "pval": t["Pr(>|t|)"], "N": m._N, "R2": m._r2,
            })

    res_df = pd.DataFrame(all_results)
    out_path = TAB_DIR / "robustness_3window_results.csv"
    res_df.to_csv(out_path, index=False, float_format="%.6f")

    # ------------------------------------------------------------------
    # 6. Comparison with main results
    # ------------------------------------------------------------------
    main_path = TAB_DIR / "did_results.csv"
    if main_path.exists():
        print("\n" + "=" * 72)
        print("COMPARISON: Main (2-window) vs Robustness (3-window)")
        print("=" * 72)
        main_res = pd.read_csv(main_path)

        print(f"\n  {'Outcome':<28} {'Main (2w)':>18} {'Robust (3w)':>18}")
        print(f"  {'-'*64}")
        for depvar, label in outcomes:
            m3 = main_res[(main_res["depvar"]==depvar) & (main_res["spec"]=="spec3")]
            r3 = res_df[(res_df["depvar"]==depvar) & (res_df["spec"]=="spec3")]
            if len(m3) > 0 and len(r3) > 0:
                m = m3.iloc[0]
                r = r3.iloc[0]
                def fmt(row):
                    s = ""
                    if row["pval"] < 0.01: s = "***"
                    elif row["pval"] < 0.05: s = "**"
                    elif row["pval"] < 0.1: s = "*"
                    return f"{row['coef']:.4f}{s} ({row['se']:.4f})"
                print(f"  {label:<28} {fmt(m):>18} {fmt(r):>18}")

    # Normalization params
    print("\n" + "=" * 72)
    print("NORMALIZATION PARAMETERS (3-window)")
    print("=" * 72)
    for p in norm_params:
        print(f"  Congress {p['congress']:>3}  window {p['window']:>12}  "
              f"n_R={p['n_rep']:>4}  n_D={p['n_dem']:>4}  "
              f"mu_R={p['mu_R']:>7.3f}  mu_D={p['mu_D']:>7.3f}  gap={p['gap']:>6.3f}")

    total_time = time.time() - pipeline_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
