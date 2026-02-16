"""
Trade-feature zeroing robustness test.

Zero out LASSO coefficients for trade-related vocabulary (tariff, NAFTA,
trade, export, etc.), re-score newspaper articles, re-normalize, re-aggregate,
and re-run the DiD regression.  If the treatment effect survives, it confirms
that the slant shift operates through general political language framing,
not trade-topic coverage.

Two definitions of "trade features":
  NARROW: only unambiguously trade-related tokens (trade, tariff, nafta, ...)
  BROAD:  narrow + ambiguous "import" forms (stemmed from both import/important)

Outputs:
  - output/tables/trade_zeroing_comparison.csv
  - output/figures/trade_zeroing_comparison.pdf
"""

import gc, os, sys, time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utils"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
MODEL_DIR = cfg.MODEL_DIR
NEWS_IN_DIR = cfg.INPUT_NEWS_DIR
SPEECH_DIR = cfg.INPUT_SPEECH_DIR
LABEL_DIR = cfg.NEWSPAPER_LABELS
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

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

# --- Trade feature definitions -----------------------------------------------
# Narrow: unambiguously about trade policy
NARROW_TRADE_EXACT = {
    "trade", "tariff", "nafta", "export", "gatt", "wto", "quota",
    "importantli",  # NOT trade — but included here as exclusion
}
# Patterns for narrow: starts with these stems
NARROW_TRADE_PREFIXES = ["trade ", "export ", "fair trade", "free trade",
                          "unfair trade", "world trade", "intern trade",
                          "billion trade"]

# Broad: add "import" standalone + all "import X" bigrams
BROAD_EXTRA_EXACT = {"import"}
# All "import X" or "X import" bigrams
BROAD_IMPORT_PATTERN = True  # include any feature containing "import" token


def classify_trade_features(feature_names):
    """Classify features as narrow-trade, broad-trade, or non-trade."""
    narrow_mask = np.zeros(len(feature_names), dtype=bool)
    broad_mask = np.zeros(len(feature_names), dtype=bool)

    narrow_exact = {"trade", "tariff", "nafta", "export", "gatt", "wto", "quota"}
    broad_extra = {"import"}

    for i, feat in enumerate(feature_names):
        f = feat.lower().strip()

        # Check narrow
        is_narrow = False
        if f in narrow_exact:
            is_narrow = True
        else:
            for prefix in NARROW_TRADE_PREFIXES:
                if f.startswith(prefix) or f == prefix.strip():
                    is_narrow = True
                    break

        if is_narrow:
            narrow_mask[i] = True
            broad_mask[i] = True
            continue

        # Check broad (import-related)
        if f in broad_extra:
            broad_mask[i] = True
            continue

        # "import X" or "X import" bigrams — all are ambiguous
        tokens = f.split()
        if "import" in tokens:
            broad_mask[i] = True

    return narrow_mask, broad_mask


def score_articles_modified(model, X, coef_modified):
    """Score articles using modified coefficients."""
    pos_mask = coef_modified > 0
    neg_mask = coef_modified < 0

    right = X[:, pos_mask].dot(coef_modified[pos_mask])
    left = X[:, neg_mask].dot(np.abs(coef_modified[neg_mask]))

    if hasattr(right, 'A1'):
        right = right.A1
    if hasattr(left, 'A1'):
        left = left.A1

    net_slant = right - left
    politicization = right + left
    return right, left, net_slant, politicization


def run_pipeline(zero_mask, label, feature_names):
    """
    Re-run steps 08-10 with zeroed coefficients, then return panel.

    zero_mask: boolean array of features to zero out.
    """
    windows = cfg.get_windows()
    congresses = cfg.get_congresses()

    # Load shared vocab mask
    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = np.load(shared_vocab_path) if shared_vocab_path.exists() else None

    # Load speech data for normalization
    X_speech = sp.load_npz(SPEECH_DIR / "05_feature_matrix.npz")
    speech_meta = pd.read_parquet(SPEECH_DIR / "05_feature_meta.parquet")
    if shared_vocab_mask is not None:
        X_speech = X_speech[:, shared_vocab_mask]

    PARTY_COL = "party" if "party" in speech_meta.columns else "party_code"

    n_zeroed = zero_mask.sum()
    zeroed_features = feature_names[zero_mask]

    # Check which zeroed features actually had non-zero LASSO coefs
    # (done per congress below)

    print(f"\n  [{label}] Zeroing {n_zeroed} features")

    panel_chunks = []
    total_coef_weight_zeroed = []

    for i, window_congs in enumerate(windows, 1):
        cong_curr = window_congs[-1]
        window_file = "_".join(str(c) for c in window_congs)

        # Load model
        model_path = MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        coef_orig = model.coef_[0].copy()

        # Zero trade features
        coef_mod = coef_orig.copy()
        coef_mod[zero_mask] = 0.0

        # Track how much coefficient weight is zeroed
        orig_weight = np.sum(np.abs(coef_orig))
        zeroed_weight = np.sum(np.abs(coef_orig[zero_mask]))
        total_coef_weight_zeroed.append({
            "congress": cong_curr,
            "n_active_zeroed": (coef_orig[zero_mask] != 0).sum(),
            "weight_zeroed": zeroed_weight,
            "weight_total": orig_weight,
            "pct_zeroed": zeroed_weight / orig_weight * 100 if orig_weight > 0 else 0,
        })

        # --- Step 08: Score newspaper articles ---
        feat_path = NEWS_IN_DIR / f"07_newspaper_features_cong_{cong_curr}.npz"
        if not feat_path.exists():
            continue
        X_news = sp.load_npz(feat_path)
        if shared_vocab_mask is not None:
            X_news = X_news[:, shared_vocab_mask]

        right, left, net_slant, politicization = score_articles_modified(model, X_news, coef_mod)

        # --- Step 09: Normalize ---
        mask_window = speech_meta["congress_int"].isin(window_congs)
        mask_rep = mask_window & ((speech_meta[PARTY_COL] == "R") | (speech_meta[PARTY_COL] == 200))
        mask_dem = mask_window & ((speech_meta[PARTY_COL] == "D") | (speech_meta[PARTY_COL] == 100))

        idx_rep = speech_meta.index[mask_rep].values
        idx_dem = speech_meta.index[mask_dem].values

        # Compute gap with modified coefs
        right_rep, left_rep, _, _ = score_articles_modified(model, X_speech[idx_rep], coef_mod)
        right_dem, left_dem, _, _ = score_articles_modified(model, X_speech[idx_dem], coef_mod)
        mu_R = np.mean(right_rep - left_rep)
        mu_D = np.mean(right_dem - left_dem)
        gap = mu_R - mu_D

        if abs(gap) < 1e-10:
            gap = 1e-10  # avoid division by zero

        slant_df = pd.DataFrame({
            "right_intensity": right, "left_intensity": left,
            "net_slant": net_slant, "politicization": politicization,
            "net_slant_norm": net_slant / gap,
            "right_norm": right / gap, "left_norm": left / gap,
            "politicization_norm": politicization / gap,
        })

        # --- Step 10: Aggregate ---
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong_curr}.parquet"
        if not label_path.exists():
            continue
        meta = pd.read_parquet(label_path)

        sample_idx_path = cfg.NEWS_DIR / f"07_sample_idx_cong_{cong_curr}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)

        assert len(meta) == len(slant_df), f"Cong {cong_curr}: meta/slant mismatch"
        df = pd.concat([meta, slant_df], axis=1)
        df = df[df["is_news"]].copy()

        # Aggregate
        SLANT_COLS = ["right_intensity", "left_intensity", "net_slant", "politicization",
                      "right_norm", "left_norm", "net_slant_norm", "politicization_norm"]
        agg = df.groupby(["paper", "year"]).agg(
            n_articles=("net_slant", "count"),
            **{col: (col, "mean") for col in SLANT_COLS},
        ).reset_index()

        # Extensive margins
        grp = df.groupby(["paper", "year"])["net_slant"]
        agg["ext_nonzero"] = grp.apply(lambda x: (x != 0).mean()).values
        agg["ext_R"] = grp.apply(lambda x: (x > 0).mean()).values
        agg["ext_D"] = grp.apply(lambda x: (x < 0).mean()).values

        # Intensive margins
        grp2 = df.groupby(["paper", "year"])
        agg["int_net_slant_norm"] = grp2.apply(
            lambda g: g.loc[g["net_slant"] != 0, "net_slant_norm"].mean()
            if (g["net_slant"] != 0).any() else np.nan
        ).values
        agg["int_right_norm"] = grp2.apply(
            lambda g: g.loc[g["net_slant"] != 0, "right_norm"].mean()
            if (g["net_slant"] != 0).any() else np.nan
        ).values
        agg["int_left_norm"] = grp2.apply(
            lambda g: g.loc[g["net_slant"] != 0, "left_norm"].mean()
            if (g["net_slant"] != 0).any() else np.nan
        ).values

        panel_chunks.append(agg)
        del model, X_news, slant_df, meta, df, agg
        gc.collect()

    panel_new = pd.concat(panel_chunks, ignore_index=True)
    panel_new = panel_new.sort_values(["paper", "year"]).reset_index(drop=True)

    # Print weight summary
    wdf = pd.DataFrame(total_coef_weight_zeroed)
    print(f"    Active features zeroed: {wdf['n_active_zeroed'].mean():.0f} avg per congress")
    print(f"    Weight zeroed: {wdf['pct_zeroed'].mean():.1f}% of total |coef| weight")

    return panel_new, wdf


def merge_to_regression_panel(panel_new):
    """Merge re-scored newspaper panel with geography/econ data for regression."""
    # Load original panel for geography + econ variables
    orig = pd.read_parquet(PANEL_PATH)
    geo_econ_cols = [c for c in orig.columns
                     if c not in panel_new.columns or c in ("paper", "year")]
    geo_econ = orig[geo_econ_cols].drop_duplicates(subset=["paper", "year"])

    merged = panel_new.merge(geo_econ, on=["paper", "year"], how="inner")
    return merged


def run_did(df, depvar, years):
    """Run DiD spec 3 and return vuln_x_post coefficient, SE, t, p."""
    base_yr = years[0]
    manu_vars = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_vars = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])
    fml = f"{depvar} ~ vuln_x_post + {china_vars} + {manu_vars} | paper_id + year + division^year"
    m = pf.feols(fml, data=df, vcov={"CRV1": "cz"})
    t = m.tidy().loc["vuln_x_post"]
    return t["Estimate"], t["Std. Error"], t["t value"], t["Pr(>|t|)"]


def prepare_for_regression(df):
    """Add regression variables to panel."""
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

    # --- Load feature names ---
    vec = joblib.load(SPEECH_DIR / "05_vectorizer.joblib")
    all_features = vec.get_feature_names_out()

    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = np.load(shared_vocab_path) if shared_vocab_path.exists() else None
    if shared_vocab_mask is not None:
        features = all_features[shared_vocab_mask]
    else:
        features = all_features

    # --- Classify features ---
    narrow_mask, broad_mask = classify_trade_features(features)
    print(f"Feature classification:")
    print(f"  Total features: {len(features)}")
    print(f"  Narrow trade:   {narrow_mask.sum()}")
    print(f"  Broad trade:    {broad_mask.sum()}")

    # Print narrow trade features
    print(f"\n  Narrow trade features ({narrow_mask.sum()}):")
    for feat in features[narrow_mask]:
        print(f"    {feat}")

    # --- Run original baseline ---
    print("\n" + "=" * 60)
    print("  BASELINE (original coefficients)")
    print("=" * 60)

    orig_panel = pd.read_parquet(PANEL_PATH)
    orig_panel, years = prepare_for_regression(orig_panel)

    outcomes = [
        ("net_slant_norm", "Net Slant"),
        ("ext_R", "Share R-Leaning"),
        ("ext_D", "Share D-Leaning"),
        ("right_norm", "R Component"),
        ("left_norm", "L Component"),
    ]

    baseline = {}
    for depvar, label in outcomes:
        coef, se, tval, pval = run_did(orig_panel, depvar, years)
        baseline[depvar] = {"coef": coef, "se": se, "t": tval, "p": pval}
        print(f"  {label:<20s}  coef={coef:.4f}  se={se:.4f}  t={tval:.2f}  p={pval:.4f}")

    # --- Run narrow trade zeroing ---
    print("\n" + "=" * 60)
    print("  NARROW TRADE ZEROING")
    print("=" * 60)

    panel_narrow, wdf_narrow = run_pipeline(narrow_mask, "Narrow", features)
    merged_narrow = merge_to_regression_panel(panel_narrow)
    merged_narrow, years_n = prepare_for_regression(merged_narrow)

    narrow_results = {}
    for depvar, label in outcomes:
        coef, se, tval, pval = run_did(merged_narrow, depvar, years_n)
        narrow_results[depvar] = {"coef": coef, "se": se, "t": tval, "p": pval}
        print(f"  {label:<20s}  coef={coef:.4f}  se={se:.4f}  t={tval:.2f}  p={pval:.4f}")

    # --- Run broad trade zeroing ---
    print("\n" + "=" * 60)
    print("  BROAD TRADE ZEROING")
    print("=" * 60)

    panel_broad, wdf_broad = run_pipeline(broad_mask, "Broad", features)
    merged_broad = merge_to_regression_panel(panel_broad)
    merged_broad, years_b = prepare_for_regression(merged_broad)

    broad_results = {}
    for depvar, label in outcomes:
        coef, se, tval, pval = run_did(merged_broad, depvar, years_b)
        broad_results[depvar] = {"coef": coef, "se": se, "t": tval, "p": pval}
        print(f"  {label:<20s}  coef={coef:.4f}  se={se:.4f}  t={tval:.2f}  p={pval:.4f}")

    # --- Summary table ---
    rows = []
    for depvar, label in outcomes:
        b = baseline[depvar]
        n = narrow_results[depvar]
        br = broad_results[depvar]
        rows.append({
            "depvar": depvar, "label": label,
            "baseline_coef": b["coef"], "baseline_se": b["se"], "baseline_p": b["p"],
            "narrow_coef": n["coef"], "narrow_se": n["se"], "narrow_p": n["p"],
            "narrow_pct_change": (n["coef"] - b["coef"]) / abs(b["coef"]) * 100 if b["coef"] != 0 else 0,
            "broad_coef": br["coef"], "broad_se": br["se"], "broad_p": br["p"],
            "broad_pct_change": (br["coef"] - b["coef"]) / abs(b["coef"]) * 100 if b["coef"] != 0 else 0,
        })
    result_df = pd.DataFrame(rows)
    csv_path = TAB_DIR / "trade_zeroing_comparison.csv"
    result_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Results saved: {csv_path}")

    # Also save weight info
    wdf_narrow["scope"] = "narrow"
    wdf_broad["scope"] = "broad"
    weight_df = pd.concat([wdf_narrow, wdf_broad], ignore_index=True)
    weight_df.to_csv(TAB_DIR / "trade_zeroing_weights.csv", index=False, float_format="%.4f")

    # --- Figure ---
    fig, axes = plt.subplots(1, len(outcomes), figsize=(3.5 * len(outcomes), 3.5))
    if len(outcomes) == 1:
        axes = [axes]

    for ax, (depvar, label) in zip(axes, outcomes):
        b = baseline[depvar]
        n = narrow_results[depvar]
        br = broad_results[depvar]

        x_pos = [0, 1, 2]
        coefs = [b["coef"], n["coef"], br["coef"]]
        ses = [b["se"], n["se"], br["se"]]
        colors = ["#2d2d2d", "#7a7a7a", "#bf6b63"]
        labels_x = ["Baseline", "Narrow\nzeroed", "Broad\nzeroed"]

        for j, (xp, c, s, col) in enumerate(zip(x_pos, coefs, ses, colors)):
            ax.errorbar(xp, c, yerr=1.96 * s, fmt="o", color=col,
                        markersize=6, capsize=3, linewidth=1.5)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_x, fontsize=7)
        ax.set_title(label, fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("DiD coefficient", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig_path = FIG_DIR / "trade_zeroing_comparison.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # Print pct change summary
    print("\n  === SUMMARY ===")
    print(f"  {'Outcome':<20s}  {'Baseline':>10s}  {'Narrow':>10s}  {'%chg':>6s}  {'Broad':>10s}  {'%chg':>6s}")
    for _, r in result_df.iterrows():
        print(f"  {r['label']:<20s}  {r['baseline_coef']:>10.4f}  {r['narrow_coef']:>10.4f}  "
              f"{r['narrow_pct_change']:>5.1f}%  {r['broad_coef']:>10.4f}  {r['broad_pct_change']:>5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
