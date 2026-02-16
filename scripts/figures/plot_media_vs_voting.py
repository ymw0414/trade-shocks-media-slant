"""
plot_media_vs_voting.py

Descriptive scatter: change in media slant vs. change in Republican
presidential vote share across commuting zones, both residualized by
division fixed effects.

Inputs:
  - data/processed/runs/exp_shvocab_cv/panel/14_regression_panel.parquet
  - data/raw/econ/election/dataverse_files/County_Level_US_Elections_Data/
        pres_elections_release.dta
  - data/raw/econ/crosswalk/cw_cty_czone/cw_cty_czone.dta

Outputs:
  - output/figures/media_vs_voting.pdf / .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
from scipy import stats
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

PANEL_PATH = (BASE_DIR / "data" / "processed" / "runs" / "exp_shvocab_cv"
              / "panel" / "14_regression_panel.parquet")
ELECTION_PATH = (BASE_DIR / "data" / "raw" / "econ" / "election"
                 / "dataverse_files" / "County_Level_US_Elections_Data"
                 / "pres_elections_release.dta")
CROSSWALK_PATH = (BASE_DIR / "data" / "raw" / "econ" / "crosswalk"
                  / "cw_cty_czone" / "cw_cty_czone.dta")
FIG_DIR = BASE_DIR / "output" / "figures"

PRE_YEARS = [1988, 1992]
POST_YEARS = [1996, 2000, 2004]

# Census division mapping
STATE_TO_DIV = {
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1,
    "NJ": 2, "NY": 2, "PA": 2,
    "IL": 3, "IN": 3, "MI": 3, "OH": 3, "WI": 3,
    "IA": 4, "KS": 4, "MN": 4, "MO": 4, "NE": 4, "ND": 4, "SD": 4,
    "DE": 5, "FL": 5, "GA": 5, "MD": 5, "NC": 5, "SC": 5, "VA": 5,
    "WV": 5, "DC": 5,
    "AL": 6, "KY": 6, "MS": 6, "TN": 6,
    "AR": 7, "LA": 7, "OK": 7, "TX": 7,
    "AZ": 8, "CO": 8, "ID": 8, "MT": 8, "NV": 8, "NM": 8, "UT": 8, "WY": 8,
    "AK": 9, "CA": 9, "HI": 9, "OR": 9, "WA": 9,
}


def load_election_by_cz():
    """Load presidential election data, aggregate to CZ level, return ΔR."""
    # --- Election data ---
    elec = pd.read_stata(ELECTION_PATH)
    elec["fips"] = pd.to_numeric(elec["fips"], errors="coerce")
    elec = elec.dropna(subset=["fips"])
    elec["fips"] = elec["fips"].astype(int)
    elec["rep_votes"] = pd.to_numeric(elec["republican_raw_votes"], errors="coerce")
    elec["dem_votes"] = pd.to_numeric(elec["democratic_raw_votes"], errors="coerce")
    elec["total_2p"] = elec["rep_votes"] + elec["dem_votes"]
    elec = elec[elec["total_2p"] > 0].copy()

    # --- County -> CZ crosswalk ---
    cw = pd.read_stata(CROSSWALK_PATH)
    cw = cw.rename(columns={"cty_fips": "fips", "czone": "cz"})
    cw["fips"] = cw["fips"].astype(int)
    cw["cz"] = cw["cz"].astype(int)

    elec = elec.merge(cw, on="fips", how="inner")

    # --- Aggregate to CZ-period level (vote-weighted) ---
    results = []
    for label, years in [("pre", PRE_YEARS), ("post", POST_YEARS)]:
        sub = elec[elec["election_year"].isin(years)]
        agg = (sub.groupby("cz")
               .agg(rep=("rep_votes", "sum"), total=("total_2p", "sum"))
               .reset_index())
        agg[f"rep_share_{label}"] = agg["rep"] / agg["total"]
        results.append(agg[["cz", f"rep_share_{label}"]])

    cz_vote = results[0].merge(results[1], on="cz", how="inner")
    cz_vote["d_rep_share"] = cz_vote["rep_share_post"] - cz_vote["rep_share_pre"]

    print(f"  Election data: {len(cz_vote)} CZs with pre+post vote shares")
    return cz_vote


def load_media_change():
    """Compute CZ-level change in Share R from newspaper panel."""
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel[
        panel["cz"].notna()
        & panel["vulnerability1990_scaled"].notna()
        & panel["manushare1990"].notna()
    ].copy()
    panel["cz"] = panel["cz"].astype(int)

    # Pre/post indicator
    panel["post"] = (panel["year"] >= 1994).astype(int)

    # Add division from state
    panel["division"] = panel["state"].map(STATE_TO_DIV)

    # Newspaper-level change
    nw = []
    for paper, pdata in panel.groupby("paper"):
        pre = pdata.loc[pdata["post"] == 0, "ext_R"].mean()
        post = pdata.loc[pdata["post"] == 1, "ext_R"].mean()
        cz = pdata["cz"].iloc[0]
        div = pdata["division"].iloc[0]
        nw.append({"paper": paper, "cz": cz, "division": div,
                    "d_ext_R": post - pre})
    nw_df = pd.DataFrame(nw)

    # CZ-level average (across newspapers in the CZ)
    cz_media = (nw_df.groupby("cz")
                .agg(d_ext_R=("d_ext_R", "mean"),
                     division=("division", "first"),
                     n_papers=("paper", "count"))
                .reset_index())

    print(f"  Media data: {len(cz_media)} CZs, {nw_df.shape[0]} newspapers")
    return cz_media


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)

    # 1. Load data
    cz_vote = load_election_by_cz()
    cz_media = load_media_change()

    # 2. Merge
    df = cz_media.merge(cz_vote, on="cz", how="inner")
    print(f"\n  Merged: {len(df)} CZs")

    # 3. Residualize by division FE
    for col in ["d_ext_R", "d_rep_share"]:
        div_means = df.groupby("division")[col].transform("mean")
        df[f"r_{col}"] = df[col] - div_means

    # 4. Correlation
    r, p = stats.pearsonr(df["r_d_ext_R"], df["r_d_rep_share"])
    slope, intercept, _, p_ols, se = stats.linregress(
        df["r_d_ext_R"], df["r_d_rep_share"])
    print(f"\n  Correlation (division-residualized):")
    print(f"    r = {r:.3f}, p = {p:.4f}")
    print(f"    slope = {slope:.3f} (SE = {se:.3f}), p = {p_ols:.4f}")

    # 5. Scatter plot
    x = df["r_d_ext_R"].values
    y = df["r_d_rep_share"].values

    fig, ax = plt.subplots(figsize=(6, 5))

    C_R = "#bf6b63"
    ax.scatter(x, y, s=45, color="#555555", alpha=0.6,
               edgecolors="white", linewidth=0.4, zorder=3)

    # OLS fit line + CI band
    x_grid = np.linspace(x.min() - 0.01, x.max() + 0.01, 200)
    y_hat = intercept + slope * x_grid
    n = len(x)
    ss_x = np.sum((x - x.mean()) ** 2)
    resid_var = np.sum((y - (intercept + slope * x)) ** 2) / (n - 2)
    se_pred = np.sqrt(resid_var * (1.0 / n + (x_grid - x.mean()) ** 2 / ss_x))

    ax.fill_between(x_grid, y_hat - 1.96 * se_pred, y_hat + 1.96 * se_pred,
                     color=C_R, alpha=0.12, zorder=1)
    ax.plot(x_grid, y_hat, color=C_R, linewidth=1.8, zorder=4)

    ax.axhline(0, color="gray", linewidth=0.4, linestyle="-", zorder=0)
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="-", zorder=0)

    # Annotation
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    ax.annotate(
        f"$r$ = {r:.2f}{stars}\nslope = {slope:.3f} (SE = {se:.3f})",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

    ax.set_xlabel(r"$\Delta$ Share R-Leaning Articles (residualized)", fontsize=11)
    ax.set_ylabel(r"$\Delta$ Republican Vote Share (residualized)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    fig.tight_layout(pad=0.5)
    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"media_vs_voting.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05,
                    facecolor="white")
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
