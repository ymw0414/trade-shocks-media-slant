"""
map_nafta_exposure.py

Choropleth maps of NAFTA vulnerability at the county and
commuting-zone level for the contiguous US, with sample newspaper
locations overlaid on the CZ map.

Uses sextile (6-quantile) bins, Albers Equal Area Conic projection,
and state boundary outlines.

Outputs:
  - output/figures/nafta_exposure_county.png
  - output/figures/nafta_exposure_cz.png  (with newspaper markers)
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

# LaTeX-style fonts
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
sys.path.insert(0, str(BASE_DIR / "scripts" / "nlp"))
import pipeline_config as cfg

SHP_PATH = (BASE_DIR / "data" / "raw" / "econ" / "shapefiles"
            / "cb_2020_us_county_20m" / "cb_2020_us_county_20m.shp")
CZ_XW_PATH = (BASE_DIR / "data" / "raw" / "econ" / "crosswalk"
              / "cw_cty_czone" / "cw_cty_czone.dta")

# Use the Choi et al. PUBLIC replication data (all 3,026 counties, pre-balanced-panel)
VULN_DATA = (BASE_DIR / "replication" / "Replication Project" / "Replication Project"
             / "Choi et al ARE 2024_Replication Package"
             / "nafta_politics_replication_submission_PUBLIC"
             / "data" / "working_data" / "vulnerability.dta")
# Fallback: our processed data
COUNTY_DATA = (BASE_DIR / "data" / "processed" / "econ"
               / "12_nafta_vars_county.parquet")
CZ_DATA = (BASE_DIR / "data" / "processed" / "econ"
           / "12_nafta_vars_cz.parquet")
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
OUT_DIR = BASE_DIR / "output" / "figures"

# Territories and non-contiguous states to exclude
EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}

# Albers Equal Area Conic for contiguous US (EPSG:5070)
ALBERS_CRS = "EPSG:5070"

# Sequential color palette: 6 bins — warm amber-to-brown
BIN_COLORS = ["#fef0d9", "#fdcc8a", "#fc8d59", "#e34a33", "#b30000", "#7f0000"]
NO_DATA_COLOR = "#ededed"

# Sample CZ highlight style
SAMPLE_EDGE_COLOR = "#1b1b1b"
SAMPLE_EDGE_LW = 1.2

# Newspaper dot style
DOT_COLOR = "#1a1a1a"
DOT_EDGE = "white"
DOT_SIZE = 18
DOT_EDGE_LW = 0.7


def load_county_shapes():
    """Load Census county shapefile, filter to contiguous US, reproject to Albers."""
    gdf = gpd.read_file(SHP_PATH)
    gdf = gdf[~gdf["STATEFP"].isin(EXCLUDE_STATEFP)].copy()
    gdf["county"] = (gdf["STATEFP"].astype(int) * 1000
                     + gdf["COUNTYFP"].astype(int))
    gdf = gdf.to_crs(ALBERS_CRS)
    return gdf


def load_state_outlines(county_gdf):
    """Dissolve county shapefile by state to get state boundary lines."""
    return county_gdf.dissolve(by="STATEFP").reset_index()


def quantile_bins(values, k=6):
    """Compute k quantile bin edges from non-NaN values."""
    v = values.dropna().sort_values()
    edges = list(np.quantile(v, np.linspace(0, 1, k + 1)))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12

    def fmt(val):
        if val < 0.001:
            return f"{val:.4f}"
        elif val < 0.1:
            return f"{val:.3f}"
        elif val < 10:
            return f"{val:.2f}"
        else:
            return f"{val:.1f}"

    labels = [f"{fmt(edges[i])} \u2013 {fmt(edges[i+1])}" for i in range(k)]
    return edges, labels


def get_sample_newspapers(county_gdf):
    """Get sample CZ set and newspaper county-centroid locations (in Albers)."""
    panel = pd.read_parquet(PANEL_PATH)
    news = panel.drop_duplicates(subset="paper")[["paper", "fips", "cz", "state"]].copy()
    # Exclude Alaska/Hawaii (not on contiguous US map)
    news = news[~news["state"].isin(["AK", "HI"])]
    news["cz"] = news["cz"].astype(int)
    news["fips"] = news["fips"].astype(int)

    sample_czs = set(news["cz"].unique())
    n_papers = len(news)

    # Get county centroids (already in Albers)
    centroids = county_gdf[["county", "geometry"]].copy()
    centroids["cx"] = centroids.geometry.centroid.x
    centroids["cy"] = centroids.geometry.centroid.y

    news = news.merge(centroids[["county", "cx", "cy"]],
                      left_on="fips", right_on="county", how="left")

    # Small jitter for papers sharing the same county
    rng = np.random.default_rng(42)
    dups = news.duplicated(subset="fips", keep=False)
    if dups.any():
        # Jitter ~15km in Albers units (meters)
        news.loc[dups, "cx"] += rng.uniform(-15000, 15000, size=dups.sum())
        news.loc[dups, "cy"] += rng.uniform(-15000, 15000, size=dups.sum())

    print(f"  Sample: {n_papers} newspapers in {len(sample_czs)} CZs")
    return sample_czs, n_papers, news[["paper", "cx", "cy"]].dropna()


def plot_map(gdf, state_gdf, col, out_path, bin_edges, bin_labels,
             sample_czs=None, n_papers=None, news_locs=None):
    """Draw a quantile-bin choropleth with state outlines on Albers projection.

    If sample_czs is provided, those CZs get a thick dark border.
    If news_locs is provided, individual newspapers are plotted as small dots.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # Build colormap from discrete colors
    cmap = mcolors.ListedColormap(BIN_COLORS[:len(bin_labels)])
    norm = mcolors.BoundaryNorm(bin_edges, cmap.N)

    # No-data background
    gdf[gdf[col].isna()].plot(ax=ax, color=NO_DATA_COLOR, edgecolor="none")

    # Choropleth (all CZs, thin edge)
    has_data = gdf[gdf[col].notna()]
    has_data.plot(
        ax=ax, column=col, cmap=cmap, norm=norm,
        edgecolor="face", linewidth=0.05,
    )

    # Highlight sample CZs with thick dark border
    if sample_czs is not None and "cz" in gdf.columns:
        sample_gdf = gdf[gdf["cz"].isin(sample_czs)]
        sample_gdf.boundary.plot(
            ax=ax, edgecolor=SAMPLE_EDGE_COLOR,
            linewidth=SAMPLE_EDGE_LW, zorder=4,
        )

    # Newspaper location dots (one per paper)
    if news_locs is not None and len(news_locs) > 0:
        ax.scatter(
            news_locs["cx"], news_locs["cy"],
            s=DOT_SIZE, c=DOT_COLOR, edgecolors=DOT_EDGE,
            linewidths=DOT_EDGE_LW, zorder=5, clip_on=True,
        )

    # State outlines
    state_gdf.boundary.plot(ax=ax, edgecolor="#555555", linewidth=0.5)

    # Legend
    patches = [mpatches.Patch(facecolor=c, edgecolor="#aaaaaa", linewidth=0.4, label=l)
               for l, c in zip(bin_labels, BIN_COLORS[:len(bin_labels)])]
    patches.append(mpatches.Patch(facecolor=NO_DATA_COLOR, edgecolor="#aaaaaa",
                                  linewidth=0.4, label="No data"))
    if sample_czs is not None:
        # Border legend entry
        patches.append(mpatches.Patch(facecolor="none", edgecolor=SAMPLE_EDGE_COLOR,
                                      linewidth=1.6,
                                      label=f"Sample CZs ($N = {len(sample_czs)}$)"))
        # Dot legend entry
        if n_papers:
            patches.append(Line2D([0], [0], marker="o", color="none",
                                  markerfacecolor=DOT_COLOR, markeredgecolor=DOT_EDGE,
                                  markeredgewidth=DOT_EDGE_LW, markersize=5,
                                  label=f"Newspaper ($N = {n_papers}$)"))
    leg = ax.legend(
        handles=patches, loc="lower left", fontsize=9,
        frameon=True, framealpha=0.95, edgecolor="#999999",
        handlelength=1.2, handleheight=1.0, borderpad=0.8,
        title="Vulnerability (sextile)", title_fontsize=10,
    )
    leg.get_frame().set_linewidth(0.5)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counties = load_county_shapes()
    states = load_state_outlines(counties)

    # --- County map (no newspaper markers) ---
    print("County map ...")
    if VULN_DATA.exists():
        vuln_all = pd.read_stata(VULN_DATA, convert_dates=False)
        v90 = vuln_all[vuln_all["year"] == 1990][["county", "county_vul_own"]].copy()
        col = "county_vul_own"
        print(f"  Using replication data: {len(v90)} counties")
    else:
        cty = pd.read_parquet(COUNTY_DATA)
        v90 = cty[cty["year"] == 1990][["county", "vulnerability1990_scaled"]].copy()
        col = "vulnerability1990_scaled"
        print(f"  Using processed data: {len(v90)} counties")

    merged = counties.merge(v90, on="county", how="left")
    bin_edges, bin_labels = quantile_bins(merged[col], k=6)
    print(f"  Bin edges: {[f'{e:.4f}' for e in bin_edges]}")

    plot_map(
        merged, states, col,
        OUT_DIR / "nafta_exposure_county.png",
        bin_edges, bin_labels,
    )

    # --- CZ map (with newspaper markers) ---
    print("CZ map ...")
    cz_xw = pd.read_stata(CZ_XW_PATH, convert_dates=False)
    cz_xw = cz_xw.rename(columns={"cty_fips": "county", "czone": "cz"})
    cz_xw["county"] = cz_xw["county"].astype(int)
    cz_xw["cz"] = cz_xw["cz"].astype(int)

    cz_shapes = counties.merge(cz_xw, on="county", how="left")
    cz_shapes = cz_shapes[cz_shapes["cz"].notna()].copy()
    cz_shapes["cz"] = cz_shapes["cz"].astype(int)
    cz_dissolved = cz_shapes.dissolve(by="cz").reset_index()

    czd = pd.read_parquet(CZ_DATA)
    czd1990 = czd[czd["year"] == 1990][["czone", "vulnerability1990_scaled"]].copy()
    czd1990 = czd1990.rename(columns={"czone": "cz"})

    cz_col = "vulnerability1990_scaled"
    cz_merged = cz_dissolved.merge(czd1990, on="cz", how="left")

    cz_edges, cz_labels = quantile_bins(cz_merged[cz_col], k=6)
    print(f"  Bin edges: {[f'{e:.4f}' for e in cz_edges]}")

    # Load sample CZs and newspaper locations
    sample_czs, n_papers, news_locs = get_sample_newspapers(counties)

    plot_map(
        cz_merged, states, cz_col,
        OUT_DIR / "nafta_exposure_cz.png",
        cz_edges, cz_labels,
        sample_czs=sample_czs,
        n_papers=n_papers,
        news_locs=news_locs,
    )

    print("Done.")


if __name__ == "__main__":
    main()
