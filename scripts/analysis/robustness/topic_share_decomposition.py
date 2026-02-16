"""
Topic-level share decomposition.

Classify articles by topic using keyword matching on features,
then compute topic-specific share of R-leaning articles (ext_R) and run DiD.

Key question: Is the increase in R-leaning share uniform across topics,
or concentrated in specific domains (e.g., trade)?

Approach:
1. Load vocabulary and classify features into 10 topic categories
2. For each article, check which topic features are present (non-zero frequency)
3. Compute topic-specific ext_R at newspaper-year level
4. Run DiD on each topic's ext_R

Outputs:
  - output/tables/topic_share_decomposition.csv
  - output/figures/topic_share_decomposition.pdf
"""

import os, sys, gc, time
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
SPEECH_DIR = cfg.INPUT_SPEECH_DIR
NEWS_FEATURES_DIR = cfg.INPUT_NEWS_DIR
NEWS_DIR = cfg.NEWS_DIR
LABEL_DIR = cfg.NEWSPAPER_LABELS
PANEL_PATH = cfg.PANEL_DIR / "14_regression_panel.parquet"
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

NAFTA_YEAR = 1994
END_YEAR = 2004

# ======================================================================
# Topic classification rules (stemmed to match Porter-stemmed features)
# ======================================================================
TOPIC_RULES = {
    "Tax & fiscal": [
        "tax", "spend", "budget", "deficit", "surplus", "fiscal",
        "debt", "revenue", "appropriat", "taxpay", "deduct",
        "ira", "estate tax", "death tax", "capit gain",
        "tax cut", "tax relief", "tax break", "tax incent",
        "tax credit", "tax rate", "tax reform", "tax burden",
        "billion dollar", "trillion",
    ],
    "Healthcare": [
        "health", "medicar", "medicaid", "insur", "hmo",
        "prescri", "drug", "patient", "hospit", "doctor",
        "nurs", "diseas", "mental health", "managed care",
        "health care", "health insur",
    ],
    "Social policy": [
        "welfar", "poverty", "poor", "homeless", "food stamp",
        "social secur", "retir", "pension", "minimum wage",
        "unemploy", "child", "famili", "abus", "domest violenc",
        "senior", "elder", "disabl",
    ],
    "Defense & foreign": [
        "defens", "militari", "troop", "war", "weapon",
        "nuclear", "missile", "veteran", "nato", "arm",
        "terror", "homeland", "secur", "intellig",
        "foreign", "diplomat", "sanction", "nation secur",
    ],
    "Trade": [
        "trade", "tariff", "nafta", "export", "gatt", "wto",
        "quota", "dumping", "free trade", "fair trade",
        "unfair trade", "world trade", "import",
    ],
    "Environment & energy": [
        "environ", "pollut", "clean air", "clean water",
        "climat", "emiss", "oil", "energi", "nuclear",
        "wilderness", "forest", "endangered",
    ],
    "Education": [
        "educ", "school", "teacher", "student", "colleg",
        "univers", "scholarship", "classroom", "literaci",
    ],
    "Crime & justice": [
        "crime", "prison", "sentenc", "drug", "polic",
        "gun", "firearm", "violenc", "victim", "crimin",
        "death penalti", "mandatori",
    ],
    "Government & procedure": [
        "amendment", "committe", "subcommitte", "legisl",
        "bipartisan", "republican", "democrat", "partisan",
        "speaker", "chairman", "confer", "floor",
        "motion", "rule", "vote", "bill", "act",
        "gentleman", "gentlewoman", "colleague",
        "yield", "reserv", "balanc",
    ],
    "Civil rights & identity": [
        "civil right", "discrimin", "affirm", "minor",
        "women", "gender", "race", "racial", "immigr",
        "abort", "prayer", "religion", "faith",
    ],
}

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


def classify_feature(feature, rules):
    """Classify a feature into a topic group. Returns first match or 'Other'."""
    f = feature.lower().strip()
    tokens = f.split()
    for topic, keywords in rules.items():
        for kw in keywords:
            kw_tokens = kw.split()
            if len(kw_tokens) == 1:
                if any(t.startswith(kw) for t in tokens):
                    return topic
            else:
                if f.startswith(kw) or kw in f:
                    return topic
    return "Other"


def build_topic_column_map(features, rules):
    """Map each topic to the set of column indices in the feature matrix."""
    topic_cols = {topic: [] for topic in rules}
    topic_cols["Other"] = []

    for i, feat in enumerate(features):
        topic = classify_feature(feat, rules)
        topic_cols[topic].append(i)

    # Print summary
    for topic, cols in sorted(topic_cols.items(), key=lambda x: -len(x[1])):
        print(f"  {topic:<25s}: {len(cols):>5d} features")

    return topic_cols


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load vocabulary ---
    print("Loading vocabulary ...")
    vec = joblib.load(SPEECH_DIR / "05_vectorizer.joblib")
    all_features = vec.get_feature_names_out()

    shared_vocab_path = MODEL_DIR / "06_shared_vocab_mask.npy"
    shared_vocab_mask = np.load(shared_vocab_path) if shared_vocab_path.exists() else None
    if shared_vocab_mask is not None:
        features = all_features[shared_vocab_mask]
    else:
        features = all_features
    print(f"  {len(features)} features after shared vocab filter")

    # --- Build topic -> column index map ---
    print("\nClassifying features into topics ...")
    topic_cols = build_topic_column_map(features, TOPIC_RULES)

    # All topics except "Other" for the analysis
    topics = [t for t in TOPIC_RULES.keys()]  # ordered as defined

    # --- Process each congress ---
    print("\nProcessing articles by congress ...")
    congresses = [w[-1] for w in cfg.get_windows()]
    all_chunks = []
    pipeline_start = time.time()

    for cong in congresses:
        t0 = time.time()

        # Load article metadata
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        if not label_path.exists():
            print(f"  Congress {cong}: label file not found, skipping")
            continue
        meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])

        # Apply sample indices if they exist
        sample_idx_path = NEWS_FEATURES_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)

        # Load slant scores
        slant_path = NEWS_DIR / f"08_article_slant_cong_{cong}.parquet"
        if not slant_path.exists():
            print(f"  Congress {cong}: slant file not found, skipping")
            continue
        slant = pd.read_parquet(slant_path, columns=["net_slant"])
        assert len(meta) == len(slant), f"Congress {cong}: meta/slant size mismatch"

        # Filter to news articles
        is_news = meta["is_news"].values
        news_idx = np.where(is_news)[0]

        # Load feature matrix
        features_path = NEWS_FEATURES_DIR / f"07_newspaper_features_cong_{cong}.npz"
        X = sp.load_npz(features_path)
        if shared_vocab_mask is not None:
            X = X[:, shared_vocab_mask]

        # Filter to news articles
        X_news = X[news_idx]
        net_slant_news = slant["net_slant"].values[news_idx]
        paper_news = meta["paper"].values[news_idx]
        year_news = meta["year"].values[news_idx]
        n_news = len(news_idx)

        # R-leaning indicator
        is_R = (net_slant_news > 0).astype(np.int8)

        # For each topic, check if article mentions any feature in that topic
        topic_flags = {}
        for topic in topics:
            cols = topic_cols[topic]
            if len(cols) == 0:
                topic_flags[topic] = np.zeros(n_news, dtype=np.int8)
            else:
                # Sum across topic columns; non-zero means article mentions topic
                topic_sum = np.array(X_news[:, cols].sum(axis=1)).ravel()
                topic_flags[topic] = (topic_sum > 0).astype(np.int8)

        # Build a DataFrame for aggregation
        df_art = pd.DataFrame({
            "paper": paper_news,
            "year": year_news,
            "is_R": is_R,
        })
        for topic in topics:
            safe_name = topic.replace(" & ", "_").replace(" ", "_").lower()
            df_art[f"topic_{safe_name}"] = topic_flags[topic]
            df_art[f"R_in_{safe_name}"] = is_R * topic_flags[topic]

        # Also create "any_topic" (mentions at least one specific topic)
        any_topic = np.zeros(n_news, dtype=np.int8)
        for topic in topics:
            any_topic |= topic_flags[topic]
        df_art["has_topic"] = any_topic
        df_art["no_topic"] = 1 - any_topic
        df_art["R_in_no_topic"] = is_R * (1 - any_topic)

        # Aggregate to newspaper-year
        grp = df_art.groupby(["paper", "year"])
        agg = grp.agg(
            n_articles=("is_R", "count"),
            n_R=("is_R", "sum"),
            ext_R=("is_R", "mean"),
        ).reset_index()

        # Topic-specific ext_R
        for topic in topics:
            safe_name = topic.replace(" & ", "_").replace(" ", "_").lower()
            topic_col = f"topic_{safe_name}"
            r_col = f"R_in_{safe_name}"

            # Number of articles mentioning this topic
            agg[f"n_{safe_name}"] = grp[topic_col].sum().values
            # Number of R-leaning articles mentioning this topic
            agg[f"nR_{safe_name}"] = grp[r_col].sum().values
            # Share R among articles mentioning this topic
            agg[f"ext_R_{safe_name}"] = (
                agg[f"nR_{safe_name}"] / agg[f"n_{safe_name}"]
            ).replace([np.inf, -np.inf], np.nan)

        # Non-topic articles
        agg["n_no_topic"] = grp["no_topic"].sum().values
        agg["nR_no_topic"] = grp["R_in_no_topic"].sum().values
        agg["ext_R_no_topic"] = (
            agg["nR_no_topic"] / agg["n_no_topic"]
        ).replace([np.inf, -np.inf], np.nan)

        all_chunks.append(agg)

        elapsed = time.time() - t0
        # Summary for this congress
        n_trade = topic_flags.get("Trade", np.zeros(1)).sum()
        pct_trade = n_trade / n_news * 100 if n_news > 0 else 0
        print(f"  Congress {cong}: {n_news:>10,} news articles, "
              f"trade={n_trade:>8,} ({pct_trade:.1f}%), "
              f"{elapsed:.1f}s")

        del X, X_news, meta, slant, df_art, topic_flags
        gc.collect()

    # --- Combine all congresses ---
    print("\nCombining across congresses ...")
    panel_topic = pd.concat(all_chunks, ignore_index=True)
    panel_topic = panel_topic.sort_values(["paper", "year"]).reset_index(drop=True)
    print(f"  Panel: {len(panel_topic)} newspaper-year obs")

    # --- Merge with regression panel for CZ/vulnerability ---
    print("Merging with regression panel ...")
    reg = pd.read_parquet(PANEL_PATH)
    reg = reg[reg["cz"].notna() & reg["vulnerability1990_scaled"].notna()].copy()
    reg = reg[reg["year"] <= END_YEAR].copy()

    # Merge on paper + year
    merged = panel_topic.merge(
        reg[["paper", "year", "cz", "fips", "vulnerability1990_scaled",
             "manushare1990", "china_shock"]],
        on=["paper", "year"], how="inner"
    )
    print(f"  Merged panel: {len(merged)} obs")

    # Build regression variables
    merged["state_fips"] = (merged["fips"] // 1000).astype(int)
    merged["division"] = merged["state_fips"].map(STATE_TO_DIVISION)
    merged["paper_id"] = merged["paper"].astype("category").cat.codes
    merged["post"] = (merged["year"] >= NAFTA_YEAR).astype(int)
    merged["vuln_x_post"] = merged["vulnerability1990_scaled"] * merged["post"]

    years = sorted(merged["year"].unique())
    base_yr = years[0]
    for yr in years:
        if yr == base_yr:
            continue
        merged[f"manu_{yr}"] = (merged["year"] == yr).astype(float) * merged["manushare1990"].fillna(0)
        merged[f"china_{yr}"] = (merged["year"] == yr).astype(float) * merged["china_shock"].fillna(0)

    manu_str = " + ".join([f"manu_{yr}" for yr in years if yr != base_yr])
    china_str = " + ".join([f"china_{yr}" for yr in years if yr != base_yr])

    # --- Run DiD on each topic's ext_R ---
    print("\nRunning DiD regressions ...")
    fml_base = "vuln_x_post + {china} + {manu} | paper_id + year + division^year".format(
        china=china_str, manu=manu_str
    )

    # Build list of outcomes
    outcome_list = [("ext_R", "All articles", "all")]
    for topic in topics:
        safe_name = topic.replace(" & ", "_").replace(" ", "_").lower()
        outcome_list.append((f"ext_R_{safe_name}", topic, safe_name))
    outcome_list.append(("ext_R_no_topic", "No specific topic", "no_topic"))

    results = []
    print(f"\n  {'Outcome':<30s} {'N':>6s} {'Coef':>8s} {'SE':>8s} {'p':>8s} {'Share':>6s}")
    print("  " + "-" * 72)

    for depvar, label, key in outcome_list:
        if depvar not in merged.columns:
            continue
        subset = merged[merged[depvar].notna()].copy()
        n_obs = len(subset)
        if n_obs < 100:
            print(f"  {label:<30s} {n_obs:>6d}  (too few obs, skipping)")
            continue

        fml = f"{depvar} ~ {fml_base}"
        try:
            m = pf.feols(fml, data=subset, vcov={"CRV1": "cz"})
            t = m.tidy().loc["vuln_x_post"]
            coef = t["Estimate"]
            se = t["Std. Error"]
            p = t["Pr(>|t|)"]
        except Exception as e:
            print(f"  {label:<30s}  ERROR: {e}")
            continue

        # Average share of articles in this topic
        if key == "all":
            avg_share = 1.0
        elif key == "no_topic":
            avg_share = merged["n_no_topic"].sum() / merged["n_articles"].sum()
        else:
            n_col = f"n_{key}"
            avg_share = merged[n_col].sum() / merged["n_articles"].sum() if n_col in merged.columns else np.nan

        print(f"  {label:<30s} {n_obs:>6d} {coef:>8.4f} {se:>8.4f} {p:>8.4f} {avg_share:>6.1%}")

        results.append({
            "topic": label,
            "key": key,
            "n_obs": n_obs,
            "coef": coef,
            "se": se,
            "p_value": p,
            "avg_article_share": avg_share,
        })

    # --- Save results ---
    res_df = pd.DataFrame(results)
    csv_path = TAB_DIR / "topic_share_decomposition.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nResults saved: {csv_path}")

    # --- Plot ---
    plot_data = res_df[res_df["key"] != "all"].copy()
    plot_data = plot_data.sort_values("coef", ascending=True)
    all_coef = res_df[res_df["key"] == "all"]["coef"].values[0]

    # Standard academic palette
    SIG_COLOR = "#2d2d2d"     # dark navy for significant
    INSIG_COLOR = "#b0b0b0"   # light gray for not significant
    TRADE_COLOR = "#bf6b63"   # crimson for trade (highlight)
    REF_COLOR = "#7a7a7a"     # reference line

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y_pos = np.arange(len(plot_data))
    coefs = plot_data["coef"].values
    ses = plot_data["se"].values
    pvals = plot_data["p_value"].values
    is_trade = plot_data["key"].values == "trade"
    is_sig = pvals < 0.05

    # Light horizontal gridlines for readability
    for y in y_pos:
        ax.axhline(y, color="#f0f0f0", linewidth=0.6, zorder=0)

    # Overall coefficient reference line
    ax.axvline(all_coef, color=REF_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=1)
    ax.axvline(0, color="black", linewidth=0.6, zorder=1)

    # Plot each topic individually
    for i, y in enumerate(y_pos):
        c, se_i, sig, trade = coefs[i], ses[i], is_sig[i], is_trade[i]

        if trade:
            color = TRADE_COLOR
            marker = "D"
        elif sig:
            color = SIG_COLOR
            marker = "o"
        else:
            color = INSIG_COLOR
            marker = "o"

        ms = 7
        elw = 1.4
        alpha = 0.9

        ax.errorbar(c, y, xerr=1.96 * se_i,
                    fmt=marker, color=color, markersize=ms,
                    capsize=3.5, capthick=elw,
                    ecolor=color, elinewidth=elw,
                    markeredgecolor="white", markeredgewidth=0.8,
                    alpha=alpha, zorder=3)

        # Article share annotation on the right
        share = plot_data["avg_article_share"].values[i]
        ax.text(0.205, y, f"{share:.0%}",
                fontsize=7.5, color="#888888", va="center", ha="right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["topic"], fontsize=9)
    ax.set_xlabel("DiD coefficient on Share R-leaning", fontsize=10)
    ax.set_xlim(-0.09, 0.215)

    # Custom legend — upper left
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SIG_COLOR,
               markersize=7, markeredgecolor="white", markeredgewidth=0.8,
               label="$p < 0.05$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INSIG_COLOR,
               markersize=6, markeredgecolor="white", markeredgewidth=0.8,
               alpha=0.7, label="Not significant"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=TRADE_COLOR,
               markersize=7, markeredgecolor="white", markeredgewidth=0.8,
               label="Trade"),
        Line2D([0], [0], color=REF_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Overall = {all_coef:.3f}"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left",
              framealpha=0.95, edgecolor="#dddddd")

    # Article share column header
    ax.text(0.205, len(y_pos) - 0.3, "share", fontsize=7.5,
            color="#999999", va="bottom", ha="right",
            fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", length=3, width=0.5)

    fig.tight_layout()
    fig_path = FIG_DIR / "topic_share_decomposition.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    total_time = time.time() - pipeline_start
    print(f"\nTotal time: {total_time:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
