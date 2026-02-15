"""
Feature importance table and coefficient-group decomposition.

1. Feature importance: Top N LASSO features by avg |coefficient| across
   congresses, with stability analysis and trade-feature flagging.
2. Topic decomposition: Classify features into broad topic groups, sum
   coefficient weights, show what % of the total partisan signal comes
   from each group.

This directly addresses the concern that the slant shift is about trade
coverage rather than general political framing.

Outputs:
  - output/tables/feature_importance_top50.csv
  - output/tables/topic_decomposition.csv
  - output/figures/topic_decomposition.pdf
"""

import os, sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "utils"))
import pipeline_config as cfg

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
MODEL_DIR = cfg.MODEL_DIR
SPEECH_DIR = cfg.INPUT_SPEECH_DIR
FIG_DIR = cfg.FIG_DIR
TAB_DIR = cfg.TAB_DIR

# ======================================================================
# Topic classification rules
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
        "unfair trade", "world trade",
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


def classify_feature(feature, rules):
    """Classify a feature into a topic group. Returns the first matching group."""
    f = feature.lower().strip()
    tokens = f.split()

    for topic, keywords in rules.items():
        for kw in keywords:
            kw_tokens = kw.split()
            if len(kw_tokens) == 1:
                # Single token: check if any token starts with the keyword
                if any(t.startswith(kw) for t in tokens):
                    return topic
            else:
                # Multi-token: check if the feature starts with or contains the phrase
                if f.startswith(kw) or kw in f:
                    return topic
    return "Other"


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

    n_features = len(features)
    print(f"Total features: {n_features}")

    # --- Load all LASSO coefficients ---
    windows = cfg.get_windows()
    congress_coefs = {}  # congress -> coef array

    for window_congs in windows:
        cong = window_congs[-1]
        window_file = "_".join(str(c) for c in window_congs)
        model_path = MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        congress_coefs[cong] = model.coef_[0].copy()

    congresses = sorted(congress_coefs.keys())
    n_cong = len(congresses)
    print(f"Congresses: {congresses}")

    # Build coefficient matrix: (n_features, n_congresses)
    coef_mat = np.column_stack([congress_coefs[c] for c in congresses])

    # =====================================================================
    # 1. Feature importance table
    # =====================================================================
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE")
    print("=" * 60)

    avg_abs_coef = np.mean(np.abs(coef_mat), axis=1)
    max_abs_coef = np.max(np.abs(coef_mat), axis=1)
    n_active = np.sum(coef_mat != 0, axis=1)  # congresses with non-zero coef
    avg_coef = np.mean(coef_mat, axis=1)  # signed average (direction)

    # Topic classification
    topics = np.array([classify_feature(f, TOPIC_RULES) for f in features])

    # Is trade-related?
    trade_mask = topics == "Trade"

    # Build DataFrame
    feat_df = pd.DataFrame({
        "feature": features,
        "topic": topics,
        "is_trade": trade_mask,
        "avg_abs_coef": avg_abs_coef,
        "max_abs_coef": max_abs_coef,
        "avg_coef": avg_coef,
        "n_congresses_active": n_active,
        "direction": np.where(avg_coef > 0, "Republican", "Democrat"),
    })

    # Sort by avg absolute coefficient
    feat_df = feat_df.sort_values("avg_abs_coef", ascending=False).reset_index(drop=True)
    feat_df["rank"] = feat_df.index + 1

    # Save top 50
    top50 = feat_df.head(50)
    csv_path = TAB_DIR / "feature_importance_top50.csv"
    top50.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Top 50 features saved: {csv_path}")

    # Print top 30
    print(f"\n  {'Rank':<5} {'Feature':<25} {'Topic':<20} {'Dir':<10} "
          f"{'Avg |Coef|':>10} {'Active':>6} {'Trade':>5}")
    print("-" * 90)
    for _, r in top50.head(30).iterrows():
        trade_flag = "*" if r["is_trade"] else ""
        print(f"  {int(r['rank']):<5} {r['feature']:<25} {r['topic']:<20} {r['direction']:<10} "
              f"{r['avg_abs_coef']:>10.2f} {int(r['n_congresses_active']):>3}/{n_cong}  {trade_flag}")

    # Trade features in top 50
    n_trade_top50 = top50["is_trade"].sum()
    print(f"\n  Trade features in top 50: {n_trade_top50}")

    # =====================================================================
    # 2. Topic decomposition
    # =====================================================================
    print("\n" + "=" * 60)
    print("  TOPIC DECOMPOSITION")
    print("=" * 60)

    # Per-topic: sum of |coef| weight
    topic_stats = []
    for topic in sorted(set(topics)):
        mask = topics == topic
        n_feats = mask.sum()
        n_active_any = (n_active[mask] > 0).sum()  # features active in any congress
        weight = avg_abs_coef[mask].sum()
        pct = weight / avg_abs_coef.sum() * 100

        # Average across congresses
        per_cong_weights = []
        for j, c in enumerate(congresses):
            w = np.sum(np.abs(coef_mat[mask, j]))
            per_cong_weights.append(w)
        per_cong_pcts = []
        for j, c in enumerate(congresses):
            total_w = np.sum(np.abs(coef_mat[:, j]))
            if total_w > 0:
                per_cong_pcts.append(np.sum(np.abs(coef_mat[mask, j])) / total_w * 100)

        topic_stats.append({
            "topic": topic,
            "n_features": n_feats,
            "n_active": n_active_any,
            "avg_abs_weight": weight,
            "pct_total_weight": pct,
            "avg_pct_per_congress": np.mean(per_cong_pcts) if per_cong_pcts else 0,
            "min_pct_per_congress": np.min(per_cong_pcts) if per_cong_pcts else 0,
            "max_pct_per_congress": np.max(per_cong_pcts) if per_cong_pcts else 0,
        })

    topic_df = pd.DataFrame(topic_stats).sort_values("pct_total_weight", ascending=False)
    topic_df = topic_df.reset_index(drop=True)

    csv_path2 = TAB_DIR / "topic_decomposition.csv"
    topic_df.to_csv(csv_path2, index=False, float_format="%.2f")
    print(f"\n  Topic decomposition saved: {csv_path2}")

    print(f"\n  {'Topic':<25} {'Features':>8} {'Active':>6} "
          f"{'% Weight':>8} {'Min%':>6} {'Max%':>6}")
    print("-" * 70)
    for _, r in topic_df.iterrows():
        print(f"  {r['topic']:<25} {int(r['n_features']):>8} {int(r['n_active']):>6} "
              f"{r['pct_total_weight']:>7.1f}% {r['min_pct_per_congress']:>5.1f}% "
              f"{r['max_pct_per_congress']:>5.1f}%")

    # Highlight trade
    trade_row = topic_df[topic_df["topic"] == "Trade"]
    if len(trade_row) > 0:
        pct = trade_row.iloc[0]["pct_total_weight"]
        print(f"\n  >> Trade features account for {pct:.1f}% of total LASSO coefficient weight")

    # =====================================================================
    # 3. Per-congress decomposition (for stability)
    # =====================================================================
    per_cong_rows = []
    for j, c in enumerate(congresses):
        total_w = np.sum(np.abs(coef_mat[:, j]))
        for topic in sorted(set(topics)):
            mask = topics == topic
            w = np.sum(np.abs(coef_mat[mask, j]))
            per_cong_rows.append({
                "congress": c,
                "topic": topic,
                "weight": w,
                "pct": w / total_w * 100 if total_w > 0 else 0,
            })
    per_cong_df = pd.DataFrame(per_cong_rows)
    per_cong_df.to_csv(TAB_DIR / "topic_decomposition_by_congress.csv",
                       index=False, float_format="%.4f")

    # =====================================================================
    # Figure: Topic decomposition bar chart
    # =====================================================================
    # Sort topics by weight (exclude "Other" from primary display)
    plot_df = topic_df[topic_df["topic"] != "Other"].copy()
    plot_df = plot_df.sort_values("pct_total_weight", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel a: Average weight by topic
    ax = axes[0]
    colors = ["#c44e52" if t == "Trade" else "#4c72b0" for t in plot_df["topic"]]
    bars = ax.barh(range(len(plot_df)), plot_df["pct_total_weight"].values,
                   color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["topic"].values, fontsize=8)
    ax.set_xlabel("% of total LASSO coefficient weight", fontsize=9)
    ax.set_title("(a) Topic Decomposition of Partisan Vocabulary", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, plot_df["pct_total_weight"].values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7)

    # Panel b: Stacked area by congress (top 5 topics + other)
    ax = axes[1]
    top_topics = topic_df.head(6)["topic"].tolist()
    pivot = per_cong_df.pivot(index="congress", columns="topic", values="pct").fillna(0)

    # Combine small topics into "Other"
    other_cols = [c for c in pivot.columns if c not in top_topics]
    if other_cols:
        pivot["Other"] = pivot[other_cols].sum(axis=1)
        pivot = pivot.drop(columns=other_cols)

    # Reorder columns by avg weight
    col_order = [t for t in topic_df["topic"] if t in pivot.columns]
    pivot = pivot[col_order]

    # Use distinct colors
    cmap = plt.cm.Set2
    topic_colors = {t: cmap(i / len(col_order)) for i, t in enumerate(col_order)}
    topic_colors["Trade"] = "#c44e52"

    bottom = np.zeros(len(pivot))
    for col in col_order:
        vals = pivot[col].values
        ax.bar(pivot.index, vals, bottom=bottom, label=col,
               color=topic_colors.get(col, "#cccccc"), edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xlabel("Congress", fontsize=9)
    ax.set_ylabel("% of coefficient weight", fontsize=9)
    ax.set_title("(b) Topic Shares by Congress", fontsize=10)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig_path = FIG_DIR / "topic_decomposition.pdf"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
