"""
plot_measurement_precision.py

Diagnostic figures demonstrating that aggregation improves measurement precision:

(a) Legislator-level classification accuracy:
    Individual speeches have ~70% accuracy, but aggregating all speeches by a
    legislator yields >90% accuracy -- confirming the LASSO captures genuine
    partisan signal despite noise in individual speech classification.

(b) Newspaper slant convergence:
    Article-level slant scores are noisy, but the newspaper-year mean converges
    rapidly as article count grows. With typical newspaper sizes (10-30K articles),
    the standard error of mean slant is negligible relative to cross-newspaper
    variation and the treatment effect.

Outputs:
  - output/figures/accuracy_convergence.pdf
  - output/figures/slant_convergence.pdf
  - output/tables/measurement_precision.csv
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
})
from pathlib import Path

# Pipeline config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nlp"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import text_analyzer  # noqa: F401 (needed for vectorizer unpickling)
import pipeline_config as cfg

BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR",
                                "C:/Users/ymw04/Dropbox/shifting_slant"))
FIG_DIR = BASE_DIR / "output" / "figures"
TABLE_DIR = BASE_DIR / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LABEL_DIR = cfg.NEWSPAPER_LABELS
NEWS_DIR = cfg.NEWS_DIR

# ====================================================================
# PART 1: Legislator-level accuracy convergence
# ====================================================================
def compute_legislator_accuracy():
    """
    For each congress window, predict on all speeches, then aggregate by
    legislator at varying numbers of speeches to show accuracy convergence.
    """
    print("=" * 72)
    print("PART 1: Legislator-level accuracy convergence")
    print("=" * 72)

    # Load speech data
    print("\nLoading speech feature matrix ...")
    X_all = sp.load_npz(cfg.INPUT_SPEECH_DIR / "05_feature_matrix.npz")
    meta = pd.read_parquet(cfg.INPUT_SPEECH_DIR / "05_feature_meta.parquet")

    # Apply shared vocab mask
    mask_path = cfg.MODEL_DIR / "06_shared_vocab_mask.npy"
    if mask_path.exists():
        shared_vocab_mask = np.load(mask_path)
        X_all = X_all[:, shared_vocab_mask]
        print(f"  Applied shared vocab mask: {X_all.shape[1]:,} features")

    PARTY_COL = "party" if "party" in meta.columns else "party_code"
    CORE_ONLY = cfg.CONFIG.get("partisan_core_only", False)

    # Subsample sizes to test
    k_values = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    n_bootstrap = 200  # bootstrap repetitions per k

    rng = np.random.default_rng(42)
    all_results = []
    congress_summary = []

    for window_congs in cfg.get_windows():
        cong = window_congs[-1]
        window_file = "_".join(str(c) for c in window_congs)
        model_path = cfg.MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
        if not model_path.exists():
            continue

        print(f"\n  Congress {cong} ...")
        model = joblib.load(model_path)

        # Select speeches
        mask_window = meta["congress_int"].isin(window_congs)
        if CORE_ONLY:
            mask_rep = meta["label_rep_core"] == 1
            mask_dem = meta["label_dem_core"] == 1
            mask_train = mask_window & (mask_rep | mask_dem)
            idx = meta.index[mask_train].values
            y_true = meta.loc[idx, "label_rep_core"].values.astype(int)
        else:
            mask_train = mask_window
            idx = meta.index[mask_train].values
            party_vals = meta.loc[idx, PARTY_COL].values
            y_true = ((party_vals == "R") | (party_vals == 200)).astype(int)

        X = X_all[idx]
        y_prob = model.predict_proba(X)[:, 1]

        # Speech-level accuracy
        y_pred_speech = (y_prob > 0.5).astype(int)
        speech_acc = (y_pred_speech == y_true).mean()

        # Build legislator-level data
        speaker_ids = meta.loc[idx, "speakerid"].values
        leg_df = pd.DataFrame({
            "speakerid": speaker_ids,
            "y_true": y_true,
            "y_prob": y_prob,
        })
        leg_counts = leg_df.groupby("speakerid").size()

        # All-speech legislator accuracy
        leg_agg = leg_df.groupby("speakerid").agg(
            y_true=("y_true", "first"),
            y_prob_mean=("y_prob", "mean"),
            n_speeches=("y_prob", "count"),
        )
        leg_all_acc = ((leg_agg["y_prob_mean"] > 0.5).astype(int) == leg_agg["y_true"]).mean()

        congress_summary.append({
            "congress": cong,
            "n_speeches": len(idx),
            "n_legislators": len(leg_agg),
            "speech_accuracy": speech_acc,
            "legislator_accuracy": leg_all_acc,
            "median_speeches_per_leg": leg_agg["n_speeches"].median(),
        })
        print(f"    Speeches: {len(idx):,}, Legislators: {len(leg_agg):,}")
        print(f"    Speech accuracy: {speech_acc:.3f}")
        print(f"    Legislator accuracy (all speeches): {leg_all_acc:.3f}")

        # Convergence: accuracy at k speeches per legislator
        for k in k_values:
            # Only legislators with >= k speeches
            eligible = leg_counts[leg_counts >= k].index
            if len(eligible) < 20:
                continue

            eligible_df = leg_df[leg_df["speakerid"].isin(eligible)]
            grouped = {sid: g for sid, g in eligible_df.groupby("speakerid")}

            accs = []
            for b in range(n_bootstrap):
                correct = 0
                total = 0
                for sid, g in grouped.items():
                    sample = g.sample(n=k, random_state=rng.integers(1e9))
                    pred = int(sample["y_prob"].mean() > 0.5)
                    true = g["y_true"].iloc[0]
                    correct += (pred == true)
                    total += 1
                accs.append(correct / total)

            all_results.append({
                "congress": cong,
                "k": k,
                "accuracy_mean": np.mean(accs),
                "accuracy_se": np.std(accs),
                "n_legislators": len(eligible),
            })
            print(f"    k={k:>3}: acc={np.mean(accs):.3f} (SE={np.std(accs):.3f}), "
                  f"n_leg={len(eligible)}")

        del model, X, y_prob, leg_df
        gc.collect()

    del X_all
    gc.collect()

    results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(congress_summary)

    return results_df, summary_df


def plot_accuracy_convergence(results_df, summary_df):
    """Plot legislator accuracy vs number of speeches aggregated."""
    print("\nPlotting accuracy convergence ...")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    congresses = sorted(results_df["congress"].unique())

    # Individual congress curves: thin gray traces
    for cong in congresses:
        sub = results_df[results_df["congress"] == cong].sort_values("k")
        if len(sub) < 2:
            continue
        ax.plot(sub["k"], sub["accuracy_mean"], "-", color="#b0b0b0",
                markersize=0, linewidth=0.8, alpha=0.6, zorder=2)
        ax.fill_between(sub["k"],
                        sub["accuracy_mean"] - 1.96 * sub["accuracy_se"],
                        sub["accuracy_mean"] + 1.96 * sub["accuracy_se"],
                        color="#b0b0b0", alpha=0.05)
    # Ghost label for legend
    ax.plot([], [], "-", color="#b0b0b0", linewidth=0.8, alpha=0.6,
            label=f"Individual congresses ({congresses[0]}\u2013{congresses[-1]})")

    # Median across congresses: bold red line
    C_R = "#bf6b63"
    median_curve = results_df.groupby("k")["accuracy_mean"].median()
    ax.plot(median_curve.index, median_curve.values, "-o", color=C_R,
            linewidth=2.2, markersize=4, label="Median", zorder=10)

    ax.set_xlabel("Number of speeches per legislator", fontsize=11)
    ax.set_ylabel("Classification accuracy", fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0.55, 1.02)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out_path = FIG_DIR / "accuracy_convergence.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ====================================================================
# PART 2: Newspaper slant convergence
# ====================================================================
def compute_slant_convergence():
    """
    For each newspaper-year, compute article-level SD of net_slant_norm
    and the implied SE = SD/sqrt(n). Also bootstrap at several subsample
    sizes for a few representative newspapers to show convergence paths.
    """
    print("\n" + "=" * 72)
    print("PART 2: Newspaper slant convergence")
    print("=" * 72)

    congresses = cfg.get_congresses()
    ny_stats = []  # newspaper-year level stats

    for cong in congresses:
        label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        slant_path = NEWS_DIR / f"09_article_slant_norm_cong_{cong}.parquet"

        if not label_path.exists() or not slant_path.exists():
            print(f"  WARNING: Congress {cong} files not found, skipping")
            continue

        print(f"  Congress {cong}: loading ...")
        meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])
        slant = pd.read_parquet(slant_path, columns=["net_slant_norm"])

        # Handle subsampling if used in step 07
        sample_idx_path = NEWS_DIR / f"07_sample_idx_cong_{cong}.npy"
        if sample_idx_path.exists():
            idx = np.load(sample_idx_path)
            meta = meta.iloc[idx].reset_index(drop=True)

        assert len(meta) == len(slant)

        # Filter to news
        is_news = meta["is_news"].values
        papers = meta["paper"].values[is_news]
        years = meta["year"].values[is_news]
        slant_vals = slant["net_slant_norm"].values[is_news]

        # Group by newspaper-year (exclude national papers)
        df = pd.DataFrame({"paper": papers, "year": years, "slant": slant_vals})
        df = df[~df["paper"].isin(cfg.NATIONAL_PAPERS)]
        grp = df.groupby(["paper", "year"])["slant"]
        stats = grp.agg(["count", "mean", "std"]).reset_index()
        stats.columns = ["paper", "year", "n_articles", "mean_slant", "sd_slant"]
        stats["se_slant"] = stats["sd_slant"] / np.sqrt(stats["n_articles"])
        stats["congress"] = cong
        ny_stats.append(stats)

        del meta, slant, df
        gc.collect()

    ny_df = pd.concat(ny_stats, ignore_index=True)
    print(f"\n  Total newspaper-year obs: {len(ny_df):,}")
    print(f"  Median n_articles: {ny_df['n_articles'].median():,.0f}")
    print(f"  Median SE: {ny_df['se_slant'].median():.4f}")

    return ny_df


def compute_convergence_paths(n_newspapers=4, n_draws=30):
    """
    For representative newspaper-years, randomly subsample articles at
    increasing sizes to show convergence of mean slant estimate.
    Returns draw-level data for trace plotting.
    """
    print("\n  Computing convergence paths ...")

    target_cong = 104

    label_path = LABEL_DIR / f"04_newspaper_labeled_cong_{target_cong}.parquet"
    slant_path = NEWS_DIR / f"09_article_slant_norm_cong_{target_cong}.parquet"

    meta = pd.read_parquet(label_path, columns=["paper", "year", "is_news"])
    slant = pd.read_parquet(slant_path, columns=["net_slant_norm"])

    sample_idx_path = NEWS_DIR / f"07_sample_idx_cong_{target_cong}.npy"
    if sample_idx_path.exists():
        idx = np.load(sample_idx_path)
        meta = meta.iloc[idx].reset_index(drop=True)

    is_news = meta["is_news"].values
    df = pd.DataFrame({
        "paper": meta["paper"].values[is_news],
        "year": meta["year"].values[is_news],
        "slant": slant["net_slant_norm"].values[is_news],
    })
    df = df[~df["paper"].isin(cfg.NATIONAL_PAPERS)]

    # Hand-picked well-known newspapers spanning the slant distribution
    PREFERRED = [
        "THE SAN FRANCISCO CHRONICLE",   # low slant (~0.07)
        "Chicago Sun-Times",             # mid (~0.19)
        "Miami Herald, The (FL)",         # upper-mid (~0.27)
        "Salt Lake Tribune, The (UT)",    # high (~0.46)
    ]

    ny_counts = df.groupby(["paper", "year"]).size().reset_index(name="n")
    means = df.groupby(["paper", "year"])["slant"].mean().reset_index()
    means.columns = ["paper", "year", "mean_slant"]
    ny_counts = ny_counts.merge(means).sort_values("mean_slant").reset_index(drop=True)

    # Select preferred newspapers (pick highest-article year for each)
    selected = []
    for pref in PREFERRED:
        match = ny_counts[ny_counts["paper"] == pref]
        if len(match) > 0:
            selected.append(match.sort_values("n", ascending=False).iloc[0])
    if selected:
        ny_counts = pd.DataFrame(selected).sort_values("mean_slant").reset_index(drop=True)
    else:
        # Fallback: auto-select if no preferred papers found
        ny_counts = ny_counts[ny_counts["n"] >= 10000]
        if len(ny_counts) > n_newspapers:
            step = max(1, len(ny_counts) // n_newspapers)
            ny_counts = ny_counts.iloc[::step].head(n_newspapers)

    rng = np.random.default_rng(42)
    k_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    paths = []

    for _, row in ny_counts.iterrows():
        paper, year = row["paper"], row["year"]
        articles = df[(df["paper"] == paper) & (df["year"] == year)]["slant"].values
        n_total = len(articles)
        true_mean = articles.mean()
        print(f"    {paper[:40]}: N={n_total:,}, mean={true_mean:.3f}")

        for draw in range(n_draws):
            shuffled = rng.permutation(articles)
            for k in k_values:
                if k > n_total:
                    break
                paths.append({
                    "paper": paper,
                    "year": year,
                    "draw": draw,
                    "k": k,
                    "mean_slant": shuffled[:k].mean(),
                    "true_mean": true_mean,
                    "n_total": n_total,
                })

    del meta, slant, df
    gc.collect()

    return pd.DataFrame(paths)


def plot_slant_convergence(paths_df, ny_df=None):
    """
    Single-panel figure: for representative newspapers, show that the
    estimated mean slant converges to a stable value as article count grows.
    Individual random draws shown as thin traces; bold line = median.
    If ny_df is provided, adds a shaded band showing the IQR and median
    of actual newspaper-year article counts across the full sample.
    """
    print("\nPlotting slant convergence ...")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    papers = paths_df["paper"].unique()
    palette = ["#2d2d2d", "#7a7a7a", "#b0b0b0", "#bf6b63"]
    colors = dict(zip(papers, palette[:len(papers)]))

    for paper in papers:
        sub = paths_df[paths_df["paper"] == paper]
        c = colors[paper]

        # Individual draws: thin, semi-transparent traces
        for draw in sub["draw"].unique():
            d = sub[sub["draw"] == draw].sort_values("k")
            ax.plot(d["k"], d["mean_slant"], color=c,
                    alpha=0.18, linewidth=0.4, zorder=2)

        # Median trace: bold
        med = sub.groupby("k")["mean_slant"].median().reset_index()

        # Clean newspaper name for legend
        short = paper.split(",")[0].strip()
        # Fix ALL-CAPS names, but preserve known acronyms
        KEEP_UPPER = {"USA"}
        if short == short.upper() and len(short) > 5:
            words = short.split()
            words = [w if w in KEEP_UPPER else w.title() for w in words]
            short = " ".join(words)
        short = short.replace("The ", "").strip()
        # Only add city if the name doesn't already contain parenthetical info
        if "(" not in short and "(" in paper:
            city = paper.split("(")[1].rstrip(")")
            short = f"{short} ({city})"
        if len(short) > 30:
            short = short[:27] + "..."

        ax.plot(med["k"], med["mean_slant"], color=c,
                linewidth=1.2, label=short, zorder=3)

    # Add sample article-count distribution context
    if ny_df is not None:
        q25 = ny_df["n_articles"].quantile(0.25)
        q50 = ny_df["n_articles"].median()
        q75 = ny_df["n_articles"].quantile(0.75)
        ylo, yhi = ax.get_ylim()
        ax.axvspan(q25, q75, color="#888888", alpha=0.08, zorder=0)
        ax.axvline(q50, color="#888888", linewidth=0.8, linestyle="--",
                   alpha=0.5, zorder=1)
        ax.annotate(f"Sample median\n({int(q50):,} articles)",
                    xy=(q50, yhi - 0.02 * (yhi - ylo)),
                    xytext=(q50 * 1.5, yhi - 0.05 * (yhi - ylo)),
                    fontsize=7.5, color="#555555", ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color="#888888",
                                   linewidth=0.6))

    ax.set_xlabel("Number of articles", fontsize=11)
    ax.set_ylabel("Estimated mean slant ($\\tilde{S}$)", fontsize=11)
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="best", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    out_path = FIG_DIR / "slant_convergence.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    import time
    t0 = time.time()

    # Part 1: Legislator accuracy
    acc_results, acc_summary = compute_legislator_accuracy()
    plot_accuracy_convergence(acc_results, acc_summary)

    # Save tables
    acc_summary.to_csv(TABLE_DIR / "legislator_accuracy_summary.csv", index=False)
    acc_results.to_csv(TABLE_DIR / "accuracy_convergence.csv", index=False)
    print(f"\n  Summary:\n{acc_summary.to_string(index=False)}")

    # Part 2: Slant convergence
    ny_df = compute_slant_convergence()
    paths_df = compute_convergence_paths(n_newspapers=4, n_draws=30)
    plot_slant_convergence(paths_df, ny_df=ny_df)

    # Descriptive stats for article counts
    print("\n  Newspaper-year article count descriptive stats:")
    desc = ny_df["n_articles"].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
    for stat in ["mean", "std", "min", "10%", "25%", "50%", "75%", "90%", "max"]:
        print(f"    {stat:>5s}: {desc[stat]:,.0f}")
    print(f"    N newspaper-years: {len(ny_df):,}")

    # Save tables
    ny_df.to_csv(TABLE_DIR / "newspaper_year_precision.csv", index=False)
    paths_df.to_csv(TABLE_DIR / "slant_convergence_paths.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")
