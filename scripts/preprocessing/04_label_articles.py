"""
04_label_articles.py

Label newspaper articles with:
  1. is_news  — True for real news articles, False for death notices,
                classifieds, recipes, comics, etc.
  2. econ_score — cosine similarity to an economy anchor vector
                  (higher = more likely economic content)

Approach (economy score):
  - Define economy seed words, stem with PorterStemmer.
  - Build a sparse anchor vector from vocabulary hits, L2-normalize.
  - Cosine similarity = dot product (article feature rows are L2-normed).

Approach (is_news):
  - Title-based pattern matching against known non-news categories.

Inputs:
  - data/processed/speeches/05_feature_vectorizer.joblib
  - data/processed/newspapers/07_newspaper_features_cong_{cong}.npz
  - data/processed/newspapers/07_newspaper_meta_cong_{cong}.parquet

Outputs (per congress):
  - data/processed/newspapers/04_newspaper_labeled_cong_{cong}.parquet
"""

import gc
import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from nltk.stem.porter import PorterStemmer

# Ensure text_analyzer is importable (needed for unpickling the vectorizer)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import text_analyzer  # noqa: F401

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

VECTORIZER_PATH = BASE_DIR / "data" / "processed" / "speeches" / "05_feature_vectorizer.joblib"
NEWSPAPER_DIR = BASE_DIR / "data" / "processed" / "newspapers"

OUT_DIR = NEWSPAPER_DIR  # save alongside step-07 outputs

# ------------------------------------------------------------------
# Non-news title patterns (case-insensitive)
# ------------------------------------------------------------------
NON_NEWS_PATTERNS = [
    # Death / obituaries
    r"death\s*notice", r"obituar", r"funeral", r"\bdeaths\b", r"\bobituary\b",
    r"area deaths", r"deaths elsewhere", r"paid notice",
    # Births / weddings / life events
    r"\bbirths?\b", r"\bwedding", r"\bengag", r"\banniversar",
    # Classifieds / legal
    r"classif", r"legal\s*notice", r"public\s*notice",
    # Non-news sections
    r"\brecipe", r"\bcook(?:ing|book)", r"\bhoroscop", r"\bcomic",
    r"\bpuzzle", r"\bcrossword", r"\bcalendar\b",
    r"tv\s*listing", r"movie\s*listing", r"tv\s*schedule",
    # Photo captions (not articles)
    r"^photo\s*caption", r"^photo\b$",
    # Weather
    r"^weather\b",
    # Church listings
    r"church\s*(?:director|listing|calendar|service)",
]

# Compile into a single pattern for speed
_NON_NEWS_RE = re.compile("|".join(NON_NEWS_PATTERNS), re.IGNORECASE)

# ------------------------------------------------------------------
# Economy seed words (raw English — will be stemmed at runtime)
# ------------------------------------------------------------------
ECONOMY_SEEDS_UNI = [
    "tax", "taxes", "taxation",
    "budget", "budgets", "budgetary",
    "deficit", "deficits",
    "economy", "economic", "economics", "economist",
    "inflation", "inflationary",
    "unemployment", "unemployed",
    "trade", "trading",
    "fiscal",
    "spending", "spend", "expenditure", "expenditures",
    "debt", "debts",
    "recession", "recessionary",
    "tariff", "tariffs",
    "subsidy", "subsidies", "subsidize",
    "wage", "wages",
    "labor", "labour",
    "financial", "finance", "finances",
    "bank", "banking", "banks", "banker",
    "invest", "investment", "investments", "investor",
    "profit", "profits", "profitable",
    "revenue", "revenues",
    "income", "incomes",
    "monetary",
    "surplus",
    "gdp",
    "treasury",
    "appropriation", "appropriations",
    "entitlement", "entitlements",
    "deregulation", "deregulate",
    "privatization", "privatize",
    "mortgage", "mortgages",
    "pension", "pensions",
    "currency",
    "export", "exports", "exporting",
    "import", "imports", "importing",
    "manufacture", "manufacturing",
    "productivity",
    "commodity", "commodities",
    "shareholder", "shareholders",
    "dividend", "dividends",
    "corporate",
    "entrepreneur",
]

ECONOMY_SEEDS_BI = [
    ("tax", "cut"),
    ("tax", "cuts"),
    ("tax", "reform"),
    ("tax", "relief"),
    ("tax", "increase"),
    ("tax", "break"),
    ("tax", "code"),
    ("tax", "rate"),
    ("tax", "rates"),
    ("interest", "rate"),
    ("interest", "rates"),
    ("stock", "market"),
    ("federal", "reserve"),
    ("minimum", "wage"),
    ("trade", "deficit"),
    ("trade", "agreement"),
    ("free", "trade"),
    ("budget", "deficit"),
    ("budget", "surplus"),
    ("economic", "growth"),
    ("economic", "policy"),
    ("national", "debt"),
    ("social", "security"),
    ("small", "business"),
    ("wall", "street"),
    ("real", "estate"),
    ("credit", "card"),
    ("health", "insurance"),
    ("capital", "gains"),
    ("balanced", "budget"),
    ("fiscal", "year"),
    ("fiscal", "policy"),
    ("monetary", "policy"),
    ("income", "tax"),
    ("estate", "tax"),
    ("corporate", "tax"),
    ("payroll", "tax"),
    ("gross", "domestic"),
]


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    stemmer = PorterStemmer()

    # ---- Economy anchor ----
    print("Loading vectorizer ...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    vocab = vectorizer.vocabulary_
    n_features = len(vocab)
    print(f"  Vocabulary size: {n_features:,}")

    print("\nBuilding economy anchor vector ...")

    seed_indices = []
    hits_uni, misses_uni = [], []
    seen_stems = set()
    for word in ECONOMY_SEEDS_UNI:
        stem = stemmer.stem(word.lower())
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        idx = vocab.get(stem)
        if idx is not None:
            seed_indices.append(idx)
            hits_uni.append(stem)
        else:
            misses_uni.append(stem)

    hits_bi, misses_bi = [], []
    seen_bigrams = set()
    for w1, w2 in ECONOMY_SEEDS_BI:
        s1 = stemmer.stem(w1.lower())
        s2 = stemmer.stem(w2.lower())
        bigram = f"{s1} {s2}"
        if bigram in seen_bigrams:
            continue
        seen_bigrams.add(bigram)
        idx = vocab.get(bigram)
        if idx is not None:
            seed_indices.append(idx)
            hits_bi.append(bigram)
        else:
            misses_bi.append(bigram)

    print(f"  Unigram seeds: {len(hits_uni)} hits, {len(misses_uni)} misses")
    if misses_uni:
        print(f"    Misses: {misses_uni}")
    print(f"  Bigram seeds:  {len(hits_bi)} hits, {len(misses_bi)} misses")
    if misses_bi:
        print(f"    Misses: {misses_bi}")
    print(f"  Total anchor features: {len(seed_indices)}")

    anchor = sp.csr_matrix(
        (np.ones(len(seed_indices)), (np.zeros(len(seed_indices), dtype=int), seed_indices)),
        shape=(1, n_features),
    )
    norm = sp.linalg.norm(anchor)
    anchor = anchor / norm
    print(f"  Anchor L2 norm (before normalization): {norm:.4f}")

    del vectorizer
    gc.collect()

    # ---- Process each congress ----
    print("\nLabeling articles per congress ...")
    congresses = list(range(100, 109))

    all_stats = []

    for cong in congresses:
        features_path = NEWSPAPER_DIR / f"07_newspaper_features_cong_{cong}.npz"
        meta_path = NEWSPAPER_DIR / f"07_newspaper_meta_cong_{cong}.parquet"

        if not features_path.exists():
            print(f"  WARNING: {features_path.name} not found, skipping")
            continue

        X_news = sp.load_npz(features_path)
        meta = pd.read_parquet(meta_path)

        # --- is_news flag ---
        titles = meta["title"].fillna("")
        is_non_news = titles.str.contains(_NON_NEWS_RE, regex=True)
        meta["is_news"] = ~is_non_news

        # --- econ_score ---
        econ_scores = X_news.dot(anchor.T).toarray().ravel()
        meta["econ_score"] = econ_scores

        # Save
        out_path = OUT_DIR / f"04_newspaper_labeled_cong_{cong}.parquet"
        meta.to_parquet(out_path)

        # Stats
        n_total = len(meta)
        n_news = meta["is_news"].sum()
        n_non_news = n_total - n_news
        pct_non_news = n_non_news / n_total * 100

        news_econ = econ_scores[meta["is_news"].values]
        stats = {
            "congress": cong,
            "n_total": n_total,
            "n_news": int(n_news),
            "n_non_news": int(n_non_news),
            "pct_non_news": pct_non_news,
            "econ_mean": np.mean(econ_scores),
            "econ_p90": np.percentile(econ_scores, 90),
            "econ_p95": np.percentile(econ_scores, 95),
            "econ_mean_news": np.mean(news_econ) if len(news_econ) > 0 else 0,
        }
        all_stats.append(stats)

        print(f"  Congress {cong}: {n_total:>9,} total  |  "
              f"{n_news:>9,} news  {n_non_news:>7,} non-news ({pct_non_news:.1f}%)  |  "
              f"econ p90={stats['econ_p90']:.4f}  p95={stats['econ_p95']:.4f}")

        del X_news, meta, econ_scores
        gc.collect()

    # ---- Summary ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    stats_df = pd.DataFrame(all_stats)
    total = stats_df["n_total"].sum()
    total_news = stats_df["n_news"].sum()
    total_non = stats_df["n_non_news"].sum()

    print(f"\n  Total articles:     {total:,}")
    print(f"  News articles:      {total_news:,} ({total_news/total*100:.1f}%)")
    print(f"  Non-news filtered:  {total_non:,} ({total_non/total*100:.1f}%)")
    print(f"\n  Avg econ_score (all):  {stats_df['econ_mean'].mean():.4f}")
    print(f"  Avg econ_score (news): {stats_df['econ_mean_news'].mean():.4f}")
    print(f"\n  Saved to -> {OUT_DIR}")
    print("=" * 72)
