"""
05_build_tfidf.py

Build a uniform TF-IDF matrix from congressional speeches (Congresses 99-108)
with comprehensive text preprocessing to isolate genuine partisan language.

Preprocessing pipeline:
  1. Remove parliamentary procedure phrases from raw text
  2. Tokenize (lowercase, alphabetic tokens >= 2 chars)
  3. Remove English stop words (sklearn default list)
  4. Remove geographic terms (state names, major cities, state abbreviations)
  5. Apply Porter Stemmer to all remaining tokens
  6. Form unigrams and bigrams
  7. Filter out legislator full-name bigrams (stemmed)
     (Individual name tokens are NOT removed -- many are common words
      like "young", "brown", "price", "bill")

Steps:
  1. Load speech text (01) and partisan-core labels (04).
  2. Build stop-word lists (geographic, parliamentary, legislator names).
  3. Aggregate all speeches by legislator-congress (icpsr + congress).
  4. Fit a single TF-IDF vectorizer with custom analyzer on the full corpus.
  5. Save the sparse TF-IDF matrix, labels, and the fitted vectorizer.

TF-IDF settings:
  ngram_range  = (1, 2)  -- handled in custom analyzer
  min_df       = 0.001
  sublinear_tf = True
  stemming     = Porter Stemmer (NLTK)
"""

import os
import re
import time
import joblib
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
INTER_DIR = BASE_DIR / "data" / "intermediate" / "speeches"

SPEECHES_PATH = INTER_DIR / "01_speeches_merged.parquet"
LABELS_PATH = INTER_DIR / "04_speeches_with_partisan_core.parquet"
VOTEVIEW_PATH = BASE_DIR / "data" / "raw" / "voteview_nominate" / "HSall_members.csv"

OUT_DIR = BASE_DIR / "data" / "processed" / "speeches" / "minwoo"
OUT_TFIDF = OUT_DIR / "05_tfidf_matrix.npz"
OUT_META = OUT_DIR / "05_tfidf_meta.parquet"
OUT_VECTORIZER = OUT_DIR / "05_tfidf_vectorizer.joblib"

# ------------------------------------------------------------------
# Parliamentary procedure phrases to remove
# ------------------------------------------------------------------
PARLIAMENTARY_PHRASES = [
    "i yield back the balance of my time",
    "i yield back",
    "i yield to the gentleman",
    "i yield to the gentlewoman",
    "i yield to my colleague",
    "i yield myself such time as i may consume",
    "i yield myself the balance of my time",
    "reclaiming my time",
    "mr speaker",
    "mr chairman",
    "mr president",
    "madam speaker",
    "madam chairman",
    "madam president",
    "the gentleman from",
    "the gentlewoman from",
    "the distinguished gentleman",
    "the distinguished gentlewoman",
    "my distinguished colleague",
    "my good friend",
    "the chair recognizes",
    "pursuant to house rule",
    "pursuant to clause",
    "pursuant to the rule",
    "pursuant to the order",
    "i ask unanimous consent",
    "without objection so ordered",
    "the question is on",
    "the yeas and nays are ordered",
    "the previous question is ordered",
    "i rise today",
    "i rise in support",
    "i rise in opposition",
    "i reserve the balance of my time",
    "under a previous order of the house",
    "i include for the record",
    "i insert for the record",
    "permission to revise and extend",
    "i move to strike the last word",
    "the time of the gentleman has expired",
    "the time of the gentlewoman has expired",
]

# ------------------------------------------------------------------
# US state names (including multi-word)
# ------------------------------------------------------------------
STATE_NAMES = [
    "alabama", "alaska", "arizona", "arkansas", "california",
    "colorado", "connecticut", "delaware", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia",
]

STATE_ABBREVS = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
    "dc",
]

# Major US cities likely to appear in congressional speech
MAJOR_CITIES = [
    "new york", "los angeles", "chicago", "houston", "phoenix",
    "philadelphia", "san antonio", "san diego", "dallas", "san jose",
    "austin", "jacksonville", "san francisco", "columbus", "indianapolis",
    "charlotte", "seattle", "denver", "boston", "detroit",
    "nashville", "memphis", "portland", "oklahoma city", "las vegas",
    "louisville", "baltimore", "milwaukee", "albuquerque", "tucson",
    "fresno", "sacramento", "mesa", "kansas city", "atlanta",
    "omaha", "colorado springs", "raleigh", "miami", "minneapolis",
    "cleveland", "tampa", "pittsburgh", "cincinnati", "st louis",
    "orlando", "honolulu", "anchorage", "richmond", "buffalo",
    "birmingham", "brooklyn", "manhattan",
    # State capitals not in list above
    "montgomery", "juneau", "little rock", "hartford", "dover",
    "tallahassee", "boise", "springfield", "des moines", "topeka",
    "frankfort", "baton rouge", "augusta", "annapolis", "lansing",
    "jackson", "jefferson city", "helena", "lincoln", "carson city",
    "concord", "trenton", "santa fe", "albany", "bismarck",
    "salem", "harrisburg", "providence", "columbia", "pierre",
    "salt lake city", "montpelier", "olympia", "charleston", "madison",
    "cheyenne",
]


# ==================================================================
# Helper: build stop-word sets and name bigram set
# ==================================================================
def build_filter_sets(voteview_path):
    """Build all stop-word sets for the custom analyzer."""
    stemmer = PorterStemmer()

    # --- English stop words (stemmed) ---
    english_stops_stemmed = {stemmer.stem(w) for w in ENGLISH_STOP_WORDS}

    # --- Geographic unigrams (stemmed single-word names + abbreviations) ---
    geo_unigrams_stemmed = set()
    for name in STATE_NAMES + MAJOR_CITIES:
        tokens = name.split()
        if len(tokens) == 1:
            geo_unigrams_stemmed.add(stemmer.stem(tokens[0]))
    for abbr in STATE_ABBREVS:
        geo_unigrams_stemmed.add(abbr)  # abbreviations are short, don't stem

    # --- Geographic bigrams (stemmed multi-word names) ---
    geo_bigrams_stemmed = set()
    for name in STATE_NAMES + MAJOR_CITIES:
        tokens = name.split()
        if len(tokens) == 2:
            bg = stemmer.stem(tokens[0]) + " " + stemmer.stem(tokens[1])
            geo_bigrams_stemmed.add(bg)

    # --- Legislator names from Voteview (Congresses 99-108, House) ---
    voteview = pd.read_csv(voteview_path)
    house = voteview[
        (voteview["congress"] >= 99)
        & (voteview["congress"] <= 108)
        & (voteview["chamber"] == "House")
    ]

    # Only filter full-name bigrams, NOT individual name tokens.
    # Many legislators share common surnames (Young, Brown, Price, Long)
    # that would incorrectly remove real words like "young voters" or
    # "price controls".
    legislator_bigrams_stemmed = set()

    for bioname in house["bioname"].dropna().unique():
        # bioname format: "LASTNAME, Firstname Middle (Nickname), Suffix"
        parts = bioname.split(",")
        if len(parts) < 2:
            continue
        lastname = parts[0].strip().lower()
        firstname_part = parts[1].strip().lower()
        # Extract first token of firstname (ignore middle/nickname)
        firstname = re.sub(r"[^a-z ]", "", firstname_part).split()
        if not firstname:
            continue
        firstname = firstname[0]
        lastname_clean = re.sub(r"[^a-z]", "", lastname)

        # Full name bigram: "firstname lastname" and "lastname firstname"
        if len(firstname) >= 2 and len(lastname_clean) >= 2:
            bg = stemmer.stem(firstname) + " " + stemmer.stem(lastname_clean)
            legislator_bigrams_stemmed.add(bg)
            bg_rev = stemmer.stem(lastname_clean) + " " + stemmer.stem(firstname)
            legislator_bigrams_stemmed.add(bg_rev)

    # Combine blocked bigrams
    blocked_bigrams = geo_bigrams_stemmed | legislator_bigrams_stemmed

    print(f"  English stop words (stemmed): {len(english_stops_stemmed)}")
    print(f"  Geographic unigrams (stemmed): {len(geo_unigrams_stemmed)}")
    print(f"  Geographic bigrams (stemmed): {len(geo_bigrams_stemmed)}")
    print(f"  Legislator name bigrams (stemmed): {len(legislator_bigrams_stemmed)}")
    print(f"  Total blocked bigrams: {len(blocked_bigrams)}")

    return english_stops_stemmed, geo_unigrams_stemmed, blocked_bigrams


# ==================================================================
# Helper: build the custom analyzer function
# ==================================================================
def build_analyzer(english_stops, geo_unigrams, blocked_bigrams):
    """
    Return a callable analyzer for TfidfVectorizer.

    Pipeline per document:
      1. Lowercase
      2. Remove parliamentary phrases
      3. Tokenize (alpha only, >= 2 chars)
      4. Remove english stop words (stemmed)
      5. Remove geographic unigrams (stemmed)
      6. Stem remaining tokens with Porter Stemmer
      7. Form unigrams + bigrams
      8. Filter blocked bigrams (geographic + legislator names)
    """
    stemmer = PorterStemmer()
    stem_cache = {}  # memoize: avoids re-stemming repeated tokens
    token_re = re.compile(r"[a-z]+")

    # Combine all parliamentary phrases into ONE regex (much faster than 40 loops)
    parl_pattern = re.compile(
        "|".join(re.escape(p) for p in PARLIAMENTARY_PHRASES),
        re.IGNORECASE,
    )

    # Combined single-token stops: english + geographic (all stemmed)
    single_stops = english_stops | geo_unigrams

    def analyzer(doc):
        # 1. Lowercase
        text = doc.lower()

        # 2. Remove parliamentary phrases (single pass)
        text = parl_pattern.sub(" ", text)

        # 3. Tokenize (alphabetic, >= 2 chars)
        raw_tokens = token_re.findall(text)

        # 4-6. Stem (cached), then filter stop words + geographic
        stemmed = []
        for t in raw_tokens:
            if len(t) < 2:
                continue
            s = stem_cache.get(t)
            if s is None:
                s = stemmer.stem(t)
                stem_cache[t] = s
            if len(s) >= 2 and s not in single_stops:
                stemmed.append(s)

        # 7. Form unigrams + bigrams
        features = list(stemmed)
        for i in range(len(stemmed) - 1):
            bg = stemmed[i] + " " + stemmed[i + 1]
            # 8. Filter blocked bigrams
            if bg not in blocked_bigrams:
                features.append(bg)

        return features

    return analyzer


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading speech text ...")
    speeches = pd.read_parquet(SPEECHES_PATH)

    print("Loading partisan-core labels ...")
    labels = pd.read_parquet(LABELS_PATH)

    # ------------------------------------------------------------------
    # 2. Build filter sets
    # ------------------------------------------------------------------
    print("\nBuilding filter sets ...")
    english_stops, geo_unigrams, blocked_bigrams = build_filter_sets(VOTEVIEW_PATH)

    # ------------------------------------------------------------------
    # 3. Merge text with labels
    # ------------------------------------------------------------------
    print("\nMerging text with labels ...")
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    labels["speech_id"] = labels["speech_id"].astype(str)

    merged = labels.merge(
        speeches[["speech_id", "speech"]],
        on="speech_id",
        how="inner",
    )
    print(f"  Merged: {len(merged):,} speeches with text + labels")

    # ------------------------------------------------------------------
    # 4. Aggregate text by legislator-congress
    # ------------------------------------------------------------------
    print("\nAggregating by legislator-congress ...")

    agg = (
        merged
        .groupby(["icpsr", "congress_int"])
        .agg(
            text=("speech", lambda x: " ".join(x.astype(str))),
            party_code=("party", "first"),
            label_rep_core=("label_rep_core", "max"),
            label_dem_core=("label_dem_core", "max"),
            nokken_poole_dim1=("nokken_poole_dim1", "first"),
            n_speeches=("speech_id", "count"),
        )
        .reset_index()
    )
    print(f"  Legislator-congress documents: {len(agg):,}")

    # ------------------------------------------------------------------
    # 5. Fit TF-IDF vectorizer with custom analyzer
    # ------------------------------------------------------------------
    print("\nFitting TF-IDF vectorizer (with Porter Stemmer + filtering) ...")

    base_analyzer = build_analyzer(english_stops, geo_unigrams, blocked_bigrams)

    # Wrap analyzer with tqdm progress tracking
    n_docs = len(agg)
    pbar = tqdm(total=n_docs * 2, desc="  TF-IDF (fit+transform)", unit="doc")

    def tracking_analyzer(doc):
        result = base_analyzer(doc)
        pbar.update(1)
        return result

    vectorizer = TfidfVectorizer(
        analyzer=tracking_analyzer,
        min_df=0.001,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(agg["text"])
    pbar.close()

    elapsed = time.time() - pipeline_start
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"  Non-zero entries: {tfidf_matrix.nnz:,}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp.save_npz(OUT_TFIDF, tfidf_matrix)
    print(f"\n  Saved TF-IDF matrix -> {OUT_TFIDF}")

    meta = agg.drop(columns=["text"])
    meta.to_parquet(OUT_META)
    print(f"  Saved metadata -> {OUT_META}")

    joblib.dump(vectorizer, OUT_VECTORIZER)
    print(f"  Saved vectorizer -> {OUT_VECTORIZER}")

    # ------------------------------------------------------------------
    # 7. Validation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION: Documents per congress")
    print("=" * 72)

    summary = (
        meta
        .groupby("congress_int")
        .agg(
            n_legislators=("icpsr", "count"),
            rep_core=("label_rep_core", "sum"),
            dem_core=("label_dem_core", "sum"),
            avg_speeches=("n_speeches", "mean"),
        )
    )
    summary["avg_speeches"] = summary["avg_speeches"].round(1)
    summary[["rep_core", "dem_core"]] = summary[["rep_core", "dem_core"]].astype(int)
    print(summary.to_string())
    print("=" * 72)

    # Show sample features
    feature_names = vectorizer.get_feature_names_out()
    n_unigrams = sum(1 for f in feature_names if " " not in f)
    n_bigrams = sum(1 for f in feature_names if " " in f)
    print(f"\n  Unigrams: {n_unigrams:,}  |  Bigrams: {n_bigrams:,}")
    print(f"  Sample unigrams: {list(feature_names[:10])}")
    bigram_samples = [f for f in feature_names if " " in f][:10]
    print(f"  Sample bigrams: {bigram_samples}")

    total_time = time.time() - pipeline_start
    print(f"\n  Total time: {total_time:.1f}s")
