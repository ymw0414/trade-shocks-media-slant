"""
05_build_tfidf.py

Build feature matrix from congressional speeches following Widmer et al. methodology.

Two modes controlled by pipeline_config:
  - Widmer mode (default): CountVectorizer + relative freq, bigrams only,
    individual speeches, all R/D, per-party frequency filter
  - Legacy mode: TfidfVectorizer, unigrams+bigrams, legislator-congress aggregation

Preprocessing pipeline (both modes):
  1. Remove parliamentary procedure phrases from raw text
  2. Tokenize (lowercase, alphabetic tokens >= 2 chars)
  3. Remove English stop words + procedural/legal stop words
  4. Remove geographic terms (state names, major cities, abbreviations)
  5. Remove distinctive legislator name tokens
  6. Apply Porter Stemmer
  7. Form bigrams (or unigrams+bigrams in legacy mode)
  8. Filter out legislator full-name bigrams

Widmer frequency filter (B.2):
  V = {b in (V^0.1_Rep ∪ V^0.1_Dem) ∩ (V^0.01_Rep ∩ V^0.01_Dem)}
  - Must appear in >=0.1% of either party's speeches (OR)
  - Must appear in >=0.01% of both parties' speeches (AND)
  Result: ~14,224 bigrams (Widmer gets 14,224)

Outputs:
  - 05_feature_matrix.npz  (sparse: relative freq or TF-IDF)
  - 05_speech_meta.parquet  (per-speech or per-legislator metadata)
  - 05_vectorizer.joblib    (fitted vectorizer for step 07)
  - 05_vocab_filter_idx.npy (column indices of filtered vocabulary)
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

# ------------------------------------------------------------------
# Paths (from pipeline_config — change RUN_NAME there for new runs)
# ------------------------------------------------------------------
import pipeline_config as cfg

SPEECHES_PATH  = cfg.SPEECHES_PATH
LABELS_PATH    = cfg.LABELS_PATH
VOTEVIEW_PATH  = cfg.VOTEVIEW_PATH

OUT_DIR        = cfg.SPEECH_DIR
OUT_MATRIX     = OUT_DIR / "05_feature_matrix.npz"
OUT_META       = OUT_DIR / "05_speech_meta.parquet"
OUT_VECTORIZER = OUT_DIR / "05_vectorizer.joblib"
OUT_VOCAB_IDX  = OUT_DIR / "05_vocab_filter_idx.npy"

# For backward compatibility (steps that reference old names)
OUT_TFIDF      = OUT_DIR / "05_tfidf_matrix.npz"
OUT_TFIDF_META = OUT_DIR / "05_tfidf_meta.parquet"

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
    "i yield to",
    "the gentleman yields",
    "the gentlewoman yields",
    "mr speaker", "mr chairman", "mr president",
    "madam speaker", "madam chairman", "madam president",
    "the gentleman from", "the gentlewoman from",
    "the distinguished gentleman", "the distinguished gentlewoman",
    "my distinguished colleague", "my good friend",
    "the chair recognizes",
    "the speaker pro tempore", "the speaker announced",
    "the speaker appoints", "the speaker laid before the house",
    "pursuant to house rule", "pursuant to clause",
    "pursuant to the rule", "pursuant to the order",
    "pursuant to the provisions",
    "i ask unanimous consent",
    "without objection so ordered", "without objection",
    "the question is on", "the yeas and nays are ordered",
    "the previous question is ordered", "the previous question",
    "the question was taken", "the question recurs",
    "i demand a recorded vote", "a recorded vote is demanded",
    "demand a recorded vote",
    "i demand a second", "demand a second",
    "the yeas and nays", "a recorded vote", "a rollcall vote",
    "by a recorded vote", "on this vote",
    "i rise today", "i rise in support", "i rise in opposition", "i rise to",
    "i reserve the balance of my time",
    "under a previous order of the house",
    "i include for the record", "i insert for the record",
    "permission to revise and extend",
    "revise and extend my remarks",
    "i move to strike the last word",
    "the time of the gentleman has expired",
    "the time of the gentlewoman has expired",
    "motion to table", "motion to reconsider", "motion to recommit",
    "motion to suspend the rules", "motion to instruct conferees",
    "i move to suspend the rules", "i offer a motion",
    "suspend the rules and pass", "suspend the rules and agree",
    "suspend the rules and concur",
    "amendment thereto", "amendments thereto", "amendment offered by",
    "amendment in the nature of a substitute",
    "for the purpose of amendment", "purpose of the amendment",
    "amendment to the amendment",
    "extension of remarks", "leave to extend",
    "leave to revise and extend", "general leave", "leave of absence",
    "permit the committee to sit", "permitted to sit",
    "the committee on rules", "reported the bill",
    "ordered to be reported",
    "is there objection", "objection is heard", "hearing none",
    "reserving the right to object", "i reserve the right to object",
    "i withdraw my reservation",
    "so ordered", "ordered to be printed", "ordered to lie on the table",
    "engrossment and third reading",
    "a quorum is present", "a quorum is not present",
    "the clerk will read", "the clerk will report",
    "the clerk will call the roll",
    "the clerk read", "the clerk reported", "the roll was called",
    "one minute speech", "one minute speeches",
    "morning hour", "special order", "special orders",
    "further proceedings", "further reading",
]

# ------------------------------------------------------------------
# Procedural stop words
# ------------------------------------------------------------------
PROCEDURAL_STOP_WORDS = [
    "thereto", "thereof", "hereby", "herein", "herewith",
    "whereas", "aforesaid", "notwithstanding", "therein",
    "heretofore", "hereunder", "thereunder", "thereupon",
    "whereupon", "inasmuch", "wherefor",
]

# ------------------------------------------------------------------
# Common English words that are also legislator surnames — keep as valid tokens
# ------------------------------------------------------------------
COMMON_ENGLISH_WORDS_ALSO_NAMES = {
    "archer", "baker", "barr", "bass", "bell", "berry", "bishop",
    "bland", "blunt", "bond", "brown", "camp", "castle", "chase",
    "church", "clay", "close", "cole", "collins", "cook", "cox",
    "crane", "cross", "dale", "deal", "dear", "dick", "duke",
    "early", "edge", "english", "fast", "fields", "fish", "flood",
    "ford", "foster", "fox", "frank", "free", "frost", "gore",
    "grant", "gray", "green", "hall", "hand", "hare", "hart",
    "hawk", "hay", "hill", "holt", "hook", "hope", "horn", "house",
    "hunter", "hyde", "keen", "king", "lane", "large", "lewis",
    "light", "long", "love", "mark", "marsh", "martin", "mason",
    "may", "miller", "minor", "mock", "moore", "moss", "neal",
    "nelson", "park", "penny", "porter", "post", "price", "ray",
    "reed", "rich", "ring", "rose", "rush", "sage", "sharp",
    "shepherd", "short", "silver", "singer", "small", "smith",
    "spring", "stark", "steel", "stone", "strong", "sweet", "swift",
    "taylor", "thomas", "thorn", "turner", "walker", "walsh", "ward",
    "warren", "waters", "watt", "webb", "well", "west", "white",
    "winter", "wise", "wolf", "wood", "wright", "young",
    "barton", "burton", "davis", "edwards", "evans", "harris",
    "jackson", "johnson", "jones", "lee", "martinez", "rogers",
    "wilson", "thompson", "robinson", "brooks", "flowers", "power",
    "savage", "owens", "watts", "towns",
}

# ------------------------------------------------------------------
# US state names, abbreviations, major cities
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

    english_stops_stemmed = {stemmer.stem(w) for w in ENGLISH_STOP_WORDS}
    procedural_stops_stemmed = {stemmer.stem(w) for w in PROCEDURAL_STOP_WORDS}
    english_stops_stemmed |= procedural_stops_stemmed

    geo_unigrams_stemmed = set()
    for name in STATE_NAMES + MAJOR_CITIES:
        tokens = name.split()
        if len(tokens) == 1:
            geo_unigrams_stemmed.add(stemmer.stem(tokens[0]))
    for abbr in STATE_ABBREVS:
        geo_unigrams_stemmed.add(abbr)

    geo_bigrams_stemmed = set()
    for name in STATE_NAMES + MAJOR_CITIES:
        tokens = name.split()
        if len(tokens) == 2:
            bg = stemmer.stem(tokens[0]) + " " + stemmer.stem(tokens[1])
            geo_bigrams_stemmed.add(bg)

    voteview = pd.read_csv(voteview_path)
    cong_lo, cong_hi = cfg.CONFIG["congress_range"]
    house = voteview[
        (voteview["congress"] >= cong_lo)
        & (voteview["congress"] <= cong_hi)
        & (voteview["chamber"] == "House")
    ]

    safe_name_stems = {stemmer.stem(w) for w in COMMON_ENGLISH_WORDS_ALSO_NAMES}
    safe_name_stems |= english_stops_stemmed

    legislator_bigrams_stemmed = set()
    distinctive_name_tokens = set()

    for bioname in house["bioname"].dropna().unique():
        parts = bioname.split(",")
        if len(parts) < 2:
            continue
        lastname = parts[0].strip().lower()
        firstname_part = parts[1].strip().lower()
        firstname_tokens = re.sub(r"[^a-z ]", "", firstname_part).split()
        if not firstname_tokens:
            continue
        firstname = firstname_tokens[0]
        lastname_clean = re.sub(r"[^a-z]", "", lastname)

        nickname_match = re.search(r"\(([^)]+)\)", parts[1] if len(parts) > 1 else "")
        nickname = None
        if nickname_match:
            nick_raw = nickname_match.group(1).strip().lower()
            nick_clean = re.sub(r"[^a-z]", "", nick_raw)
            if len(nick_clean) >= 2:
                nickname = nick_clean

        if len(firstname) >= 2 and len(lastname_clean) >= 2:
            fn_stem = stemmer.stem(firstname)
            ln_stem = stemmer.stem(lastname_clean)
            legislator_bigrams_stemmed.add(fn_stem + " " + ln_stem)
            legislator_bigrams_stemmed.add(ln_stem + " " + fn_stem)
            if nickname:
                nn_stem = stemmer.stem(nickname)
                legislator_bigrams_stemmed.add(nn_stem + " " + ln_stem)
                legislator_bigrams_stemmed.add(ln_stem + " " + nn_stem)

        if len(lastname_clean) >= 2:
            ln_stem = stemmer.stem(lastname_clean)
            if len(ln_stem) >= 4 and ln_stem not in safe_name_stems:
                distinctive_name_tokens.add(ln_stem)

    blocked_bigrams = geo_bigrams_stemmed | legislator_bigrams_stemmed

    # GST procedural bigrams (Roberts' Rules + Riddick's Senate Procedure)
    gst_procedural = set()
    if cfg.CONFIG.get("filter_gst_procedural", False):
        gst_path = cfg.GST_PROCEDURAL_PATH
        if gst_path.exists():
            with open(gst_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "|" in line and not line.startswith("phrase|"):
                        phrase = line.split("|")[0].strip()
                        if len(phrase.split()) == 2:
                            gst_procedural.add(phrase)
            blocked_bigrams = blocked_bigrams | gst_procedural
            print(f"  GST procedural bigrams loaded: {len(gst_procedural):,}")
        else:
            print(f"  WARNING: GST procedural file not found: {gst_path}")

    print(f"  English + procedural stop words (stemmed): {len(english_stops_stemmed)}")
    print(f"  Geographic unigrams (stemmed): {len(geo_unigrams_stemmed)}")
    print(f"  Geographic bigrams (stemmed): {len(geo_bigrams_stemmed)}")
    print(f"  Legislator name bigrams (stemmed): {len(legislator_bigrams_stemmed)}")
    print(f"  Distinctive name tokens (stemmed): {len(distinctive_name_tokens)}")
    print(f"  Total blocked bigrams: {len(blocked_bigrams)}")

    return english_stops_stemmed, geo_unigrams_stemmed, blocked_bigrams, distinctive_name_tokens


# ==================================================================
# Helper: build the custom analyzer
# ==================================================================
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from text_analyzer import TextAnalyzer


def build_analyzer(english_stops, geo_unigrams, blocked_bigrams,
                   distinctive_name_tokens=None, bigrams_only=False):
    """Return a picklable callable analyzer for CountVectorizer/TfidfVectorizer."""
    single_stops = english_stops | geo_unigrams
    if distinctive_name_tokens:
        single_stops = single_stops | distinctive_name_tokens
    return TextAnalyzer(
        single_stops=single_stops,
        blocked_bigrams=blocked_bigrams,
        bigrams_only=bigrams_only,
    )


# ==================================================================
# Widmer frequency filter
# ==================================================================
def apply_widmer_freq_filter(count_matrix, party_labels, feature_names):
    """
    Apply Widmer et al. (2020) frequency filter (Appendix B.2).

    V = {b in (V^0.1_Rep ∪ V^0.1_Dem) ∩ (V^0.01_Rep ∩ V^0.01_Dem)}

    Returns: array of column indices that pass the filter.
    """
    # Binarize: presence/absence per document
    binary = (count_matrix > 0).astype(np.float32)

    rep_mask = (party_labels == "R") | (party_labels == 200)
    dem_mask = (party_labels == "D") | (party_labels == 100)
    n_rep = rep_mask.sum()
    n_dem = dem_mask.sum()

    # Document frequency per party
    df_rep = np.asarray(binary[rep_mask].sum(axis=0)).ravel()
    df_dem = np.asarray(binary[dem_mask].sum(axis=0)).ravel()

    # Relative document frequency
    rdf_rep = df_rep / n_rep
    rdf_dem = df_dem / n_dem

    # Condition 1: >=0.1% in either party (OR)
    cond_01 = (rdf_rep >= 0.001) | (rdf_dem >= 0.001)

    # Condition 2: >=0.01% in both parties (AND)
    cond_001 = (rdf_rep >= 0.0001) & (rdf_dem >= 0.0001)

    # Combined
    keep_mask = cond_01 & cond_001
    keep_idx = np.where(keep_mask)[0]

    print(f"\n  Widmer frequency filter:")
    print(f"    Republican speeches: {n_rep:,}")
    print(f"    Democrat speeches:   {n_dem:,}")
    print(f"    Features >=0.1% either party: {cond_01.sum():,}")
    print(f"    Features >=0.01% both parties: {cond_001.sum():,}")
    print(f"    Intersection (final vocabulary): {len(keep_idx):,}")

    return keep_idx


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    pipeline_start = time.time()

    cfg.save_config()

    USE_WIDMER = cfg.CONFIG.get("use_relative_freq", True)
    BIGRAMS_ONLY = cfg.CONFIG.get("bigrams_only", True)
    AGGREGATE = cfg.CONFIG.get("aggregate_to_legislator", False)
    FREQ_FILTER = cfg.CONFIG.get("freq_filter_mode", "widmer")

    print(f"Mode: {'Widmer' if USE_WIDMER else 'Legacy TF-IDF'}")
    print(f"  N-grams: {'bigrams only' if BIGRAMS_ONLY else 'unigrams + bigrams'}")
    print(f"  Aggregation: {'legislator-congress' if AGGREGATE else 'individual speeches'}")
    print(f"  Frequency filter: {FREQ_FILTER}")

    # ------------------------------------------------------------------
    # 1. Load data (labels first, then speeches filtered by congress)
    # ------------------------------------------------------------------
    print("\nLoading labels ...")
    labels = pd.read_parquet(LABELS_PATH)
    labels["speech_id"] = labels["speech_id"].astype(str)

    # Pre-filter speeches by congress range to avoid loading all 17M rows
    cong_lo, cong_hi = cfg.CONFIG["congress_range"]
    congress_strings = [f"{c:03d}" for c in range(cong_lo, cong_hi + 1)]
    print(f"  Labels loaded: {len(labels):,} rows, will filter speeches to congresses {cong_lo}-{cong_hi}")

    print("\nLoading speech text (filtered by congress) ...")
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    pf = pq.ParquetFile(SPEECHES_PATH)
    speech_chunks = []
    for rg_idx in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=["speech_id", "speech", "congress"])
        mask = pc.is_in(tbl.column("congress"), value_set=pc.cast(congress_strings, tbl.schema.field("congress").type))
        tbl_filtered = tbl.filter(mask)
        if len(tbl_filtered) > 0:
            speech_chunks.append(tbl_filtered.select(["speech_id", "speech"]).to_pandas())
        del tbl, mask, tbl_filtered
    speeches = pd.concat(speech_chunks, ignore_index=True)
    del speech_chunks
    speeches["speech_id"] = speeches["speech_id"].astype(str)
    print(f"  Loaded {len(speeches):,} speeches for congresses {cong_lo}-{cong_hi}")

    # ------------------------------------------------------------------
    # 2. Build filter sets
    # ------------------------------------------------------------------
    print("\nBuilding filter sets ...")
    english_stops, geo_unigrams, blocked_bigrams, distinctive_names = build_filter_sets(VOTEVIEW_PATH)

    # ------------------------------------------------------------------
    # 3. Merge text with labels
    # ------------------------------------------------------------------
    print("\nMerging text with labels ...")

    merged = labels.merge(speeches, on="speech_id", how="inner")
    del speeches
    print(f"  Merged: {len(merged):,} speeches with text + labels")

    # Congress range already applied during load; verify
    merged = merged[merged["congress_int"].between(cong_lo, cong_hi)].reset_index(drop=True)
    print(f"  After congress filter ({cong_lo}-{cong_hi}): {len(merged):,} speeches")

    # Filter to R and D only (exclude independents)
    merged = merged[merged["party"].isin(["R", "D"])].reset_index(drop=True)
    print(f"  After R/D filter: {len(merged):,} speeches")

    # Dev subsample (for quick iteration)
    sample_frac = cfg.CONFIG.get("speech_sample_frac")
    if sample_frac is not None and 0 < sample_frac < 1:
        n_before = len(merged)
        merged = merged.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"  [DEV] Subsampled {sample_frac:.0%}: {n_before:,} -> {len(merged):,} speeches")

    # Min speech words filter
    min_words = cfg.CONFIG.get("min_speech_words")
    if min_words:
        merged["word_count"] = merged["speech"].str.split().str.len()
        merged = merged[merged["word_count"] >= min_words].reset_index(drop=True)
        print(f"  After min_speech_words={min_words}: {len(merged):,} speeches")

    # ------------------------------------------------------------------
    # 4. Optionally aggregate by legislator-congress
    # ------------------------------------------------------------------
    if AGGREGATE:
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
        texts = agg["text"].values
        party_labels = agg["party_code"].values
        meta = agg.drop(columns=["text"])
        print(f"  Legislator-congress documents: {len(meta):,}")
    else:
        print("\nUsing individual speeches ...")
        texts = merged["speech"].astype(str).values
        party_labels = merged["party"].values
        meta = merged.drop(columns=["speech"])
        print(f"  Individual speech documents: {len(meta):,}")

    n_docs = len(texts)

    # ------------------------------------------------------------------
    # 5. Fit vectorizer
    # ------------------------------------------------------------------
    print(f"\nFitting {'CountVectorizer' if USE_WIDMER else 'TfidfVectorizer'} ...")

    analyzer = build_analyzer(english_stops, geo_unigrams, blocked_bigrams,
                              distinctive_names, bigrams_only=BIGRAMS_ONLY)

    # Wrap with tqdm progress tracking
    pbar = tqdm(total=n_docs * 2, desc="  Vectorizer (fit+transform)", unit="doc")
    _orig_call = analyzer.__call__

    def _tracking_call(doc):
        result = _orig_call(doc)
        pbar.update(1)
        return result

    analyzer.__call__ = _tracking_call

    if USE_WIDMER:
        # Use CountVectorizer with low min_df as initial filter
        # (Widmer freq filter applied afterwards)
        vectorizer = CountVectorizer(
            analyzer=analyzer,
            min_df=5,  # low initial filter for memory efficiency
        )
        count_matrix = vectorizer.fit_transform(texts)
    else:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            min_df=cfg.CONFIG["tfidf_min_df"],
            sublinear_tf=cfg.CONFIG.get("tfidf_sublinear_tf", True),
        )
        count_matrix = vectorizer.fit_transform(texts)

    pbar.close()
    analyzer.__call__ = _orig_call

    feature_names = vectorizer.get_feature_names_out()
    print(f"  Initial vocabulary: {len(feature_names):,} features")
    print(f"  Matrix shape: {count_matrix.shape}")

    # ------------------------------------------------------------------
    # 6. Apply frequency filter
    # ------------------------------------------------------------------
    if FREQ_FILTER == "widmer":
        filter_idx = apply_widmer_freq_filter(count_matrix, party_labels, feature_names)
        count_matrix = count_matrix[:, filter_idx]
        filtered_features = feature_names[filter_idx]
        print(f"  Filtered matrix: {count_matrix.shape}")
    else:
        filter_idx = np.arange(len(feature_names))
        filtered_features = feature_names

    # ------------------------------------------------------------------
    # 7. Create filtered vectorizer for step 07
    # ------------------------------------------------------------------
    # Build a new vectorizer with the filtered vocabulary so step 07
    # can directly transform newspaper text to the correct feature space
    filtered_vocab = {feat: i for i, feat in enumerate(filtered_features)}
    filtered_vectorizer = CountVectorizer(
        analyzer=analyzer,
        vocabulary=filtered_vocab,
    )
    # Set vocabulary_ attribute (normally set by fit(), but we use fixed vocabulary)
    filtered_vectorizer.vocabulary_ = filtered_vocab
    filtered_vectorizer.fixed_vocabulary_ = True

    # ------------------------------------------------------------------
    # 8. Normalize to relative frequencies (Widmer) or keep TF-IDF
    # ------------------------------------------------------------------
    if USE_WIDMER:
        # L1 normalize rows: each row sums to 1 (relative frequency)
        # sklearn normalize handles zero rows gracefully (returns zero rows)
        feature_matrix = normalize(count_matrix.astype(np.float64), norm='l1', axis=1)
        n_zero = (np.asarray(count_matrix.sum(axis=1)).ravel() == 0).sum()
        print(f"  Normalized to relative frequencies (L1)")
        if n_zero > 0:
            print(f"  WARNING: {n_zero} zero-count rows (empty documents)")
    else:
        feature_matrix = count_matrix

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sp.save_npz(OUT_MATRIX, feature_matrix)
    print(f"\n  Saved feature matrix -> {OUT_MATRIX}")

    # Also save with legacy name for backward compatibility
    sp.save_npz(OUT_TFIDF, feature_matrix)

    meta.to_parquet(OUT_META)
    print(f"  Saved metadata -> {OUT_META}")
    meta.to_parquet(OUT_TFIDF_META)  # backward compat

    joblib.dump(filtered_vectorizer, OUT_VECTORIZER)
    print(f"  Saved vectorizer -> {OUT_VECTORIZER}")
    # Also save with legacy name
    joblib.dump(filtered_vectorizer, OUT_DIR / "05_tfidf_vectorizer.joblib")

    np.save(OUT_VOCAB_IDX, filter_idx)
    print(f"  Saved vocabulary filter indices -> {OUT_VOCAB_IDX}")

    # ------------------------------------------------------------------
    # 10. Validation summary
    # ------------------------------------------------------------------
    elapsed = time.time() - pipeline_start

    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    n_rep = ((party_labels == "R") | (party_labels == 200)).sum()
    n_dem = ((party_labels == "D") | (party_labels == 100)).sum()
    print(f"  Total documents: {n_docs:,} ({n_rep:,} Rep, {n_dem:,} Dem)")
    print(f"  Final vocabulary: {feature_matrix.shape[1]:,} features")

    n_bigrams = sum(1 for f in filtered_features if " " in f)
    n_unigrams = len(filtered_features) - n_bigrams
    print(f"  Unigrams: {n_unigrams:,}  |  Bigrams: {n_bigrams:,}")

    if n_bigrams > 0:
        bigram_samples = [f for f in filtered_features if " " in f][:15]
        print(f"  Sample bigrams: {bigram_samples}")

    # Per-congress breakdown
    cong_col = "congress_int"
    if cong_col in meta.columns:
        print(f"\n  Documents per congress:")
        for cong, group in meta.groupby(cong_col):
            n_r = (group["party"].isin(["R", 200])).sum() if "party" in group.columns else "?"
            n_d = (group["party"].isin(["D", 100])).sum() if "party" in group.columns else "?"
            print(f"    Congress {cong}: {len(group):,} docs ({n_r} Rep, {n_d} Dem)")

    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 72)
