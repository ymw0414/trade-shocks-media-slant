"""
05_build_tfidf.py

Build a uniform TF-IDF matrix from congressional speeches (Congresses 99-108)
with comprehensive text preprocessing to isolate genuine partisan language.

Preprocessing pipeline:
  1. Remove parliamentary procedure phrases from raw text (expanded list)
  2. Tokenize (lowercase, alphabetic tokens >= 2 chars)
  3. Remove English stop words (sklearn default list)
  4. Remove procedural/legal stop words (thereto, thereof, hereby, etc.)
  5. Remove geographic terms (state names, major cities, state abbreviations)
  6. Remove distinctive legislator name tokens (stemmed surnames that are
     NOT common English words, e.g. "dannemeyer" but NOT "young")
  7. Apply Porter Stemmer to all remaining tokens
  8. Form unigrams and bigrams
  9. Filter out legislator full-name bigrams, including nicknames (stemmed)

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

OUT_DIR = BASE_DIR / "data" / "processed" / "speeches"
OUT_TFIDF = OUT_DIR / "05_tfidf_matrix.npz"
OUT_META = OUT_DIR / "05_tfidf_meta.parquet"
OUT_VECTORIZER = OUT_DIR / "05_tfidf_vectorizer.joblib"

# ------------------------------------------------------------------
# Parliamentary procedure phrases to remove (must match text_analyzer.py)
# ------------------------------------------------------------------
PARLIAMENTARY_PHRASES = [
    # --- Yielding time ---
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
    # --- Forms of address ---
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
    "the speaker pro tempore",
    "the speaker announced",
    "the speaker appoints",
    "the speaker laid before the house",
    # --- Pursuant / rules ---
    "pursuant to house rule",
    "pursuant to clause",
    "pursuant to the rule",
    "pursuant to the order",
    "pursuant to the provisions",
    # --- Unanimous consent ---
    "i ask unanimous consent",
    "without objection so ordered",
    "without objection",
    # --- Questions and voting ---
    "the question is on",
    "the yeas and nays are ordered",
    "the previous question is ordered",
    "the previous question",
    "the question was taken",
    "the question recurs",
    "i demand a recorded vote",
    "a recorded vote is demanded",
    "demand a recorded vote",
    "i demand a second",
    "demand a second",
    "the yeas and nays",
    "a recorded vote",
    "a rollcall vote",
    "by a recorded vote",
    "on this vote",
    # --- Rising ---
    "i rise today",
    "i rise in support",
    "i rise in opposition",
    "i rise to",
    # --- Reserving time ---
    "i reserve the balance of my time",
    # --- Record ---
    "under a previous order of the house",
    "i include for the record",
    "i insert for the record",
    "permission to revise and extend",
    "revise and extend my remarks",
    "i move to strike the last word",
    # --- Time expiration ---
    "the time of the gentleman has expired",
    "the time of the gentlewoman has expired",
    # --- Motions ---
    "motion to table",
    "motion to reconsider",
    "motion to recommit",
    "motion to suspend the rules",
    "motion to instruct conferees",
    "i move to suspend the rules",
    "i offer a motion",
    "suspend the rules and pass",
    "suspend the rules and agree",
    "suspend the rules and concur",
    # --- Amendments ---
    "amendment thereto",
    "amendments thereto",
    "amendment offered by",
    "amendment in the nature of a substitute",
    "for the purpose of amendment",
    "purpose of the amendment",
    "amendment to the amendment",
    # --- Extensions and leaves ---
    "extension of remarks",
    "leave to extend",
    "leave to revise and extend",
    "general leave",
    "leave of absence",
    # --- Committee procedures ---
    "permit the committee to sit",
    "permitted to sit",
    "the committee on rules",
    "reported the bill",
    "ordered to be reported",
    # --- Objections / reservations ---
    "is there objection",
    "objection is heard",
    "hearing none",
    "reserving the right to object",
    "i reserve the right to object",
    "i withdraw my reservation",
    # --- Ordering / printing ---
    "so ordered",
    "ordered to be printed",
    "ordered to lie on the table",
    "engrossment and third reading",
    "a quorum is present",
    "a quorum is not present",
    # --- Clerk ---
    "the clerk will read",
    "the clerk will report",
    "the clerk will call the roll",
    "the clerk read",
    "the clerk reported",
    "the roll was called",
    # --- One-minute / special order ---
    "one minute speech",
    "one minute speeches",
    "morning hour",
    "special order",
    "special orders",
    # --- Further proceedings ---
    "further proceedings",
    "further reading",
]

# ------------------------------------------------------------------
# Procedural stop words (legal/parliamentary terms, no partisan content)
# ------------------------------------------------------------------
PROCEDURAL_STOP_WORDS = [
    "thereto", "thereof", "hereby", "herein", "herewith",
    "whereas", "aforesaid", "notwithstanding", "therein",
    "heretofore", "hereunder", "thereunder", "thereupon",
    "whereupon", "inasmuch", "wherefor",
]

# ------------------------------------------------------------------
# Common English words that are also legislator surnames -- keep as
# valid tokens (do NOT block as distinctive name tokens)
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

    # --- Procedural stop words (stemmed) ---
    procedural_stops_stemmed = {stemmer.stem(w) for w in PROCEDURAL_STOP_WORDS}
    english_stops_stemmed |= procedural_stops_stemmed

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

    # Build safe-name stems: common English words that also happen to be
    # legislator surnames.  These should NOT be blocked as stop words.
    safe_name_stems = {stemmer.stem(w) for w in COMMON_ENGLISH_WORDS_ALSO_NAMES}
    safe_name_stems |= english_stops_stemmed  # stop words already filtered

    legislator_bigrams_stemmed = set()
    distinctive_name_tokens = set()

    for bioname in house["bioname"].dropna().unique():
        # bioname format: "LASTNAME, Firstname Middle (Nickname), Suffix"
        parts = bioname.split(",")
        if len(parts) < 2:
            continue
        lastname = parts[0].strip().lower()
        firstname_part = parts[1].strip().lower()

        # Extract first token of firstname
        firstname_tokens = re.sub(r"[^a-z ]", "", firstname_part).split()
        if not firstname_tokens:
            continue
        firstname = firstname_tokens[0]
        lastname_clean = re.sub(r"[^a-z]", "", lastname)

        # Extract nickname from parenthetical: (Nickname)
        nickname_match = re.search(r"\(([^)]+)\)", parts[1] if len(parts) > 1 else "")
        nickname = None
        if nickname_match:
            nick_raw = nickname_match.group(1).strip().lower()
            nick_clean = re.sub(r"[^a-z]", "", nick_raw)
            if len(nick_clean) >= 2:
                nickname = nick_clean

        # Full name bigrams: "firstname lastname" and "lastname firstname"
        if len(firstname) >= 2 and len(lastname_clean) >= 2:
            fn_stem = stemmer.stem(firstname)
            ln_stem = stemmer.stem(lastname_clean)
            legislator_bigrams_stemmed.add(fn_stem + " " + ln_stem)
            legislator_bigrams_stemmed.add(ln_stem + " " + fn_stem)

            # Nickname bigrams: "nickname lastname" and "lastname nickname"
            if nickname:
                nn_stem = stemmer.stem(nickname)
                legislator_bigrams_stemmed.add(nn_stem + " " + ln_stem)
                legislator_bigrams_stemmed.add(ln_stem + " " + nn_stem)

        # Distinctive name tokens: stemmed last names that are NOT common
        # English words.  These are added as stop words so they cannot
        # appear as unigrams or in any bigram.
        if len(lastname_clean) >= 2:
            ln_stem = stemmer.stem(lastname_clean)
            if len(ln_stem) >= 4 and ln_stem not in safe_name_stems:
                distinctive_name_tokens.add(ln_stem)

    # Combine blocked bigrams
    blocked_bigrams = geo_bigrams_stemmed | legislator_bigrams_stemmed

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
# Import picklable TextAnalyzer class (allows joblib serialization)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from text_analyzer import TextAnalyzer


def build_analyzer(english_stops, geo_unigrams, blocked_bigrams,
                   distinctive_name_tokens=None):
    """
    Return a picklable callable analyzer for TfidfVectorizer.

    Uses TextAnalyzer (a class with __call__) so that the fitted
    vectorizer can be saved with joblib and loaded by other scripts.

    Distinctive name tokens are added to single_stops so that they are
    removed as both unigrams and from any bigrams they would form.
    """
    single_stops = english_stops | geo_unigrams
    if distinctive_name_tokens:
        single_stops = single_stops | distinctive_name_tokens
    return TextAnalyzer(
        single_stops=single_stops,
        blocked_bigrams=blocked_bigrams,
    )


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
    english_stops, geo_unigrams, blocked_bigrams, distinctive_names = build_filter_sets(VOTEVIEW_PATH)

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

    analyzer = build_analyzer(english_stops, geo_unigrams, blocked_bigrams, distinctive_names)

    # Wrap with tqdm progress tracking (for display only)
    n_docs = len(agg)
    pbar = tqdm(total=n_docs * 2, desc="  TF-IDF (fit+transform)", unit="doc")
    _orig_call = analyzer.__call__

    def _tracking_call(doc):
        result = _orig_call(doc)
        pbar.update(1)
        return result

    analyzer.__call__ = _tracking_call

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        min_df=0.001,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(agg["text"])
    pbar.close()

    # Restore original __call__ so the saved analyzer is clean
    analyzer.__call__ = _orig_call

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
