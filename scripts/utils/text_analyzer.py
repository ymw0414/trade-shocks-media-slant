"""
text_analyzer.py

Picklable TF-IDF analyzer for congressional speech / newspaper text.
Used by 05_build_tfidf.py (fitting) and 07_prepare_newspapers.py (transform).

This module exists so that joblib can serialize and deserialize the
analyzer across scripts.  Closures cannot be pickled; a callable class can.
"""

import re
from nltk.stem.porter import PorterStemmer


# Parliamentary phrases to strip (must match 05_build_tfidf.py)
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


class TextAnalyzer:
    """
    Picklable TF-IDF analyzer with Porter Stemmer + filtering.

    Pipeline per document:
      1. Lowercase
      2. Remove parliamentary phrases (single regex pass)
      3. Tokenize (alphabetic tokens >= 2 chars)
      4. Stem (cached) + filter stop words and geographic terms
      5. Form unigrams + bigrams
      6. Filter blocked bigrams (geographic + legislator names)
    """

    def __init__(self, single_stops, blocked_bigrams):
        self.single_stops = single_stops
        self.blocked_bigrams = blocked_bigrams
        self.stemmer = PorterStemmer()
        self.stem_cache = {}
        self.token_re = re.compile(r"[a-z]+")
        self.parl_pattern = re.compile(
            "|".join(re.escape(p) for p in PARLIAMENTARY_PHRASES),
            re.IGNORECASE,
        )

    def __call__(self, doc):
        text = doc.lower()
        text = self.parl_pattern.sub(" ", text)
        raw_tokens = self.token_re.findall(text)

        stemmed = []
        for t in raw_tokens:
            if len(t) < 2:
                continue
            s = self.stem_cache.get(t)
            if s is None:
                s = self.stemmer.stem(t)
                self.stem_cache[t] = s
            if len(s) >= 2 and s not in self.single_stops:
                stemmed.append(s)

        features = list(stemmed)
        for i in range(len(stemmed) - 1):
            bg = stemmed[i] + " " + stemmed[i + 1]
            if bg not in self.blocked_bigrams:
                features.append(bg)

        return features
