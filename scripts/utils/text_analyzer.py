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
