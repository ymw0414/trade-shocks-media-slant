"""
05c_rebuild_speech_tfidf.py

Rebuild the speech TF-IDF matrix using the SAVED vectorizer from step 05.
This fixes the corruption from 05b (which stacked matrices from different
vocabulary mappings).

All congresses are transformed with the SAME vectorizer, ensuring
consistent column→feature mapping across all rows.

Inputs:
  - data/intermediate/speeches/01_speeches_merged.parquet
  - data/intermediate/speeches/04_speeches_with_partisan_core.parquet
  - data/processed/speeches/05_tfidf_vectorizer.joblib

Outputs (overwrites):
  - data/processed/speeches/05_tfidf_matrix.npz
  - data/processed/speeches/05_tfidf_meta.parquet
"""

import os
import sys
import gc
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
INTER_DIR = BASE_DIR / "data" / "intermediate" / "speeches"
PROC_DIR = BASE_DIR / "data" / "processed" / "speeches"

SPEECHES_PATH = INTER_DIR / "01_speeches_merged.parquet"
LABELS_PATH = INTER_DIR / "04_speeches_with_partisan_core.parquet"
VEC_PATH = PROC_DIR / "05_tfidf_vectorizer.joblib"
TFIDF_PATH = PROC_DIR / "05_tfidf_matrix.npz"
META_PATH = PROC_DIR / "05_tfidf_meta.parquet"

MIN_SPEECH_WORDS = 100
CONGRESS_MIN = 98  # Include congress 98 for 3-window robustness
CONGRESS_MAX = 108

# ------------------------------------------------------------------
# 1. Load labels and filter
# ------------------------------------------------------------------
print("Loading partisan-core labels ...")
labels = pd.read_parquet(LABELS_PATH)
labels = labels[
    (labels["congress_int"] >= CONGRESS_MIN)
    & (labels["congress_int"] <= CONGRESS_MAX)
].copy()
print(f"  Labels: {len(labels):,} speeches, "
      f"congresses {labels['congress_int'].min()}-{labels['congress_int'].max()}")

# ------------------------------------------------------------------
# 2. Load speech text and merge
# ------------------------------------------------------------------
print("\nLoading speech text ...")
speeches = pd.read_parquet(SPEECHES_PATH, columns=["speech_id", "speech"])
speeches["speech_id"] = speeches["speech_id"].astype(str)
labels["speech_id"] = labels["speech_id"].astype(str)

merged = labels.merge(speeches, on="speech_id", how="inner")
del speeches
gc.collect()
print(f"  Merged: {len(merged):,}")

# Filter short speeches
wc = merged["speech"].astype(str).str.count(" ") + 1
merged = merged[wc >= MIN_SPEECH_WORDS].copy()
print(f"  After word count filter (>={MIN_SPEECH_WORDS}): {len(merged):,}")

# ------------------------------------------------------------------
# 3. Aggregate by legislator-congress
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
del merged
gc.collect()

for cong in sorted(agg["congress_int"].unique()):
    sub = agg[agg["congress_int"] == cong]
    n_rep = int(sub["label_rep_core"].sum())
    n_dem = int(sub["label_dem_core"].sum())
    print(f"  Congress {cong}: {len(sub)} legislators ({n_rep} R core, {n_dem} D core)")

# ------------------------------------------------------------------
# 4. Transform with saved vectorizer
# ------------------------------------------------------------------
print("\nLoading vectorizer ...")
vectorizer = joblib.load(VEC_PATH)
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

print("Transforming ...")
X = vectorizer.transform(agg["text"])
print(f"  TF-IDF shape: {X.shape}")

# ------------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------------
meta = agg.drop(columns=["text"])
meta = meta.sort_values(["congress_int", "icpsr"]).reset_index(drop=True)

# Sort TF-IDF rows to match metadata order
sort_idx = agg.sort_values(["congress_int", "icpsr"]).index.values
X = X[sort_idx]

print(f"\n  Final: {X.shape[0]} rows, {len(meta)} meta rows")
assert X.shape[0] == len(meta)

sp.save_npz(TFIDF_PATH, X)
meta.to_parquet(META_PATH)

print(f"\n  Saved TF-IDF -> {TFIDF_PATH}")
print(f"  Saved meta -> {META_PATH}")

# Verify
print(f"\n  Congress breakdown:")
for cong, group in meta.groupby("congress_int"):
    print(f"    Congress {cong}: {len(group)} legislators, "
          f"R_core={int(group['label_rep_core'].sum())}, "
          f"D_core={int(group['label_dem_core'].sum())}")
