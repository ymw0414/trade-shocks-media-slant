"""
05b_append_congress98.py

Append congress 98 legislator-congress documents to the existing TF-IDF
matrix using the saved vectorizer from step 05.

This avoids re-fitting the vectorizer (which would change the vocabulary
and require re-running step 07 for newspaper TF-IDF).

Inputs:
  - data/intermediate/speeches/01_speeches_merged.parquet
  - data/intermediate/speeches/04_speeches_with_partisan_core.parquet
  - data/processed/speeches/05_tfidf_vectorizer.joblib
  - data/processed/speeches/05_tfidf_matrix.npz
  - data/processed/speeches/05_tfidf_meta.parquet

Outputs (overwrites):
  - data/processed/speeches/05_tfidf_matrix.npz
  - data/processed/speeches/05_tfidf_meta.parquet
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# Add utils to path for TextAnalyzer (needed by vectorizer unpickling)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
INTER_DIR = BASE_DIR / "data" / "intermediate" / "speeches"
PROC_DIR = BASE_DIR / "data" / "processed" / "speeches"

SPEECHES_PATH = INTER_DIR / "01_speeches_merged.parquet"
LABELS_PATH = INTER_DIR / "04_speeches_with_partisan_core.parquet"

TFIDF_PATH = PROC_DIR / "05_tfidf_matrix.npz"
META_PATH = PROC_DIR / "05_tfidf_meta.parquet"
VEC_PATH = PROC_DIR / "05_tfidf_vectorizer.joblib"

MIN_SPEECH_WORDS = 100
TARGET_CONGRESS = 98

# ------------------------------------------------------------------
# 1. Check if congress 98 already in TF-IDF
# ------------------------------------------------------------------
print("Loading existing TF-IDF metadata ...")
meta_existing = pd.read_parquet(META_PATH)
if TARGET_CONGRESS in meta_existing["congress_int"].values:
    print(f"  Congress {TARGET_CONGRESS} already in TF-IDF ({len(meta_existing)} rows). Nothing to do.")
    exit(0)
print(f"  Existing: {len(meta_existing)} legislator-congress docs, "
      f"congresses {meta_existing['congress_int'].min()}-{meta_existing['congress_int'].max()}")

# ------------------------------------------------------------------
# 2. Load congress 98 speeches with labels
# ------------------------------------------------------------------
print(f"\nLoading congress {TARGET_CONGRESS} speeches ...")
labels = pd.read_parquet(LABELS_PATH)
labels_98 = labels[labels["congress_int"] == TARGET_CONGRESS].copy()
print(f"  Labeled speeches: {len(labels_98):,}")

speeches = pd.read_parquet(SPEECHES_PATH, columns=["speech_id", "speech"])
speeches["speech_id"] = speeches["speech_id"].astype(str)
labels_98["speech_id"] = labels_98["speech_id"].astype(str)

merged = labels_98.merge(speeches, on="speech_id", how="inner")
del speeches
print(f"  Merged with text: {len(merged):,}")

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
n_rep = int(agg["label_rep_core"].sum())
n_dem = int(agg["label_dem_core"].sum())
print(f"  Congress {TARGET_CONGRESS}: {len(agg)} legislators "
      f"({n_rep} R core, {n_dem} D core)")

# ------------------------------------------------------------------
# 4. Transform with existing vectorizer
# ------------------------------------------------------------------
print("\nTransforming with saved vectorizer ...")
vectorizer = joblib.load(VEC_PATH)
X_new = vectorizer.transform(agg["text"])
print(f"  New TF-IDF: {X_new.shape}")

# ------------------------------------------------------------------
# 5. Append to existing TF-IDF
# ------------------------------------------------------------------
print("\nAppending to existing TF-IDF ...")
X_existing = sp.load_npz(TFIDF_PATH)
print(f"  Existing: {X_existing.shape}")

X_combined = sp.vstack([X_new, X_existing])  # congress 98 first (earlier)
print(f"  Combined: {X_combined.shape}")

meta_new = agg.drop(columns=["text"])
meta_combined = pd.concat([meta_new, meta_existing], ignore_index=True)
meta_combined = meta_combined.sort_values(["congress_int", "icpsr"]).reset_index(drop=True)

# Sort TF-IDF rows to match metadata order
# Build mapping from (icpsr, congress_int) to row index in combined matrix
combined_keys = pd.concat([
    meta_new[["icpsr", "congress_int"]],
    meta_existing[["icpsr", "congress_int"]]
], ignore_index=True)
sort_order = combined_keys.reset_index().merge(
    meta_combined[["icpsr", "congress_int"]].reset_index(),
    on=["icpsr", "congress_int"]
)["index_x"].values
X_combined = X_combined[sort_order]

print(f"  Final: {X_combined.shape[0]} rows, meta: {len(meta_combined)} rows")
assert X_combined.shape[0] == len(meta_combined)

# ------------------------------------------------------------------
# 6. Save (overwrite)
# ------------------------------------------------------------------
sp.save_npz(TFIDF_PATH, X_combined)
meta_combined.to_parquet(META_PATH)

print(f"\n  Saved TF-IDF -> {TFIDF_PATH}")
print(f"  Saved meta -> {META_PATH}")

# Verify
print(f"\n  Congress breakdown:")
for cong, group in meta_combined.groupby("congress_int"):
    print(f"    Congress {cong}: {len(group)} legislators, "
          f"R_core={int(group['label_rep_core'].sum())}, "
          f"D_core={int(group['label_dem_core'].sum())}")
