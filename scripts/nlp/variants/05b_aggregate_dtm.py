"""
05b_aggregate_dtm.py

Take the speech-level DTM from step 05 (exp_unigram_gst) and aggregate
to legislator-congress level. This allows sharing the vocabulary and
newspaper transform (step 07) across speech-level and legislator-level
experiments.

Outputs:
  - 05_tfidf_matrix.npz  (legislator-congress level, L1 normalized)
  - 05_tfidf_meta.parquet (legislator-congress metadata)
  - Copies the vectorizer from the source run

Usage:
  Set PIPELINE_CONFIG_OVERRIDE to an experiment config with
  aggregate_to_legislator=True. Also set input_speech_dir to point
  to the speech-level run's speeches directory.
"""

import os
import sys
import time
import joblib
import shutil
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_config as cfg

# Source: speech-level DTM
SRC_DIR = cfg.INPUT_SPEECH_DIR
SRC_MATRIX = SRC_DIR / "05_tfidf_matrix.npz"
SRC_META = SRC_DIR / "05_tfidf_meta.parquet"
SRC_VECTORIZER = SRC_DIR / "05_tfidf_vectorizer.joblib"
SRC_VOCAB_IDX = SRC_DIR / "05_vocab_filter_idx.npy"

# Destination: legislator-level DTM
DST_DIR = cfg.SPEECH_DIR
DST_MATRIX = DST_DIR / "05_tfidf_matrix.npz"
DST_META = DST_DIR / "05_tfidf_meta.parquet"
DST_VECTORIZER = DST_DIR / "05_tfidf_vectorizer.joblib"
DST_VOCAB_IDX = DST_DIR / "05_vocab_filter_idx.npy"

cfg.save_config()

print(f"Source (speech-level): {SRC_DIR}")
print(f"Destination (legislator-level): {DST_DIR}")

# Load speech-level data
print("\nLoading speech-level DTM ...")
# Load the RAW count matrix if available, otherwise use the L1-normalized one
# We need raw counts to properly aggregate to legislator level
# Since step 05 only saves L1-normalized, we'll re-load the raw counts
# by reading the feature matrix and "un-normalizing" using row sums
# Actually, for L1-normalized data, x_ij = count_ij / sum_j(count_ij)
# We can't recover exact counts, but we can aggregate the normalized vectors
# and re-normalize. This is equivalent to weighting each speech equally
# (not by length), which is actually a reasonable choice.

X_speech = sp.load_npz(SRC_MATRIX)
meta = pd.read_parquet(SRC_META)
print(f"  Speech-level: {X_speech.shape[0]:,} docs x {X_speech.shape[1]:,} features")

# Aggregate by legislator-congress
print("\nAggregating to legislator-congress level ...")
# Group key: (icpsr, congress_int)
meta = meta.reset_index(drop=True)
group_keys = meta.groupby(["icpsr", "congress_int"]).ngroup()
n_groups = group_keys.max() + 1

# Build aggregation matrix: G[g, i] = 1 if speech i belongs to group g
rows = group_keys.values
cols = np.arange(len(meta))
data = np.ones(len(meta))
G = sp.csr_matrix((data, (rows, cols)), shape=(n_groups, len(meta)))

# Aggregate: sum speech-level feature vectors per legislator-congress
# Then L1 normalize (each legislator-congress sums to 1)
X_agg_raw = G @ X_speech
X_agg = normalize(X_agg_raw.astype(np.float64), norm='l1', axis=1)
n_zero = (np.asarray(X_agg_raw.sum(axis=1)).ravel() == 0).sum()

print(f"  Aggregated: {X_agg.shape[0]:,} legislator-congress docs x {X_agg.shape[1]:,} features")
if n_zero > 0:
    print(f"  WARNING: {n_zero} zero-count rows")

# Build legislator-level metadata
print("\nBuilding legislator-level metadata ...")
leg_meta = (
    meta.groupby(["icpsr", "congress_int"])
    .agg(
        party_code=("party", "first"),
        label_rep_core=("label_rep_core", "max"),
        label_dem_core=("label_dem_core", "max"),
        n_speeches=("speech_id" if "speech_id" in meta.columns else meta.columns[0], "count"),
    )
    .reset_index()
)

# Rename party column for compatibility
if "party" not in leg_meta.columns and "party_code" in leg_meta.columns:
    pass  # step 06 handles both party and party_code

# Add nokken_poole_dim1 if available
if "nokken_poole_dim1" in meta.columns:
    nk = meta.groupby(["icpsr", "congress_int"])["nokken_poole_dim1"].first().reset_index()
    leg_meta = leg_meta.merge(nk, on=["icpsr", "congress_int"], how="left")

print(f"  Legislator-congress metadata: {len(leg_meta):,} rows")
n_rep = (leg_meta["party_code"].isin(["R", 200])).sum()
n_dem = (leg_meta["party_code"].isin(["D", 100])).sum()
print(f"  {n_rep:,} Rep, {n_dem:,} Dem")

# Verify alignment
assert len(leg_meta) == X_agg.shape[0], \
    f"Metadata rows ({len(leg_meta)}) != matrix rows ({X_agg.shape[0]})"

# Save
DST_DIR.mkdir(parents=True, exist_ok=True)

sp.save_npz(DST_MATRIX, X_agg)
# Also save with other names for backward compatibility
sp.save_npz(DST_DIR / "05_feature_matrix.npz", X_agg)
print(f"\n  Saved DTM -> {DST_MATRIX}")

leg_meta.to_parquet(DST_META)
leg_meta.to_parquet(DST_DIR / "05_speech_meta.parquet")
print(f"  Saved meta -> {DST_META}")

# Copy vectorizer and vocab idx from source (same vocabulary!)
shutil.copy2(SRC_VECTORIZER, DST_VECTORIZER)
shutil.copy2(SRC_VECTORIZER, DST_DIR / "05_vectorizer.joblib")
print(f"  Copied vectorizer -> {DST_VECTORIZER}")

if SRC_VOCAB_IDX.exists():
    shutil.copy2(SRC_VOCAB_IDX, DST_VOCAB_IDX)
    shutil.copy2(SRC_VOCAB_IDX, DST_DIR / "05_vocab_filter_idx.npy")

# Per-congress breakdown
print(f"\n  Documents per congress:")
for cong, group in leg_meta.groupby("congress_int"):
    nr = (group["party_code"].isin(["R", 200])).sum()
    nd = (group["party_code"].isin(["D", 100])).sum()
    print(f"    Congress {cong}: {len(group):,} legislators ({nr} Rep, {nd} Dem)")

print("\nDone.")
