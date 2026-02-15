import pandas as pd
import numpy as np

# Load data
print("Loading speech data...")
df = pd.read_parquet("C:/Users/ymw04/Dropbox/shifting_slant/data/intermediate/speeches/01_speeches_merged.parquet")
print(f"Loaded {len(df):,} speeches")
print(f"Columns: {list(df.columns)}")
print()

# Compute word count
df["word_count"] = df["speech"].str.split().str.len()

# ── 1. Distribution: percentiles, mean, total ──
print("=" * 60)
print("SPEECH LENGTH DISTRIBUTION (word count)")
print("=" * 60)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
pct_values = np.percentile(df["word_count"].dropna(), percentiles)
for p, v in zip(percentiles, pct_values):
    print(f"  P{p:>2d}:  {v:>8.0f} words")
print(f"  Mean: {df['word_count'].mean():>8.1f} words")
print(f"  Std:  {df['word_count'].std():>8.1f} words")
print(f"  Min:  {df['word_count'].min():>8d} words")
print(f"  Max:  {df['word_count'].max():>8d} words")
print(f"  Total speeches: {len(df):,}")
print()

# ── 2. Counts below thresholds ──
total_n = len(df)
total_words = df["word_count"].sum()

print("=" * 60)
print("SPEECHES BELOW WORD-COUNT THRESHOLDS")
print("=" * 60)
print(f"{'Threshold':>12s}  {'Count':>10s}  {'% Speeches':>10s}  {'% Total Words':>13s}")
print("-" * 50)
for threshold in [50, 100, 150, 200]:
    mask = df["word_count"] < threshold
    n = mask.sum()
    pct_speeches = 100.0 * n / total_n
    pct_words = 100.0 * df.loc[mask, "word_count"].sum() / total_words
    print(f"  < {threshold:>4d} words  {n:>10,d}  {pct_speeches:>9.2f}%  {pct_words:>12.2f}%")
print()

# ── 3. Examples of short speeches (< 50 words) ──
print("=" * 60)
print("EXAMPLES: SHORT SPEECHES (< 50 words)")
print("=" * 60)
short = df[df["word_count"] < 50].sample(n=min(8, (df["word_count"] < 50).sum()), random_state=42)
for i, (idx, row) in enumerate(short.iterrows()):
    print(f"\n--- Example {i+1} ({row['word_count']} words) ---")
    # Show available metadata
    meta_cols = [c for c in ["speech_id", "speakerid", "congress", "chamber"] if c in df.columns]
    for c in meta_cols:
        print(f"  {c}: {row[c]}")
    print(f"  Text: {row['speech'][:300]}")
print()

# ── 4. Examples of medium speeches (90-110 words) ──
print("=" * 60)
print("EXAMPLES: MEDIUM SPEECHES (90-110 words)")
print("=" * 60)
medium = df[(df["word_count"] >= 90) & (df["word_count"] <= 110)].sample(n=min(5, ((df["word_count"] >= 90) & (df["word_count"] <= 110)).sum()), random_state=42)
for i, (idx, row) in enumerate(medium.iterrows()):
    print(f"\n--- Example {i+1} ({row['word_count']} words) ---")
    meta_cols = [c for c in ["speech_id", "speakerid", "congress", "chamber"] if c in df.columns]
    for c in meta_cols:
        print(f"  {c}: {row[c]}")
    print(f"  Text: {row['speech'][:500]}")
print()

print("Done.")
