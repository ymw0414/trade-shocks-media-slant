"""
04_preprocess_text_1980s_paragraph_filtered.py

This script preprocesses congressional speech data by:

1. Selecting ONLY speeches from the 1980s decade
   (97th–100th Congress).

2. Applying paragraph-level filtering:
       - At least 2 sentences
       - At least 200 characters
   This removes procedural floor remarks and short, low-signal statements.

3. Cleaning text using Unicode normalization and whitespace reduction.

The resulting dataset contains high-quality, content-rich speeches suitable
for downstream classification tasks.

Output:
    data/processed/speeches_clean_1980s_paragraph.parquet
"""


import pandas as pd
import re
import unicodedata
from pathlib import Path

# -------------------------------
# 1. Paths (LOCAL version)
# -------------------------------
BASE = Path("C:/Users/ymw04/Dropbox/shifting_slant")

INPUT_PATH = BASE / "data/processed/speeches_with_party.parquet"
OUTPUT_PATH = BASE / "data/processed/speeches_clean_1980s.parquet"

# -------------------------------
# 2. Load full dataset
# -------------------------------
df = pd.read_parquet(INPUT_PATH)
print("Loaded:", df.shape)

# -------------------------------
# 3. Convert file_id to integer
# -------------------------------
df["file_id"] = df["file_id"].astype(int)
print("file_id dtype:", df["file_id"].dtype)

# -------------------------------
# 4. Keep ONLY 1980s decade (97–100 Congress)
# -------------------------------
df = df[df["file_id"].isin([97, 98, 99, 100])]
print("After filtering for 1980s (file_id 97–100):", df.shape)

# -------------------------------
# 5. Keep only valid party labels
# -------------------------------
df = df[df["party"].isin(["D", "R"])]
print("After party filter:", df.shape)

# -------------------------------
# 6. Drop missing / empty text
# -------------------------------
df = df.dropna(subset=["speech"])
df = df[df["speech"].str.strip().str.len() > 0]
print("After removing empty speeches:", df.shape)

# -------------------------------
# 7. Keep only paragraph-level speeches
#    Condition: ≥2 sentences AND ≥200 characters
# -------------------------------
df["num_sent"] = df["speech"].str.count(r"[\.!?]")
df = df[(df["num_sent"] >= 2) & (df["speech"].str.len() >= 200)]
print("After paragraph filtering:", df.shape)

# -------------------------------
# 8. Unicode normalization
# -------------------------------
def fix_unicode(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKC", text)

df["speech"] = df["speech"].apply(fix_unicode)

# -------------------------------
# 9. Whitespace normalization
# -------------------------------
def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

df["speech"] = df["speech"].apply(clean_whitespace)

# -------------------------------
# 10. Save cleaned dataset
# -------------------------------
df.to_parquet(OUTPUT_PATH)
print("Saved cleaned dataset:", df.shape)
print("Saved to:", OUTPUT_PATH)

print(df.head())
