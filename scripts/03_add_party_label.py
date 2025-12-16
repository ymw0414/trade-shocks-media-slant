"""
03_add_party_label.py

This script merges the speech dataset with SpeakerMap metadata,
adding party labels to each speech based on speech_id.

Output:
    data/processed/speeches_with_party.parquet
"""

import pandas as pd
from pathlib import Path

speech_path = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speeches_merged.parquet"
)

speaker_map_path = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speaker_map.parquet"
)

# Load both datasets
speeches = pd.read_parquet(speech_path)
speaker_map = pd.read_parquet(speaker_map_path)

# Merge on speech_id
merged = speeches.merge(speaker_map, on="speech_id", how="left")

out_path = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speeches_with_party.parquet"
)
merged.to_parquet(out_path)

print("Done.", merged.shape)
print("Saved to:", out_path)
