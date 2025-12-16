"""
02_merge_speaker_map.py

This script loads all SpeakerMap files (SpeakerMap_043.txt ~ SpeakerMap_111.txt),
parses them, and stores the merged speaker metadata as a parquet file.

Output:
    data/processed/speaker_map.parquet
"""

import pandas as pd
from pathlib import Path

# Directory containing SpeakerMap files
speaker_dir = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\Congressional Speech Record Data\hein-bound"
)

all_rows = []

for i in range(43, 112):
    suffix = f"{i:03d}"
    file = speaker_dir / f"SpeakerMap_{suffix}.txt"

    if not file.exists():
        print("skip:", file)
        continue

    # SpeakerMap files follow same encoding and delimiter format as the speeches
    df = pd.read_csv(
        file,
        sep="|",
        header=None,
        names=["speech_id", "speaker", "state", "party"],
        dtype=str,
        encoding="cp1252",
    )

    all_rows.append(df)

# Combine into single dataframe
speaker_map = pd.concat(all_rows, ignore_index=True)

# Output location
out = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speaker_map.parquet"
)

speaker_map.to_parquet(out)

print("Done.", speaker_map.shape)
print("Saved to:", out)
