"""
01_load_speeches.py

This script loads all Congressional speech files (speeches_043.txt ~ speeches_111.txt),
parses them safely using a custom splitter, and stores the merged dataset as a parquet file.

Output:
    data/processed/speeches_merged.parquet
"""

import pandas as pd
from pathlib import Path

speech_dir = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\Congressional Speech Record Data\hein-bound"
)

all_rows = []

for i in range(43, 112):
    suffix = f"{i:03d}"
    file = speech_dir / f"speeches_{suffix}.txt"

    if not file.exists():
        print("skip:", file)
        continue

    # Using cp1252 due to historical encoding of Congressional Record texts
    with open(file, "r", encoding="cp1252") as f:
        header = next(f)  # skip header

        for line in f:
            line = line.rstrip("\n")
            parts = line.split("|", 1)  # split only at the first pipe

            if len(parts) != 2:
                continue

            speech_id, speech = parts
            all_rows.append((speech_id.strip(), speech.strip(), suffix))

df = pd.DataFrame(all_rows, columns=["speech_id", "speech", "file_id"])

out = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speeches_merged.parquet"
)
df.to_parquet(out)

print("Done.", df.shape)
print("Saved to:", out)
