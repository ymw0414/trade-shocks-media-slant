"""
02_merge_speaker_map.py

Load SpeakerMap files correctly and save speaker-party mapping.
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths (OS-agnostic via env var)
# --------------------------------------------------

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "speeches"
    / "hein-bound"
)

OUT_DIR = (
    BASE_DIR
    / "data"
    / "intermediate"
    / "speeches"
)

# --------------------------------------------------
# Load function
# --------------------------------------------------

def load_speaker_map(raw_dir: Path) -> pd.DataFrame:
    dfs = []

    for i in range(43, 112):
        suffix = f"{i:03d}"
        file = raw_dir / f"{suffix}_SpeakerMap.txt"

        if not file.exists():
            print("skip:", file)
            continue

        df = pd.read_csv(
            file,
            sep="|",
            dtype=str,
            encoding="cp1252",
        )

        df["congress"] = suffix
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No SpeakerMap files loaded")

    return pd.concat(dfs, ignore_index=True)

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_speaker_map(RAW_DIR)

    out = OUT_DIR / "02_speaker_map.parquet"
    df.to_parquet(out)

    print("Saved:", out)
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()