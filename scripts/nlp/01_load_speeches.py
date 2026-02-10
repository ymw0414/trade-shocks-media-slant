"""
01_load_speeches.py
Load Congressional speeches and save as parquet
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

def load_speeches(raw_dir: Path) -> pd.DataFrame:
    rows = []

    for i in range(43, 112):
        suffix = f"{i:03d}"
        file = raw_dir / f"speeches_{suffix}.txt"

        if not file.exists():
            print("skip:", file)
            continue

        with open(file, encoding="cp1252") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("|", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1], suffix))

    return pd.DataFrame(
        rows,
        columns=["speech_id", "speech", "congress"]
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_speeches(RAW_DIR)

    out = OUT_DIR / "01_speeches_merged.parquet"
    df.to_parquet(out)

    print("Saved:", out)
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
