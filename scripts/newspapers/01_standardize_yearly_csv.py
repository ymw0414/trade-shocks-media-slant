"""
01_standardize_yearly_csv.py

Standardize already-yearly newspaper CSV files
by converting them to parquet with unified filenames.
"""

import pandas as pd
from pathlib import Path
import re

RAW_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\newspapers\scrapes_since_1980"
)

OUT_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\intermediate\newspapers\yearly"
)

YEAR_RE = re.compile(r"(19|20)\d{2}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))

    for f in files:
        m = YEAR_RE.search(f.stem)
        if not m:
            print("skip (no year):", f.name)
            continue

        year = m.group()

        df = pd.read_csv(
            f,
            engine="python",
            on_bad_lines="skip"
        )

        out = OUT_DIR / f"newspapers_{year}.parquet"
        df.to_parquet(out)

        print("Saved:", out, df.shape)


if __name__ == "__main__":
    main()
