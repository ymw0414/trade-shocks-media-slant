"""
01_standardize_yearly_csv.py

Convert raw yearly newspaper CSV files (1980-2004) to parquet format
with unified filenames.

Input:  data/raw/newspapers/scrapes_since_1980/*.csv
Output: data/intermediate/newspapers/yearly/newspapers_{year}.parquet
"""

import os
import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
RAW_DIR = BASE_DIR / "data" / "raw" / "newspapers" / "scrapes_since_1980"
OUT_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly"

YEAR_RE = re.compile(r"(19|20)\d{2}")
MAX_YEAR = 2004  # analysis period ends here


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))

    for f in files:
        m = YEAR_RE.search(f.stem)
        if not m:
            print("skip (no year):", f.name)
            continue

        year = int(m.group())
        if year > MAX_YEAR:
            continue

        df = pd.read_csv(
            f,
            engine="python",
            on_bad_lines="skip",
        )

        out = OUT_DIR / f"newspapers_{year}.parquet"
        df.to_parquet(out)

        print(f"  {year}: {df.shape[0]:,} rows -> {out.name}")


if __name__ == "__main__":
    main()
