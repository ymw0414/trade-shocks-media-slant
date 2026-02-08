"""
02_merge_daily_to_year.py

Merge daily newspaper CSV files (2005–2008) into yearly files,
standardize schema to match Congress 107–108,
and report how many rows are skipped.
"""

import pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError

RAW_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\newspapers\scrapes_raw"
)

OUT_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\intermediate\newspapers\yearly"
)

CANON_COLS = [
    "date",
    "paper",
    "text",
    "title",
    "word_count",
    "year",
]

def count_rows_fast(path):
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(2005, 2009):
        files = sorted(RAW_DIR.glob(f"{year}-*.csv"))
        print(f"Processing {year}, files={len(files)}")

        if not files:
            continue

        total_raw = 0
        total_read = 0
        dfs = []

        for f in files:
            raw_rows = count_rows_fast(f)
            total_raw += raw_rows

            try:
                df = pd.read_csv(
                    f,
                    engine="python",
                    on_bad_lines="skip"
                )
            except EmptyDataError:
                print(f"{f.name}: EMPTY FILE, skipped entirely")
                continue

            read_rows = len(df)
            total_read += read_rows
            skipped = raw_rows - read_rows
            print(f"{f.name}: skipped {skipped}")

            if read_rows == 0:
                continue

            # schema standardization
            df = df.rename(columns={
                "paragraph": "text",
                "office": "paper",
                "words": "word_count",
            })

            for c in CANON_COLS:
                if c not in df.columns:
                    df[c] = pd.NA

            df["year"] = year
            df = df[CANON_COLS]

            dfs.append(df)

        if total_read == 0:
            print(f"WARNING: {year} has zero valid rows, no parquet written")
            continue

        merged = pd.concat(dfs, ignore_index=True)

        out = OUT_DIR / f"newspapers_{year}.parquet"
        merged.to_parquet(out)

        print(
            f"Year {year}: raw={total_raw}, kept={total_read}, "
            f"skipped={total_raw - total_read}"
        )
        print("-" * 40)

if __name__ == "__main__":
    main()
