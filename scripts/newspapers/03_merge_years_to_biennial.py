"""
03_merge_yearly_to_biennial.py

Merge yearly newspaper parquet files into biennial (Congress) files.
"""

import pandas as pd
from pathlib import Path

YEARLY_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\intermediate\newspapers\yearly"
)

OUT_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\intermediate\newspapers\biennial"
)

CONGRESS_MAP = {
    1980: "96", 1981: "97", 1982: "97", 1983: "98",
    1984: "98", 1985: "99", 1986: "99", 1987: "100",
    1988: "100", 1989: "101", 1990: "101", 1991: "102",
    1992: "102", 1993: "103", 1994: "103", 1995: "104",
    1996: "104", 1997: "105", 1998: "105", 1999: "106",
    2000: "106", 2001: "107", 2002: "107", 2003: "108",
    2004: "108", 2005: "109", 2006: "109",
    2007: "110", 2008: "110",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = {}

    for f in YEARLY_DIR.glob("newspapers_*.parquet"):
        year = int(f.stem.split("_")[-1])
        congress = CONGRESS_MAP.get(year)

        if congress is None:
            continue

        df = pd.read_parquet(f)
        df["year"] = year
        df["congress"] = congress

        groups.setdefault(congress, []).append(df)

    for congress, dfs in groups.items():
        merged = pd.concat(dfs, ignore_index=True)
        out = OUT_DIR / f"newspapers_congress_{congress}.parquet"
        merged.to_parquet(out)
        print("Saved:", out, merged.shape)


if __name__ == "__main__":
    main()
