"""
check_2008_presence.py

Check whether 2008 daily CSV files exist and match glob patterns.
"""

from pathlib import Path

RAW_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\raw\newspapers\scrapes_raw"
)


def main():
    all_2008 = sorted(RAW_DIR.glob("*2008*.csv"))
    dash_2008 = sorted(RAW_DIR.glob("2008-*.csv"))

    print("Total *2008*.csv files:", len(all_2008))
    print("Total 2008-*.csv files:", len(dash_2008))

    print("\nSample filenames (*2008*.csv):")
    for f in all_2008[:10]:
        print(" ", f.name)

    print("\nSample filenames (2008-*.csv):")
    for f in dash_2008[:10]:
        print(" ", f.name)


if __name__ == "__main__":
    main()
