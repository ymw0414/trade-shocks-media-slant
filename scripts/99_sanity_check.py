# -------------------------------------------------------------------------
# FILE: 99_sanity_check.py
# DESC: Check column names and preview content of newspaper metadata files
#       to identify the correct text column for classification.
# -------------------------------------------------------------------------

import os
import argparse
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--congress", type=int, default=103, help="Congress number to check (default: 103)")
args = parser.parse_args()
CONGRESS = args.congress

# --------------------------------------------------
# Paths
# --------------------------------------------------

# Check for environment variable, fallback to current directory if not set
if "SHIFTING_SLANT_DIR" in os.environ:
    BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
else:
    # Assuming the script is run from the root or scripts folder if env is not set
    BASE_DIR = Path(os.getcwd()).parent
    print(f"(!) Warning: SHIFTING_SLANT_DIR not set. Using: {BASE_DIR}")

# Path to the metadata file (where the text usually resides)
META_NEWS = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams" / f"meta_newspapers_congress_{CONGRESS}.parquet"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"\n>>> [Sanity Check] Inspecting file for Congress {CONGRESS}")
    print(f"    File: {META_NEWS}")

    if not META_NEWS.exists():
        print(f"\n[ERROR] File not found: {META_NEWS}")
        print("Please check the congress number or file path.")
        return

    # Load Parquet file
    try:
        df = pd.read_parquet(META_NEWS)
    except Exception as e:
        print(f"\n[ERROR] Failed to read parquet file: {e}")
        return

    # 1. Print All Column Names
    print("\n" + "=" * 60)
    print(f"DATA COLUMNS (Total: {len(df.columns)})")
    print("=" * 60)
    print(df.columns.tolist())

    # 2. Preview String/Object Columns (To find the text body)
    print("\n" + "=" * 60)
    print("CONTENT PREVIEW (First 100 chars of string columns)")
    print("=" * 60)

    # Filter columns that are likely text (object type)
    string_cols = df.select_dtypes(include=['object', 'string']).columns

    found_text_candidate = False

    for col in string_cols:
        # Skip obviously non-text columns (like IDs) to keep output clean,
        # but you can remove this if needed.
        if "id" in col.lower() and "article" not in col.lower():
            continue

        try:
            first_val = df[col].iloc[0]
            preview = str(first_val)[:100].replace('\n', ' ')  # Remove newlines for clean display

            print(f"[*] Column: '{col}'")
            print(f"    Sample: {preview}...")
            print("-" * 40)

            found_text_candidate = True
        except:
            continue

    if not found_text_candidate:
        print("(!) No string columns found. Check if text is encoded or missing.")

    print("\n>>> Check complete. Use the correct column name in '06_project_widmer.py'.")


if __name__ == "__main__":
    main()