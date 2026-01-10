"""
FILE: 08a_extract_locations.py
DESCRIPTION:
    1. Loads unique newspaper names from the finalized panel (07c output).
    2. Uses Regex to extract City and State:
       - Format "Name (City, ST)" -> Auto-Success.
       - Format "Name (ST)" -> Partial extraction (City missing) -> Goes to Failed list for review.
       - No brackets -> Failed list for manual entry.
    3. Outputs two files: 'locations_auto_success.csv' and 'locations_auto_failed.csv'.
"""

import os
import pandas as pd
import re
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
if "SHIFTING_SLANT_DIR" not in os.environ:
    BASE_DIR = Path(r"C:\Users\ymw04\Dropbox\shifting_slant")
else:
    BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# UPDATED: Input file from 07c (Aggregation Step)
INPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004.csv"
OUT_DIR = BASE_DIR / "data" / "geo"

# Create output directory if not exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE_SUCCESS = OUT_DIR / "locations_auto_success.csv"
OUT_FILE_FAILED = OUT_DIR / "locations_auto_failed.csv"

# --------------------------------------------------
# Extraction Logic
# --------------------------------------------------

def extract_auto(paper_name):
    """
    Attempts to parse City and State from the string.

    Returns:
        (city, state) if fully successful (City, ST format)
        (None, state) if partial match (ST format only)
        (None, None) if no location info found
    """
    # 1. Pattern for Full Location: "(City, ST)"
    # Matches: "Star Tribune (Minneapolis, MN)"
    match_full = re.search(r'\(([^)]+),\s*([A-Z]{2})\)', paper_name)
    if match_full:
        return match_full.group(1).strip(), match_full.group(2).strip()

    # 2. Pattern for State Only: "(ST)"
    # Matches: "Akron Beacon Journal (OH)"
    match_state = re.search(r'\(([A-Z]{2})\)', paper_name)
    if match_state:
        # We found state, but City is unknown (or implicit in title)
        return None, match_state.group(1).strip()

    # 3. No brackets detected (e.g., "Richmond Times-Dispatch")
    return None, None

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        print("Please run Step 07c (Aggregation) first.")
        return

    # Load unique papers from the panel
    print(f">>> [08a] Loading unique papers from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    unique_papers = sorted(df['paper'].unique())

    success_list = []
    failed_list = []

    print(f">>> Analyzing {len(unique_papers)} newspaper titles...")

    for paper in unique_papers:
        city, state = extract_auto(paper)

        if city and state:
            # Fully successful extraction
            success_list.append({"paper": paper, "city": city, "state": state})
        else:
            # Partial or Failed extraction -> Needs manual review
            # If state was detected (e.g. from "(OH)"), we save it to help manual process
            failed_list.append({"paper": paper, "city": "", "detected_state": state})

    # Save outputs
    pd.DataFrame(success_list).to_csv(OUT_FILE_SUCCESS, index=False)
    pd.DataFrame(failed_list).to_csv(OUT_FILE_FAILED, index=False)

    print("-" * 50)
    print(f"Total Unique Papers: {len(unique_papers)}")
    print(f"  - Auto-Success (City & State found): {len(success_list)}")
    print(f"  - Failed/Partial (Needs manual fill): {len(failed_list)}")
    print("-" * 50)
    print(f"Saved Success list to: {OUT_FILE_SUCCESS}")
    print(f"Saved Failed list to : {OUT_FILE_FAILED}")
    print("-" * 50)
    print("NEXT STEP: Open 'locations_auto_failed.csv' and manually fill the empty 'city' columns.")
    print("If 'detected_state' is correct, move it to a 'state' column.")

if __name__ == "__main__":
    main()