"""
07b_manual_prep.py

[PURPOSE]
- Prepares a CSV file for MANUAL human verification.
- Disables AI/Gemini integration completely.
- Creates 'paper_name_crosswalk_fuzzy_manual.csv' for the user to edit in Excel.
"""

import os
import pandas as pd
from pathlib import Path

# 1. Path Setup
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
FUZZY_PATH = BASE_DIR / "data" / "meta" / "newspapers" / "paper_name_crosswalk_fuzzy.csv"
OUTPUT_PATH = BASE_DIR / "data" / "meta" / "newspapers" / "paper_name_crosswalk_fuzzy_manual.csv"

def main():
    if not FUZZY_PATH.exists():
        print(f"Error: {FUZZY_PATH} not found.")
        return

    # Load the fuzzy matches
    df = pd.read_csv(FUZZY_PATH)

    # Create columns for manual entry
    # Initialize 'valid' as empty so you can fill it (1 for Same, 0 for Different/Merger)
    df['valid'] = ""
    df['notes'] = "" # Space for your own comments if needed

    # Save to a new file for manual editing
    df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 30)
    print("READY FOR MANUAL REVIEW")
    print("-" * 30)
    print(f"File created: {OUTPUT_PATH}")
    print("\n[INSTRUCTIONS]")
    print("1. Open this file in Excel.")
    print("2. Look at 'original_name' vs 'clean_name'.")
    print("3. In the 'valid' column, type:")
    print("   - '1' if they are the SAME newspaper.")
    print("   - '0' if they are MERGERS or DIFFERENT.")
    print("4. Save and Close the file.")
    print("5. Let me know when you are done.")

if __name__ == "__main__":
    main()