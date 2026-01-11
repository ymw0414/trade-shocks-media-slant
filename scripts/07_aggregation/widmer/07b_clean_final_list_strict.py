"""
FILE: 07b_clean_final_list_strict.py
DESCRIPTION:
    - Loads the raw list from 07a.
    - Strictly filters out national/specialized media based on user's constraint (No circulation data).
    - Standardizes ONLY evident naming variants (e.g., "Paper (VA)" -> "Paper").
    - Keeps ambiguous cases (Morning vs Sunday) SEPARATE.
    - Result: A clean list of local newspapers defined strictly by their publication name.
"""

import pandas as pd
from pathlib import Path
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))
INPUT_FILE = BASE_DIR / "data" / "temp" / "candidate_papers_1992_1995.csv"
OUTPUT_FILE = BASE_DIR / "data" / "temp" / "final_target_papers.csv"

# --------------------------------------------------
# 1. Exclusion List (National, Student, Magazines)
# --------------------------------------------------
# Removing outlets that are clearly NOT local daily newspapers.
DROP_LIST = [
    # National / Wire Services
    "USA TODAY", "UPI (USA)", "CHRISTIAN SCIENCE MONITOR",
    "All Things Considered [NPR] (USA)", "Weekend Edition Saturday [NPR] (USA)",
    "Weekend Edition Sunday [NPR] (USA)", "Scripps Howard News Service",

    # Magazines / Specialized / Student
    "Newsweek", "New Scientist", "Science News", "Governing Magazine (USA)",
    "Reason (USA)", "Catalyst Chicago (IL)", "Chicago Reporter, The (IL)",
    "Daily Pennsylvanian, The: University of Pennsylvania (Philadelphia, PA)",
    "Miami New Times (FL)" # Alternative weekly, distinctly different from dailies
]

# --------------------------------------------------
# 2. Minimal Variant Mapping (Strict Rule)
# --------------------------------------------------
# Only merge if it is undeniable they are the exact same string with a suffix difference.
# We DO NOT merge "Herald-Journal" and "Herald American" because without circulation data,
# we treat different names as different units.
NAME_FIX_MAP = {
    "Richmond Times-Dispatch (VA)": "Richmond Times-Dispatch",
    "St. Louis Post-Dispatch (MO)": "St. Louis Post-Dispatch",
    "SACRAMENTO BEE": "Sacramento Bee, The (CA)",
    "THE SAN FRANCISCO CHRONICLE": "San Francisco Chronicle (CA)",
    "THE SEATTLE TIMES": "Seattle Times, The (WA)",
    "Star Tribune: Newspaper of the Twin Cities": "Star Tribune (Minneapolis, MN)"
}

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run 07a first.")
        return

    # Load raw list
    df = pd.read_csv(INPUT_FILE)

    # 1. Standardize Variants (Fixing typos/suffixes only)
    df['paper_clean'] = df['paper'].replace(NAME_FIX_MAP).str.strip()

    # 2. Filter National/Specialized Noise
    df = df[~df['paper_clean'].isin(DROP_LIST)]

    # --------------------------------------------------
    # 2.5 Exclude Alaska (CONUS only)
    # --------------------------------------------------
    df = df[~df['paper_clean'].str.contains(r'\(AK\)|Alaska', case=False, na=False)]

    # Filter by keywords to catch any remaining non-local/academic noise
    df = df[~df['paper_clean'].str.contains('University|College|Magazine|News Service', case=False, na=False)]

    # 3. Create Final List (Keeping distinct names distinct)
    final_list = sorted(df['paper_clean'].unique())
    final_df = pd.DataFrame({'paper': final_list})

    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"Total Local Newspaper Titles Selected: {len(final_df)}")
    print(f"Note: Distinct names (e.g., Morning vs Sunday) are preserved as separate units.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 50)

if __name__ == "__main__":
    main()