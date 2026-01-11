"""
FILE: 07d_normalize_slant_by_year.py
DESCRIPTION:
    - Loads the aggregated panel data (newspaper x year).
    - Performs YEAR-BY-YEAR standardization (Z-score) on 'slant_weighted'.
    - Logic: Within each year, calculate how many std devs a paper is from the mean.
    - This ensures strictly cross-sectional comparison per year.
    - Output: 'newspaper_panel_1986_2004_normalized.csv'
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths Configuration
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))
INPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004.csv"
OUTPUT_FILE = BASE_DIR / "data" / "analysis" / "newspaper_panel_1986_2004_normalized.csv"

def main():
    print("-" * 60)
    print("-" * 60)
    print(">>> [Step 07d] Normalizing Slant Scores (Year-by-Year)...")
    print("-" * 60)

    # 1. Load Data
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df):,} rows.")

    # 2. Basic Cleanup
    # Remove rows where slant is missing before calculation
    df = df.dropna(subset=['slant_weighted'])

    # 3. Apply Year-by-Year Normalization
    # Formula: Z_it = (Slant_it - Mean_t) / Std_t
    print("\n🔄 Applying Standardization within each Year...")

    df['slant_normalized'] = df.groupby('year')['slant_weighted'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # 4. Verify Results
    # Check that means are ~0 and std are ~1 for a few sample years
    print(f"\n📊 Verification (Per-Year Stats should be Mean~0, Std~1):")
    verify_years = [1990, 1996, 2000]
    for y in verify_years:
        if y in df['year'].values:
            subset = df[df['year'] == y]['slant_normalized']
            print(f"   [Year {y}] Mean: {subset.mean():.4f} | Std: {subset.std():.4f} | N: {len(subset)}")

    # 5. Save Output
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 60)
    print(f"✅ Normalization Complete.")
    print(f"💾 Saved to: {OUTPUT_FILE}")
    print("-" * 60)

if __name__ == "__main__":
    main()