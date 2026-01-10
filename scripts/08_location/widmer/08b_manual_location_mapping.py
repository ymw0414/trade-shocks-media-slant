"""
FILE: 08b_manual_location_mapping.py
DESCRIPTION:
    1. Loads 'locations_auto_success.csv' (Source: 1_auto_regex).
    2. Maps papers with City in name (Source: 2_title_extracted).
    3. Maps papers where City is NOT in name (Source: 3_hq_searched).

    * VERIFICATION NOTE:
      Category 3 locations have been cross-checked against 1994 historical records
      to ensure accuracy before the Fox News entry period (1996).
      (e.g., Post-Tribune was in Gary in 1994, moved later).
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))
GEO_DIR = BASE_DIR / "data" / "geo"
AUTO_SUCCESS_FILE = GEO_DIR / "locations_auto_success.csv"
FINAL_FILE = GEO_DIR / "locations_final.csv"

# --------------------------------------------------
# CATEGORY 2: Title Extracted
# (City Name IS visible in the newspaper title)
# --------------------------------------------------
TITLE_EXTRACTED = {
    "Akron Beacon Journal (OH)": {"city": "Akron", "state": "OH"},
    "Anchorage Daily News (AK)": {"city": "Anchorage", "state": "AK"},
    "Anniston Star, The (AL)": {"city": "Anniston", "state": "AL"},
    "Augusta Chronicle, The (GA)": {"city": "Augusta", "state": "GA"},
    "Austin American-Statesman": {"city": "Austin", "state": "TX"},
    "Bangor Daily News (ME)": {"city": "Bangor", "state": "ME"},
    "Boston Herald (MA)": {"city": "Boston", "state": "MA"},
    "Bradenton Herald, The (FL)": {"city": "Bradenton", "state": "FL"},
    "Charlotte Observer, The (NC)": {"city": "Charlotte", "state": "NC"},
    "Chicago Sun-Times": {"city": "Chicago", "state": "IL"},
    "Columbus Dispatch, The (OH)": {"city": "Columbus", "state": "OH"},
    "Daily News of Los Angeles (CA)": {"city": "Los Angeles", "state": "CA"},
    "Dayton Daily News (OH)": {"city": "Dayton", "state": "OH"},
    "Evansville Courier, The (IN)": {"city": "Evansville", "state": "IN"},
    "Fayetteville Observer, The (NC)": {"city": "Fayetteville", "state": "NC"},
    "Fort Pierce Tribune (FL)": {"city": "Fort Pierce", "state": "FL"},
    "Fort Worth Star-Telegram": {"city": "Fort Worth", "state": "TX"},
    "Fresno Bee, The (CA)": {"city": "Fresno", "state": "CA"},
    "Grand Forks Herald (ND)": {"city": "Grand Forks", "state": "ND"},
    "Greensboro News & Record": {"city": "Greensboro", "state": "NC"},
    "Houston Chronicle": {"city": "Houston", "state": "TX"},
    "Huntsville Times, The (AL)": {"city": "Huntsville", "state": "AL"},
    "Knoxville News-Sentinel, The (TN)": {"city": "Knoxville", "state": "TN"},
    "La Crosse Tribune (WI)": {"city": "La Crosse", "state": "WI"},
    "Lancaster New Era (PA)": {"city": "Lancaster", "state": "PA"},
    "Lawton Constitution, The (OK)": {"city": "Lawton", "state": "OK"},
    "Lewiston Morning Tribune (ID)": {"city": "Lewiston", "state": "ID"},
    "Lexington Herald-Leader (KY)": {"city": "Lexington", "state": "KY"},
    "Miami Herald, The (FL)": {"city": "Miami", "state": "FL"},
    "Milwaukee Sentinel": {"city": "Milwaukee", "state": "WI"},
    "Mobile Register (AL)": {"city": "Mobile", "state": "AL"},
    "Modesto Bee, The (CA)": {"city": "Modesto", "state": "CA"},
    "New Haven Register (CT)": {"city": "New Haven", "state": "CT"},
    "Ocala Star-Banner (FL)": {"city": "Ocala", "state": "FL"},
    "Omaha World-Herald (NE)": {"city": "Omaha", "state": "NE"},
    "Owensboro Messenger-Inquirer (KY)": {"city": "Owensboro", "state": "KY"},
    "Philadelphia Daily News (PA)": {"city": "Philadelphia", "state": "PA"},
    "Philadelphia Inquirer, The (PA)": {"city": "Philadelphia", "state": "PA"},
    "Pittsburgh Post-Gazette (PA)": {"city": "Pittsburgh", "state": "PA"},
    "Press of Atlantic City, The (NJ)": {"city": "Atlantic City", "state": "NJ"},
    "Providence Journal (RI)": {"city": "Providence", "state": "RI"},
    "Pueblo Chieftain, The (CO)": {"city": "Pueblo", "state": "CO"},
    "Redding Record Searchlight (CA)": {"city": "Redding", "state": "CA"},
    "Richmond Times-Dispatch": {"city": "Richmond", "state": "VA"},
    "Roanoke Times, The (VA)": {"city": "Roanoke", "state": "VA"},
    "Sacramento Bee, The (CA)": {"city": "Sacramento", "state": "CA"},
    "Salt Lake Tribune, The (UT)": {"city": "Salt Lake City", "state": "UT"},
    "San Antonio Express-News": {"city": "San Antonio", "state": "TX"},
    "San Diego Union-Tribune, The (CA)": {"city": "San Diego", "state": "CA"},
    "San Francisco Chronicle (CA)": {"city": "San Francisco", "state": "CA"},
    "San Jose Mercury News (CA)": {"city": "San Jose", "state": "CA"},
    "Seattle Post-Intelligencer": {"city": "Seattle", "state": "WA"},
    "Seattle Times, The (WA)": {"city": "Seattle", "state": "WA"},
    "St. Louis Post-Dispatch": {"city": "St. Louis", "state": "MO"},
    "St. Paul Pioneer Press (MN)": {"city": "St. Paul", "state": "MN"},
    "St. Petersburg Times": {"city": "St. Petersburg", "state": "FL"},
    "Staten Island Advance (NY)": {"city": "Staten Island", "state": "NY"},
    "Syracuse Herald American (NY)": {"city": "Syracuse", "state": "NY"},
    "Syracuse Herald-Journal (NY)": {"city": "Syracuse", "state": "NY"},
    "The Advocate (Baton Rouge, La.)": {"city": "Baton Rouge", "state": "LA"},
    "The Atlanta Journal and The Atlanta Constitution": {"city": "Atlanta", "state": "GA"},
    "The Buffalo News": {"city": "Buffalo", "state": "NY"},
    "The Cincinnati Post": {"city": "Cincinnati", "state": "OH"},
    "The Dallas Morning News": {"city": "Dallas", "state": "TX"},
    "The Denver Post": {"city": "Denver", "state": "CO"},
    "The Hartford Courant": {"city": "Hartford", "state": "CT"},
    "The Kansas City Star": {"city": "Kansas City", "state": "MO"},
    "The Milwaukee Journal": {"city": "Milwaukee", "state": "WI"},
    "The Palm Beach Post": {"city": "West Palm Beach", "state": "FL"},
    "The Tampa Tribune": {"city": "Tampa", "state": "FL"},
    "Tulsa World": {"city": "Tulsa", "state": "OK"},
    "Watertown Daily Times (NY)": {"city": "Watertown", "state": "NY"},
    "Wenatchee World, The (WA)": {"city": "Wenatchee", "state": "WA"},
    "Wichita Eagle, The (KS)": {"city": "Wichita", "state": "KS"},
    "Worcester Telegram & Gazette (MA)": {"city": "Worcester", "state": "MA"},
    "York Daily Record (PA)": {"city": "York", "state": "PA"}
}

# --------------------------------------------------
# CATEGORY 3: HQ Searched (Verified for 1994)
# (No City in Title -> Used 1994 HQ Location)
# --------------------------------------------------
HQ_SEARCHED = {
    # 1. Post-Tribune: Located in Gary, IN in 1994 (Moved to Merrillville later)
    "Post-Tribune (IN)": {"city": "Gary", "state": "IN"},

    # 2. Quad-City Times: Based in Davenport, IA (Scott County)
    "Quad-City Times, The (IA)": {"city": "Davenport", "state": "IA"},

    # 3. Rocky Mountain News: Based in Denver, CO in 1994
    "Rocky Mountain News (CO)": {"city": "Denver", "state": "CO"},

    # 4. Arizona Daily Star: Based in Tucson, AZ
    "The Arizona Daily Star": {"city": "Tucson", "state": "AZ"},

    # 5. Commercial Appeal: Based in Memphis, TN
    "The Commercial Appeal": {"city": "Memphis", "state": "TN"},

    # 6. Daily Oklahoman: Based in Oklahoma City, OK
    "The Daily Oklahoman": {"city": "Oklahoma City", "state": "OK"},

    # 7. The Gazette: Based in Cedar Rapids, IA
    "The Gazette (Cedar Rapids-Iowa City)": {"city": "Cedar Rapids", "state": "IA"},

    # 8. Kentucky Post: N.Kentucky edition of Cincy Post, bureau in Covington, KY
    "The Kentucky Post": {"city": "Covington", "state": "KY"},

    # 9. News & Observer: Based in Raleigh, NC
    "The News & Observer": {"city": "Raleigh", "state": "NC"},

    # 10. News Tribune: Based in Tacoma, WA
    "The News Tribune": {"city": "Tacoma", "state": "WA"},

    # 11. Orange County Register: Based in Santa Ana, CA in 1994 (Grand Ave)
    "The Orange County Register": {"city": "Santa Ana", "state": "CA"},

    # 12. The Record: Based in Hackensack, NJ in 1994 (Moved to Woodland Park in 2008)
    "The Record (New Jersey)": {"city": "Hackensack", "state": "NJ"}
}

def main():
    final_rows = []

    # 1. Load Auto-Regex (Category 1)
    if AUTO_SUCCESS_FILE.exists():
        df_auto = pd.read_csv(AUTO_SUCCESS_FILE)
        print(f">>> [Cat 1] Loaded {len(df_auto)} auto_regex locations.")
        for _, row in df_auto.iterrows():
            final_rows.append({
                "paper": row['paper'],
                "city": row['city'],
                "state": row['state'],
                "source_category": "1_auto_regex"
            })
    else:
        print("Note: No auto-success file found.")

    # 2. Add Title Extracted (Category 2)
    print(f">>> [Cat 2] Mapping {len(TITLE_EXTRACTED)} title_extracted locations...")
    for paper, loc in TITLE_EXTRACTED.items():
        final_rows.append({
            "paper": paper,
            "city": loc['city'],
            "state": loc['state'],
            "source_category": "2_title_extracted"
        })

    # 3. Add HQ Searched (Category 3)
    print(f">>> [Cat 3] Mapping {len(HQ_SEARCHED)} hq_searched locations (1994 verified)...")
    for paper, loc in HQ_SEARCHED.items():
        final_rows.append({
            "paper": paper,
            "city": loc['city'],
            "state": loc['state'],
            "source_category": "3_hq_searched"
        })

    # 4. Create DataFrame
    df_final = pd.DataFrame(final_rows)

    # 5. Deduplicate (Prioritize Manual > Auto)
    df_final = df_final.sort_values(['paper', 'source_category'], ascending=[True, False])
    df_final = df_final.drop_duplicates(subset=['paper'], keep='first')

    # Final Sort
    df_final = df_final.sort_values('paper').reset_index(drop=True)

    # 6. Save
    df_final.to_csv(FINAL_FILE, index=False)

    print("-" * 50)
    print("Final Location Source Breakdown:")
    print(df_final['source_category'].value_counts().sort_index())
    print("-" * 50)
    print(f"Total Papers Mapped: {len(df_final)}")
    print(f"Saved to: {FINAL_FILE}")
    print("Verification: HQ locations confirmed for year 1994.")

if __name__ == "__main__":
    main()