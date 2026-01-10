"""
FILE: 08c_map_counties.py
DESCRIPTION:
    1. Loads 'locations_final.csv'.
    2. Maps Cities to FIPS Codes using the Golden Map.
    3. SIMPLIFIED CATEGORIES:
       - 'Exact_Match': 1-to-1 mapping (Includes Single Counties AND Independent Cities).
       - 'Primary_Multi': City spans multiple counties; selected the primary one.
    4. Outputs 'newspaper_county_map.csv'.
"""

import os
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))
GEO_DIR = BASE_DIR / "data" / "geo"
INPUT_FILE = GEO_DIR / "locations_final.csv"
OUTPUT_FILE = GEO_DIR / "newspaper_county_map.csv"

# --------------------------------------------------
# GOLDEN MAPPING DICTIONARY (Simplified Logic)
# Key: (City, State)
# Value: (FIPS, County Name, Match_Type)
# --------------------------------------------------

CITY_COUNTY_MAP = {
    # --- TYPE A: Primary Multi (We had to CHOOSE the main county) ---
    ("Atlanta", "GA"): ("13121", "Fulton County", "Primary_Multi"),
    ("Austin", "TX"): ("48453", "Travis County", "Primary_Multi"),
    ("Chicago", "IL"): ("17031", "Cook County", "Primary_Multi"),
    ("Columbus", "OH"): ("39049", "Franklin County", "Primary_Multi"),
    ("Dallas", "TX"): ("48113", "Dallas County", "Primary_Multi"),
    ("Fort Worth", "TX"): ("48439", "Tarrant County", "Primary_Multi"),
    ("Houston", "TX"): ("48201", "Harris County", "Primary_Multi"),
    ("Kansas City", "MO"): ("29095", "Jackson County", "Primary_Multi"),
    ("Milwaukee", "WI"): ("55079", "Milwaukee County", "Primary_Multi"),
    ("New York", "NY"): ("36061", "New York County", "Primary_Multi"),
    ("Oklahoma City", "OK"): ("40109", "Oklahoma County", "Primary_Multi"),
    ("Portland", "OR"): ("41051", "Multnomah County", "Primary_Multi"),
    ("San Antonio", "TX"): ("48029", "Bexar County", "Primary_Multi"),
    ("Tulsa", "OK"): ("40143", "Tulsa County", "Primary_Multi"),

    # --- TYPE B: Exact Match (1-to-1 Mapping, includes Independent Cities) ---
    # Independent Cities (Merged here as per user request)
    ("Baltimore", "MD"): ("24510", "Baltimore city", "Exact_Match"),
    ("Norfolk", "VA"): ("51710", "Norfolk city", "Exact_Match"),
    ("Richmond", "VA"): ("51760", "Richmond city", "Exact_Match"),
    ("Roanoke", "VA"): ("51770", "Roanoke city", "Exact_Match"),
    ("St. Louis", "MO"): ("29510", "St. Louis city", "Exact_Match"),

    # Standard Single Counties
    ("Akron", "OH"): ("39153", "Summit County", "Exact_Match"),
    ("Albany", "NY"): ("36001", "Albany County", "Exact_Match"),
    ("Allentown", "PA"): ("42077", "Lehigh County", "Exact_Match"),
    ("Anchorage", "AK"): ("02020", "Anchorage Municipality", "Exact_Match"),
    ("Annapolis", "MD"): ("24003", "Anne Arundel County", "Exact_Match"),
    ("Anniston", "AL"): ("01015", "Calhoun County", "Exact_Match"),
    ("Atlantic City", "NJ"): ("34001", "Atlantic County", "Exact_Match"),
    ("Augusta", "GA"): ("13245", "Richmond County", "Exact_Match"),
    ("Bangor", "ME"): ("23019", "Penobscot County", "Exact_Match"),
    ("Baton Rouge", "LA"): ("22033", "East Baton Rouge Parish", "Exact_Match"),
    ("Bloomington", "IL"): ("17113", "McLean County", "Exact_Match"),
    ("Bloomington", "IN"): ("18105", "Monroe County", "Exact_Match"),
    ("Boston", "MA"): ("25025", "Suffolk County", "Exact_Match"),
    ("Bradenton", "FL"): ("12081", "Manatee County", "Exact_Match"),
    ("Buffalo", "NY"): ("36029", "Erie County", "Exact_Match"),
    ("Cedar Rapids", "IA"): ("19113", "Linn County", "Exact_Match"),
    ("Charleston", "IL"): ("17029", "Coles County", "Exact_Match"),
    ("Charlotte", "NC"): ("37119", "Mecklenburg County", "Exact_Match"),
    ("Cincinnati", "OH"): ("39061", "Hamilton County", "Exact_Match"),
    ("Cleveland", "OH"): ("39035", "Cuyahoga County", "Exact_Match"),
    ("Colorado Springs", "CO"): ("08041", "El Paso County", "Exact_Match"),
    ("Columbia", "SC"): ("45079", "Richland County", "Exact_Match"),
    ("Covington", "KY"): ("21117", "Kenton County", "Exact_Match"),
    ("Davenport", "IA"): ("19163", "Scott County", "Exact_Match"),
    ("Dayton", "OH"): ("39113", "Montgomery County", "Exact_Match"),
    ("Decatur", "IL"): ("17115", "Macon County", "Exact_Match"),
    ("Denver", "CO"): ("08031", "Denver County", "Exact_Match"),
    ("Dover", "DE"): ("10001", "Kent County", "Exact_Match"),
    ("Evansville", "IN"): ("18163", "Vanderburgh County", "Exact_Match"),
    ("Fayetteville", "NC"): ("37051", "Cumberland County", "Exact_Match"),
    ("Flemington", "NJ"): ("34019", "Hunterdon County", "Exact_Match"),
    ("Fort Pierce", "FL"): ("12111", "St. Lucie County", "Exact_Match"),
    ("Fort Wayne", "IN"): ("18003", "Allen County", "Exact_Match"),
    ("Fresno", "CA"): ("06019", "Fresno County", "Exact_Match"),
    ("Gary", "IN"): ("18089", "Lake County", "Exact_Match"),
    ("Grand Forks", "ND"): ("38035", "Grand Forks County", "Exact_Match"),
    ("Greensboro", "NC"): ("37081", "Guilford County", "Exact_Match"),
    ("Hackensack", "NJ"): ("34003", "Bergen County", "Exact_Match"),
    ("Harrisburg", "PA"): ("42043", "Dauphin County", "Exact_Match"),
    ("Hartford", "CT"): ("09003", "Hartford County", "Exact_Match"),
    ("Huntsville", "AL"): ("01089", "Madison County", "Exact_Match"),
    ("Idaho Falls", "ID"): ("16019", "Bonneville County", "Exact_Match"),
    ("Knoxville", "TN"): ("47093", "Knox County", "Exact_Match"),
    ("La Crosse", "WI"): ("55063", "La Crosse County", "Exact_Match"),
    ("Lancaster", "PA"): ("42071", "Lancaster County", "Exact_Match"),
    ("Lawrence", "KS"): ("20045", "Douglas County", "Exact_Match"),
    ("Lawton", "OK"): ("40031", "Comanche County", "Exact_Match"),
    ("Lewiston", "ID"): ("16069", "Nez Perce County", "Exact_Match"),
    ("Lexington", "KY"): ("21067", "Fayette County", "Exact_Match"),
    ("Long Beach", "CA"): ("06037", "Los Angeles County", "Exact_Match"),
    ("Los Angeles", "CA"): ("06037", "Los Angeles County", "Exact_Match"),
    ("Madison", "WI"): ("55025", "Dane County", "Exact_Match"),
    ("Manchester", "NH"): ("33011", "Hillsborough County", "Exact_Match"),
    ("Mattoon", "IL"): ("17029", "Coles County", "Exact_Match"),
    ("Memphis", "TN"): ("47157", "Shelby County", "Exact_Match"),
    ("Miami", "FL"): ("12086", "Miami-Dade County", "Exact_Match"),
    ("Minneapolis", "MN"): ("27053", "Hennepin County", "Exact_Match"),
    ("Mobile", "AL"): ("01097", "Mobile County", "Exact_Match"),
    ("Modesto", "CA"): ("06099", "Stanislaus County", "Exact_Match"),
    ("Munster", "IN"): ("18089", "Lake County", "Exact_Match"),
    ("New Haven", "CT"): ("09009", "New Haven County", "Exact_Match"),
    ("New Orleans", "LA"): ("22071", "Orleans Parish", "Exact_Match"),
    ("Newark", "NJ"): ("34013", "Essex County", "Exact_Match"),
    ("Ocala", "FL"): ("12083", "Marion County", "Exact_Match"),
    ("Omaha", "NE"): ("31055", "Douglas County", "Exact_Match"),
    ("Owensboro", "KY"): ("21059", "Daviess County", "Exact_Match"),
    ("Peoria", "IL"): ("17143", "Peoria County", "Exact_Match"),
    ("Philadelphia", "PA"): ("42101", "Philadelphia County", "Exact_Match"),
    ("Pittsburgh", "PA"): ("42003", "Allegheny County", "Exact_Match"),
    ("Providence", "RI"): ("44007", "Providence County", "Exact_Match"),
    ("Pueblo", "CO"): ("08101", "Pueblo County", "Exact_Match"),
    ("Racine", "WI"): ("55101", "Racine County", "Exact_Match"),
    ("Raleigh", "NC"): ("37183", "Wake County", "Exact_Match"),
    ("Redding", "CA"): ("06089", "Shasta County", "Exact_Match"),
    ("Riverside", "CA"): ("06065", "Riverside County", "Exact_Match"),
    ("Sacramento", "CA"): ("06067", "Sacramento County", "Exact_Match"),
    ("Salt Lake City", "UT"): ("49035", "Salt Lake County", "Exact_Match"),
    ("San Diego", "CA"): ("06073", "San Diego County", "Exact_Match"),
    ("San Francisco", "CA"): ("06075", "San Francisco County", "Exact_Match"),
    ("San Jose", "CA"): ("06085", "Santa Clara County", "Exact_Match"),
    ("Santa Ana", "CA"): ("06059", "Orange County", "Exact_Match"),
    ("Seattle", "WA"): ("53033", "King County", "Exact_Match"),
    ("Spartanburg", "SC"): ("45083", "Spartanburg County", "Exact_Match"),
    ("Springfield", "IL"): ("17167", "Sangamon County", "Exact_Match"),
    ("Springfield", "MA"): ("25013", "Hampden County", "Exact_Match"),
    ("St. Paul", "MN"): ("27123", "Ramsey County", "Exact_Match"),
    ("St. Petersburg", "FL"): ("12103", "Pinellas County", "Exact_Match"),
    ("Staten Island", "NY"): ("36085", "Richmond County", "Exact_Match"),
    ("Syracuse", "NY"): ("36067", "Onondaga County", "Exact_Match"),
    ("Tacoma", "WA"): ("53053", "Pierce County", "Exact_Match"),
    ("Tampa", "FL"): ("12057", "Hillsborough County", "Exact_Match"),
    ("Torrance", "CA"): ("06037", "Los Angeles County", "Exact_Match"),
    ("Tucson", "AZ"): ("04019", "Pima County", "Exact_Match"),
    ("Watertown", "NY"): ("36045", "Jefferson County", "Exact_Match"),
    ("Wenatchee", "WA"): ("53007", "Chelan County", "Exact_Match"),
    ("West Palm Beach", "FL"): ("12099", "Palm Beach County", "Exact_Match"),
    ("Wichita", "KS"): ("20173", "Sedgwick County", "Exact_Match"),
    ("Wilkes Barre", "PA"): ("42079", "Luzerne County", "Exact_Match"),
    ("Worcester", "MA"): ("25027", "Worcester County", "Exact_Match"),
    ("York", "PA"): ("42133", "York County", "Exact_Match"),
}

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run 08b first.")
        return

    print(f">>> [08c] Mapping Cities to FIPS with Simplified Categories...")

    df = pd.read_csv(INPUT_FILE)

    fips_list = []
    county_name_list = []
    match_type_list = []

    for _, row in df.iterrows():
        city = row['city']
        state = row['state']
        key = (city, state)

        if key in CITY_COUNTY_MAP:
            fips, c_name, m_type = CITY_COUNTY_MAP[key]
            fips_list.append(fips)
            county_name_list.append(c_name)
            match_type_list.append(m_type)
        else:
            print(f"WARNING: No mapping found for {city}, {state}")
            fips_list.append(None)
            county_name_list.append(None)
            match_type_list.append("MISSING")

    # Add columns
    df['fips'] = fips_list
    df['county_name'] = county_name_list
    df['match_type'] = match_type_list

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 50)
    print("Match Type Statistics (Simplified):")
    print(df['match_type'].value_counts())
    print("-" * 50)
    print(f"Total Papers Mapped: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()