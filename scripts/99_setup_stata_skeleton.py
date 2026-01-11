"""
FILE: 99_setup_stata_skeleton.py
DESCRIPTION:
    1. Creates the directory structure including a dedicated 'logs' folder.
    2. Generates 'Skeleton' .do files with English comments/instructions.
"""

import os
from pathlib import Path

# --------------------------------------------------
# 1. Path Configuration
# --------------------------------------------------
BASE_DIR = Path(os.environ.get("SHIFTING_SLANT_DIR", r"C:\Users\ymw04\Dropbox\shifting_slant"))

# Directories to create
NEW_DIRS = [
    # Data folders
    BASE_DIR / "data" / "processed" / "econ",  # Stata outputs
    BASE_DIR / "data" / "raw" / "econ",        # Raw inputs
    BASE_DIR / "data" / "temp",

    # Results folders
    BASE_DIR / "results" / "tables",
    BASE_DIR / "results" / "figures",
    BASE_DIR / "results" / "logs",             # [NEW] Dedicated logs folder

    # Scripts
    BASE_DIR / "scripts" / "stata",
]

# --------------------------------------------------
# 2. Do-file Templates
# --------------------------------------------------

# [00_master.do] - Adds log configuration
CONTENT_MASTER = r"""/* -------------------------------------------------------------------------- */
/* FILE: 00_master.do                                                         */
/* DESC: Master script to setup globals and run the analysis pipeline         */
/* -------------------------------------------------------------------------- */

clear all
set more off
macro drop _all
capture log close

* --- 1. Directory Setup ---
if c(os) == "MacOSX" {
    global root "/Users/ymw0414/Library/CloudStorage/Dropbox/shifting_slant"
}
else {
    global root "C:/Users/ymw04/Dropbox/shifting_slant"
}

* --- 2. Define Global Paths ---

* [Data Paths]
global raw_data     "$root/data/raw/econ"
global py_data      "$root/data/analysis"
global processed    "$root/data/processed/econ"
global working_data "$processed"               // Alias for compatibility
global temp_data    "$root/data/temp"

* [Results Paths]
global results      "$root/results"
global logs         "$root/results/logs"       // [NEW] Logs location
global scripts      "$root/scripts/stata"

* --- 3. Start Logging ---
* Saves a log file named 'master_log.txt' in the logs folder
log using "$logs/master_log.txt", replace text

* --- 4. Execute Pipeline ---

display ">>> [00_master] Setup Complete."
display ">>> Logging to: $logs/master_log.txt"

* Step 1: Construct Variables
* (Run once, then comment out)
* do "$scripts/01_build_nafta_vars.do"

* Step 2: Merge Data
* do "$scripts/03_prepare_newspaper_cz.do"

* Step 3: Analysis
* do "$scripts/03_analysis_eventstudy.do"

* End Logging
log close
"""

# [01_build_nafta_vars.do]
CONTENT_BUILD = r"""/* -------------------------------------------------------------------------- */
/* FILE: 01_build_nafta_vars.do                                               */
/* DESC: Constructs NAFTA exposure and economic variables.                    */
/* (Replication of Choi et al. Sections 1-5)                            */
/* -------------------------------------------------------------------------- */

display ">>> [Step 01] Building Variables..."

* =========================================================================== *
* INSTRUCTION:                                                                *
* 1. Copy 'Section 1' to 'Section 5' from your original code.                 *
* 2. Paste it below.                                                          *
* 3. NOTE: Your code uses $working_data. We aliased this to 'processed/econ'  *
* in 00_master.do, so NO NEED to change your code!                         *
* =========================================================================== *

* [Paste your code here]

"""

# [03_prepare_newspaper_cz.do]
CONTENT_MERGE = r"""/* -------------------------------------------------------------------------- */
/* FILE: 03_prepare_newspaper_cz.do                                                   */
/* DESC: Merges Python Slant Panel with Stata Economic Variables              */
/* -------------------------------------------------------------------------- */

display ">>> [Step 02] Merging Data..."

* 1. Load Python Data (Newspaper Panel)
import delimited "$py_data/newspaper_panel_with_geo.csv", clear

* Format FIPS code for merging (ensure 5-digit string)
tostring fips, replace
replace fips = "0" + fips if length(fips) == 4
rename fips county_fips
destring year, replace

tempfile text_panel
save `text_panel'

* 2. Load Econ Data (From Step 01)
capture confirm file "$processed/data_figure2.dta"

if _rc == 0 {
    use "$processed/data_figure2.dta", clear
    
    * Create Merge Key
    tostring county, generate(county_fips)
    replace county_fips = "0" + county_fips if length(county_fips) == 4

    * 3. Merge (m:1 text is paper-year, econ is county-year)
    use `text_panel', clear
    merge m:1 county_fips year using "$processed/data_figure2.dta"
    
    * Keep matched
    keep if _merge == 3
    drop _merge

    * 4. Save Final Dataset
    save "$processed/final_analysis_set.dta", replace
    display ">>> Final dataset saved: $processed/final_analysis_set.dta"
}
else {
    display as error "ERROR: Economic data file not found. Run Step 01 first."
}
"""

# [03_analysis_eventstudy.do]
CONTENT_ANALYSIS = r"""/* -------------------------------------------------------------------------- */
/* FILE: 03_analysis_eventstudy.do                                            */
/* DESC: Main Analysis - Event Studies and Regressions                        */
/* -------------------------------------------------------------------------- */

display ">>> [Step 03] Running Analysis..."

* =========================================================================== *
* INSTRUCTION:                                                                *
* 1. Copy 'Section 0' (Program Definitions: my_eventstudy) from your code.    *
* 2. Paste it below.                                                          *
* 3. Use "$processed/final_analysis_set.dta" for regressions.                 *
* =========================================================================== *

* use "$processed/final_analysis_set.dta", clear

* [Paste your Program Definitions here]

* [Write your Regression Commands here]

"""

DO_FILES = {
    "00_master.do": CONTENT_MASTER,
    "01_build_nafta_vars.do": CONTENT_BUILD,
    "03_prepare_newspaper_cz.do": CONTENT_MERGE,
    "03_analysis_eventstudy.do": CONTENT_ANALYSIS
}

# --------------------------------------------------
# 3. Execution Logic
# --------------------------------------------------
def main():
    print(f">>> Initializing Stata Environment in: {BASE_DIR}")

    if not BASE_DIR.exists():
        print(f"Error: Base directory {BASE_DIR} not found.")
        return

    # Create Folders
    for folder in NEW_DIRS:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            print(f"   [NEW] Directory created: {folder}")
        else:
            print(f"   [OK]  Directory exists:  {folder}")

    # Create Files
    script_dir = BASE_DIR / "scripts" / "stata"
    for filename, content in DO_FILES.items():
        filepath = script_dir / filename
        # Overwrite Master file to update log paths
        if filename == "00_master.do" or not filepath.exists():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"   [UPDATED] {filename}")
        else:
            print(f"   [SKIP] {filename} exists (preserved)")

    print("\n" + "="*50)
    print("✅ Stata environment updated with Logs folder.")
    print("1. 'results/logs' folder created.")
    print("2. '00_master.do' updated to save logs there.")
    print("="*50)

if __name__ == "__main__":
    main()