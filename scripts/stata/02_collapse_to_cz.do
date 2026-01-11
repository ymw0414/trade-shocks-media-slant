/* ------------------------------------------------------------------------- */
/* FILE: 02_collapse_to_cz.do                                          */
/* DESC: Aggregating county-level NAFTA variables to Commuting Zone (CZ) level */
/* ------------------------------------------------------------------------- */

clear all
set more off

* 1. Set Directory Paths
global root "C:/Users/ymw04/Dropbox/shifting_slant"
global processed "$root/data/processed/econ"
global raw_cw "$root/data/raw/econ/crosswalk/cw_cty_czone"

display ">>> [Step 1] Preparing CZ Crosswalk..."

* -------------------------------------------------------------------------
* A. Prepare Crosswalk File
* -------------------------------------------------------------------------
use "$raw_cw/cw_cty_czone.dta", clear

* Standardize variable name (cty_fips -> county) to match your data
rename cty_fips county

* Remove duplicates to ensure unique mapping
duplicates drop county, force
tempfile cz_map
save `cz_map'

display ">>> [Step 2] Loading Main Data & Merging CZ Info..."

* -------------------------------------------------------------------------
* B. Load User's County Level Data
* -------------------------------------------------------------------------
use "$processed/data_01_build_nafta_vars.dta", clear

* Ensure county is numeric
destring county, replace

* Merge CZ information based on county
merge m:1 county using `cz_map', keep(match master) nogenerate

* Drop observations that do not have a mapped CZ (e.g., unmatched counties)
drop if czone == .

display ">>> [Step 3] Collapsing Data to CZ Level..."

* -------------------------------------------------------------------------
* C. Collapse to CZ Level
* -------------------------------------------------------------------------
* Logic:
* 1. Ratio variables (vulnerability, manushare) -> Population weighted mean
* (Larger counties contribute more to the CZ-level shock)
* 2. Level variables (emp, pop) -> Simple Sum

collapse ///
    (mean) vulnerability1990_scaled manushare1990 ///
    (sum)  pop1990_total emp ///
    [aw=pop1990_total], ///
    by(czone year)

* -------------------------------------------------------------------------
* D. Finalize and Save
* -------------------------------------------------------------------------

* Label variables
label var czone "Commuting Zone (1990 definition)"
label var vulnerability1990_scaled "NAFTA Vulnerability (CZ Level, Pop-Weighted)"
label var manushare1990 "Manufacturing Share (CZ Level, Pop-Weighted)"
label var pop1990_total "Total Population 1990 (CZ Level Sum)"
label var emp "Total Employment (CZ Level Sum)"

* Save the final dataset
save "$processed/data_02_nafta_vars_CZ.dta", replace

display "=========================================================="
display ">>> [SUCCESS] CZ-level dataset created successfully!"
display ">>> Filename: data_02_nafta_vars_CZ.dta"
display ">>> Location: $processed/"
display "=========================================================="