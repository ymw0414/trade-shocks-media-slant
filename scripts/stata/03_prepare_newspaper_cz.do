/* ------------------------------------------------------------------------- */
/* FILE: 03_prepare_newspaper_cz.do                                        */
/* DATE: 2026-01-11                                                          */
/* DESC: Import newspaper CSV, fix FIPS, and merge CZ info.                  */
/* (Year filtering is handled in previous Python step)                 */
/* */
/* MODIFICATIONS:                                                            */
/* - Added step to exclude Alaska (State FIPS = 02).                        */
/* - Included fix for Miami-Dade FIPS (12086 -> 12025) to match 1990 CW.    */
/* ------------------------------------------------------------------------- */

clear all
set more off

* 1. Set Directory Paths
global root "C:/Users/ymw04/Dropbox/shifting_slant"
global analysis "$root/data/analysis"
global raw_cw   "$root/data/raw/econ/crosswalk/cw_cty_czone"

display ">>> [Step 1] Preparing CZ Crosswalk in Memory..."

* -------------------------------------------------------------------------
* A. Load Crosswalk
* -------------------------------------------------------------------------
use "$raw_cw/cw_cty_czone.dta", clear

* Rename variable to match newspaper data (cty_fips -> fips)
rename cty_fips fips

* Ensure unique observation per FIPS
duplicates drop fips, force
tempfile cz_map
save `cz_map'

display ">>> [Step 2] Importing Newspaper CSV..."

* -------------------------------------------------------------------------
* B. Import Newspaper Panel Data
* -------------------------------------------------------------------------
* Note: Loading from data/analysis folder
import delimited "$analysis/newspaper_panel_with_geo.csv", clear 

* Convert fips to numeric if imported as string
capture destring fips, replace force

* Drop observations with missing FIPS
drop if fips == .

* -------------------------------------------------------------------------
* C. Data Cleaning & Adjustments
* -------------------------------------------------------------------------

* [1] Exclude Alaska
* Alaska State FIPS is 02. Range: 02000 - 02999
gen state_fips = int(fips / 1000)
drop if state_fips == 2
drop state_fips

* [2] Fix Historical FIPS Codes (to match 1990 definitions)
* Miami-Dade changed from 12025 to 12086 in 1997.
* The crosswalk uses the 1990 code (12025).
replace fips = 12025 if fips == 12086

display ">>> [Step 3] Merging CZ Information..."

* -------------------------------------------------------------------------
* D. Merge with CZ Crosswalk
* -------------------------------------------------------------------------
* Merge Type: m:1 (Many-to-One) because 'fips' repeats in the panel data
merge m:1 fips using `cz_map', keep(match master) nogenerate

* -------------------------------------------------------------------------
* E. Finalize and Save
* -------------------------------------------------------------------------
label var czone "Commuting Zone (1990 definition)"
label var fips "FIPS County Code"

* Save as Stata format (.dta) in the analysis folder
save "$analysis/data_03_newspaper_panel_CZ.dta", replace

display "=========================================================="
display ">>> [SUCCESS] Newspaper data linked with CZ info!"
display ">>> Note 1: Alaska (FIPS 02) excluded."
display ">>> Note 2: Miami-Dade (12086) remapped to 12025."
display ">>> Filename: data_03_newspaper_panel_CZ.dta"
display ">>> Location: $analysis/"
display "=========================================================="