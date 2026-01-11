/* ------------------------------------------------------------------------- */
/* FILE: 04_merge_final_dataset.do                                           */
/* DESC: Merge Newspaper Panel (03) with Economic CZ Data (02)                 */
/* ------------------------------------------------------------------------- */

clear all
set more off

* 1. Set Directory Paths
global root "C:/Users/ymw04/Dropbox/shifting_slant"
global econ     "$root/data/processed/econ"
global analysis "$root/data/analysis"

display ">>> [Step 1] Loading Newspaper Panel Data (Master)..."

* -------------------------------------------------------------------------
* A. Load Newspaper Data (from Step 03)
* -------------------------------------------------------------------------
use "$analysis/data_03_newspaper_panel_CZ.dta", clear

* Drop observations with missing CZ information
* (We cannot merge economic data if we don't know the location)
drop if czone == .


display ">>> [Step 2] Merging Economic Data (Using)..."

* -------------------------------------------------------------------------
* B. Merge with Economic Data (from Step 02)
* -------------------------------------------------------------------------
* Merge Type: m:1 (Many-to-One)
* Key Variables: czone, year
* Logic: Multiple newspapers can exist in one CZ, but Econ data is unique per CZ-Year.

merge m:1 czone year using "$econ/data_02_nafta_vars_CZ.dta"

* -------------------------------------------------------------------------
* C. Post-Merge Cleaning
* -------------------------------------------------------------------------
* Keep only observations that matched successfully (_merge == 3)
* (We need both text data AND economic data for the regression)
keep if _merge == 3 
drop _merge

display ">>> [Step 3] Finalizing and Saving..."

* -------------------------------------------------------------------------
* D. Save Final Dataset
* -------------------------------------------------------------------------
* Sort for panel data structure
sort fips year

* Save the final merged dataset for regression analysis
save "$analysis/data_04_final_dataset.dta", replace

display "=========================================================="
display ">>> [SUCCESS] Final dataset created!"
display ">>> Filename: data_04_final_dataset.dta"
display ">>> Location: $analysis/"
display ">>> Use this file for your regressions."
display "=========================================================="