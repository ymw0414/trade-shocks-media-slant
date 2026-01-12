/* ------------------------------------------------------------------------- */
/* FILE: 05_event_study_slant.do                                           */
/* DATE: 2026-01-11                                                          */
/* DESC: Event Study Analysis of NAFTA Impact on Newspaper Slant.            */
/* MODIFIED: Removed 'state#year' FE due to small sample size (N < 130).     */
/* ------------------------------------------------------------------------- */

clear all
set more off
cls

* 1. Set Directory Paths
global root "C:/Users/ymw04/Dropbox/shifting_slant"
global analysis "$root/data/analysis"
global results  "$root/results"

capture mkdir "$results"

/* ========================================================================= */
/* SECTION 0: Define Event Study Program                                     */
/* ========================================================================= */

cap program drop my_eventstudy2
program define my_eventstudy2
    syntax varlist(min=2 max=2), BASEYEAR(integer) position(integer)

    local depvar : word 1 of `varlist'
    local vulvar : word 2 of `varlist'
    local baseyear = `baseyear'
    local position = `position'

    display ">>> [Analysis] Outcome: `depvar' | Shock: `vulvar' | Base Year: `baseyear'"

    // --- Step 1: Generate interaction variables (Shock x Year) ---
    quietly levelsof year, local(yrs)
    foreach y of local yrs {
        if `y' != `baseyear' {
            cap drop _es_`y'
            gen _es_`y' = `vulvar' * (year == `y')
        }
    }

    local intvars
    foreach y of local yrs {
        if `y' != `baseyear' {
            local intvars `intvars' _es_`y'
        }
    }
    
    // --- Step 1b: Generate Control Interactions (Manuf Share x Year) ---
    // This controls for secular trends in manufacturing areas regardless of NAFTA exposure.
    local manulist
    capture confirm variable manushare1990
    if _rc == 0 {
        foreach y of local yrs {
            cap drop manu1990year`y'
            gen manu1990year`y' = manushare1990 * (year == `y')
            local manulist `manulist' manu1990year`y'
        }
        local label_model2 "With Manuf. Share x Year"
    }
    else {
        local label_model2 "Model 2 (No Manuf. Control)"
    }

    // --- Step 2: First regression (Baseline) ---
    // REMOVED state#year FE. Kept fips (Unit FE) and year (Time FE).
    display ">>> Estimating Model 1 (Baseline: Unit FE + Year FE)..."
    reghdfe `depvar' `intvars' [pw = pop1990_total], ///
        absorb(news_id year) vce(robust)

    tempfile est1
    parmest, saving(`est1', replace) format(parm estimate min95 max95)

    // --- Step 3: Second regression (With Controls) ---
    display ">>> Estimating Model 2 (With Controls)..."
    reghdfe `depvar' `intvars' `manulist' [pw = pop1990_total], ///
        absorb(news_id year) vce(robust)
    
    tempfile est2
    parmest, saving(`est2', replace) format(parm estimate min95 max95)

    // --- Step 4: Load and prep results for plotting ---
    use `est1', clear
    keep if strpos(parm, "_es_") > 0
    gen year = real(substr(parm, 5, .))
    gen spec = "Baseline"
    rename estimate est1
    rename min95 min1
    rename max95 max1

    tempfile base
    save `base'

    use `est2', clear
    keep if strpos(parm, "_es_") > 0
    gen year = real(substr(parm, 5, .))
    
    // Shift the x-axis slightly for visual clarity
    replace year = year + 0.2
    gen spec = "WithFE"
    rename estimate est2
    rename min95 min2
    rename max95 max2

    merge 1:1 year using `base', nogen

    // --- Step 5: Add base year point (0) for both specs ---
    set obs `=_N + 1'
    replace year = `baseyear' in L
    replace est1 = 0 in L
    replace min1 = 0 in L
    replace max1 = 0 in L
    replace spec = "Baseline" in L

    set obs `=_N + 1'
    replace year = `baseyear' + 0.2 in L
    replace est2 = 0 in L
    replace min2 = 0 in L
    replace max2 = 0 in L
    replace spec = "WithFE" in L
    
    sort year

    // --- Step 6: Plot both ---
    sum year
    local xmin = r(min)
    local xmax = r(max)
    
    twoway ///
        (rcap min1 max1 year, lcolor(midblue%60)) ///
        (scatter est1 year, msymbol(square) mcolor(midblue) mfcolor(white)) ///
        (rcap min2 max2 year, lcolor(orange%60)) ///
        (scatter est2 year, msymbol(diamond) mcolor(orange) mfcolor(white)), ///
        yline(0, lpattern(dot) lcolor(black)) ///
        xline(`baseyear', lpattern(dash) lcolor(gs10)) ///
        ytitle("Effect on Newspaper Slant (Normalized)") ///
        xtitle("Year") ///
        title("Event Study: NAFTA Vulnerability on Newspaper Slant") ///
        note("Method: Weighted FE (Weights: 1990 Pop). FE: Newspaper & Year. Cluster: State.") ///
        xlabel(`xmin'(2)`xmax', labsize(small) angle(45)) ///
        ylabel(, angle(horizontal) nogrid format(%04.2f)) ///
        legend(order(2 "Baseline (Year FE)" 4 "`label_model2'") ///
               size(small) rows(2) ring(0) position(`position')) ///
        graphregion(color(white)) ///
        plotregion(style(none))
end

/* ========================================================================= */
/* SECTION 1: Execute Analysis                                               */
/* ========================================================================= */

display ">>> [Step 1] Loading Final Dataset..."
use "$analysis/data_04_final_dataset.dta", clear

egen news_id = group(paper)

* [Check] Ensure manushare1990 exists. If not, generate dummy for code safety.
capture confirm variable manushare1990
if _rc != 0 {
    display ">>> WARNING: 'manushare1990' not found. Creating placeholder (0)."
    gen manushare1990 = 0 
}

display ">>> [Step 2] Running Event Study Program..."

* Run the program
* Outcome: slant_normalized
* Shock:   vulnerability1990_scaled
* Base Year: 1993
my_eventstudy2 slant_normalized vulnerability1990_scaled, baseyear(1993) position(11)

display ">>> [Step 3] Saving Graph..."
graph export "$results/figure_event_study_slant_noStateYearFE.pdf", replace

display "=========================================================="
display ">>> [SUCCESS] Event Study Completed (No State-Year FE)."
display ">>> Graph saved to: $results/figure_event_study_slant_noStateYearFE.pdf"
display "=========================================================="