"""
12_build_nafta_vars.py

Build NAFTA exposure variables at the county and commuting-zone level.
Translates scripts/stata/01_build_nafta_vars.do + 02_collapse_to_cz.do
into Python.

Pipeline:
  Section 1: Ad valorem equivalent (AVE) tariff from Romalis + USITC data
  Section 2: Import-weighted tariff and revealed comparative advantage (RCA)
  Section 3: County Business Patterns employment (1986-2008)
  Section 4: NAFTA vulnerability measure (Hakobyan & McLaren)
  Section 5: Controls (population, education, manufacturing share, income)
  Section 6: Collapse county panel to commuting zones (pop-weighted)

Inputs:
  - data/raw/econ/tariff/tariff_89-01_romalis/USHTS{yy}.TXT
  - data/raw/econ/tariff/tariff_02-17_usitc/tariff_database_{yyyy}.txt
  - data/raw/econ/import/import_usitc/dataweb-queryExport.xlsx
  - data/raw/econ/import/import_hakobyan_mclaren/imports1990-2000.dta
  - data/raw/econ/export/export_hakobyan_mclaren/worldex19902000.csv
  - data/raw/econ/etc/crosswalk_david_dorn/cw_hs6_sic87dd.dta
  - data/raw/econ/cbp/cbp{yy}co.txt
  - data/raw/econ/demographic/county_by_sex_race_age_8089.xls
  - data/raw/econ/demographic/stch{yyyy}.dta
  - data/raw/econ/demographic/coest00intalldata.dta
  - data/raw/econ/education/Education.xls
  - data/raw/econ/income/median_hh_income.csv
  - data/raw/econ/crosswalk/countyfip_name_crosswalk.dta
  - data/raw/econ/crosswalk/cw_cty_czone/cw_cty_czone.dta

Outputs:
  - data/processed/econ/minwoo/12_nafta_vars_county.parquet
  - data/processed/econ/minwoo/12_nafta_vars_cz.parquet
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

RAW = BASE_DIR / "data" / "raw" / "econ"
PROCESSED = BASE_DIR / "data" / "processed" / "econ"
OUT_DIR = PROCESSED / "minwoo"

TARIFF_ROMALIS_DIR = RAW / "tariff" / "tariff_89-01_romalis"
TARIFF_USITC_DIR = RAW / "tariff" / "tariff_02-17_usitc"
IMPORT_USITC_EXCEL = RAW / "import" / "import_usitc" / "dataweb-queryExport.xlsx"
IMPORT_HM_PATH = RAW / "import" / "import_hakobyan_mclaren" / "imports1990-2000.dta"
EXPORT_HM_PATH = RAW / "export" / "export_hakobyan_mclaren" / "worldex19902000.csv"
HS6_SIC_PATH = RAW / "etc" / "crosswalk_david_dorn" / "cw_hs6_sic87dd.dta"
CBP_DIR = RAW / "cbp"
DEMO_80S_PATH = RAW / "demographic" / "county_by_sex_race_age_8089.xls"
DEMO_00S_PATH = RAW / "demographic" / "coest00intalldata.dta"
EDUCATION_PATH = RAW / "education" / "Education.xls"
INCOME_PATH = RAW / "income" / "median_hh_income.csv"
FIPS_XW_PATH = RAW / "crosswalk" / "countyfip_name_crosswalk.dta"
CZ_XW_PATH = RAW / "crosswalk" / "cw_cty_czone" / "cw_cty_czone.dta"

# Stata intermediate for exact replication (optional)
STATA_TARIFF_RCA_PATH = PROCESSED / "tariff_rca_MEX.dta"

# Employment flag midpoints (CBP suppressed employment ranges)
EMP_FLAG_MAP = {
    "A": 10, "B": 60, "C": 175, "E": 375, "F": 750,
    "G": 1750, "H": 3750, "I": 7500, "J": 17500,
    "K": 37500, "L": 75000,
}

# Unit quantity description -> abbreviation (for USITC unit value merge)
UNIT_ABBREV = {
    "kilograms": "KG", "liters": "L", "cubic meters": "X",
    "metric tons": "T", "square meters": "M2", "pieces": "PCS",
    "no units collected": "NA", "proof liters": "PFL", "number": "NO",
    "thousand units": "THS", "thousands of cubic meters": "M3",
    "thousand meters": "KM3", "barrels": "BBL", "megawatt hours": "MWH",
    "gold content grams": "GM", "grams": "G", "dozens": "DOZ",
    "ton raw value": "THM", "component grams": "GCN", "meters": "M",
    "dozen pairs": "DPR", "gross": "GRS", "clean yield kilograms": "CYK",
    "pairs": "PRS", "squares": "SQ", "dozen pieces": "DPC",
    "component tons": "CTN", "square centimeters": "CM2",
    "component kilograms": "GR", "hundred units": "HUN", "carats": "CAR",
    "fiber meters": "SME", "pack": "PK", "linear meters": "LNM",
    "Megabecquerels": "FBM", "doses": "DS", "tons": "KTS",
}


# ==================================================================
# Section 1: Ad Valorem Equivalent Tariff
# ==================================================================
def build_tariffs():
    """Build combined tariff series 1990-2008 with ad valorem equivalents."""
    print("  Section 1: Ad Valorem Equivalent Tariff")

    # --- Romalis tariffs (1990-2001) ---
    romalis_years = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1]
    romalis_dfs = []
    for y in romalis_years:
        yy = f"{y:02d}"
        path = TARIFF_ROMALIS_DIR / f"USHTS{yy}.TXT"
        df = pd.read_csv(path, sep="\t", encoding="latin1", low_memory=False)
        df["year"] = df["year"].fillna(y)  # Stata: replace year = `y' if missing(year)
        keep = ["hts8", "year", "brief_description", "quantity_1_code",
                "mexico_ad_val_rate", "mexico_specific_rate", "unitvalue"]
        df = df[[c for c in keep if c in df.columns]]
        romalis_dfs.append(df)

    romalis = pd.concat(romalis_dfs, ignore_index=True)
    romalis["year"] = romalis["year"].astype(float).astype(int)
    romalis.loc[romalis["year"] < 100, "year"] += 1900
    print(f"    Romalis: {len(romalis):,} rows (1990-2001)")

    # --- USITC tariffs (2002-2008) ---
    usitc_dfs = []
    for y in range(2002, 2009):
        path = TARIFF_USITC_DIR / f"tariff_database_{y}.txt"
        df = pd.read_csv(path, sep="|", encoding="latin1", low_memory=False)
        df["year"] = y
        keep = ["hts8", "year", "brief_description", "quantity_1_code",
                "mexico_rate_type_code", "mexico_ad_val_rate", "mexico_specific_rate"]
        df = df[[c for c in keep if c in df.columns]]
        # Coerce hts8 to numeric (drop non-numeric)
        df["hts8"] = pd.to_numeric(df["hts8"].astype(str).str.replace(".", "", regex=False),
                                    errors="coerce")
        df = df.dropna(subset=["hts8"])
        df["hts8"] = df["hts8"].astype(int)
        usitc_dfs.append(df)

    usitc = pd.concat(usitc_dfs, ignore_index=True)
    print(f"    USITC: {len(usitc):,} rows (2002-2008)")

    # --- Unit values for USITC years ---
    # Read Customs Value (wide -> long)
    val = pd.read_excel(IMPORT_USITC_EXCEL, sheet_name="Customs Value", header=2)
    val = val.drop(columns=["Data Type"], errors="ignore")
    year_cols_val = [c for c in val.columns if str(c).startswith("Year")]
    id_cols_val = [c for c in val.columns if c not in year_cols_val]
    val_long = val.melt(id_vars=id_cols_val, value_vars=year_cols_val,
                        var_name="year_str", value_name="value")
    val_long["year"] = val_long["year_str"].str.extract(r"(\d+)").astype(int)
    val_long = val_long.drop(columns=["year_str"])

    # Read Quantity (wide -> long)
    qty = pd.read_excel(IMPORT_USITC_EXCEL, sheet_name="First Unit of Quantity", header=2)
    qty = qty.drop(columns=["Data Type"], errors="ignore")
    year_cols_qty = [c for c in qty.columns if str(c).startswith("Year")]
    id_cols_qty = [c for c in qty.columns if c not in year_cols_qty]
    qty_long = qty.melt(id_vars=id_cols_qty, value_vars=year_cols_qty,
                        var_name="year_str", value_name="quantity")
    qty_long["year"] = qty_long["year_str"].str.extract(r"(\d+)").astype(int)
    qty_long = qty_long.drop(columns=["year_str"])

    # Merge value and quantity
    uv = val_long.merge(qty_long, on=["HTS Number", "Quantity Description", "year"], how="outer")
    uv = uv[uv["year"] >= 2002].copy()

    # Rename to match Stata conventions
    uv = uv.rename(columns={"HTS Number": "hts8"})
    uv["unitvalue"] = uv["value"] / uv["quantity"]

    # Map quantity description to abbreviation
    uv["quantity_1_code"] = uv["Quantity Description"].map(UNIT_ABBREV)

    # Clean hts8
    uv["hts8"] = pd.to_numeric(uv["hts8"].astype(str).str.replace(".", "", regex=False),
                                errors="coerce")
    uv = uv.dropna(subset=["hts8"])
    uv["hts8"] = uv["hts8"].astype(int)

    # Keep relevant columns and merge onto USITC tariffs
    # Stata merges m:1 on (hts8, year, quantity_1_code) — all 3 keys
    uv_merge = uv[["year", "hts8", "unitvalue", "quantity_1_code"]].drop_duplicates(
        subset=["year", "hts8", "quantity_1_code"], keep="first"
    )

    usitc = usitc.merge(uv_merge, on=["hts8", "year", "quantity_1_code"], how="left")

    # --- Combine Romalis + USITC ---
    # Ensure common columns
    for col in ["mexico_ad_val_rate", "mexico_specific_rate", "unitvalue"]:
        if col not in romalis.columns:
            romalis[col] = np.nan
        if col not in usitc.columns:
            usitc[col] = np.nan

    combined = pd.concat([
        romalis[["hts8", "year", "mexico_ad_val_rate", "mexico_specific_rate", "unitvalue"]],
        usitc[["hts8", "year", "mexico_ad_val_rate", "mexico_specific_rate", "unitvalue"]],
    ], ignore_index=True)

    # Replace sentinel values (USITC stores 9999.999999 instead of exact 10000;
    # Stata's float precision rounds to 10000 and matches, so we round before comparing)
    combined.loc[combined["mexico_ad_val_rate"].round() == 10000, "mexico_ad_val_rate"] = np.nan
    combined.loc[combined["mexico_specific_rate"].round() == 10000, "mexico_specific_rate"] = np.nan

    # Compute AVE
    combined["tariff_AVE"] = (
        combined["mexico_ad_val_rate"]
        + combined["mexico_specific_rate"] / combined["unitvalue"]
    )
    # Fallback: if unitvalue missing but specific_rate == 0
    mask_fallback = combined["unitvalue"].isna() & (combined["mexico_specific_rate"] == 0)
    combined.loc[mask_fallback, "tariff_AVE"] = combined.loc[mask_fallback, "mexico_ad_val_rate"]

    # Collapse duplicates: max tariff per (year, hts8)
    tariff = (
        combined.groupby(["year", "hts8"])["tariff_AVE"]
        .max()
        .reset_index()
    )
    # Keep NaN tariff_AVE rows — they matter for weight normalization in Section 2
    # (NaN-tariff products still have imports that contribute to the weight denominator)

    print(f"    Combined tariff: {len(tariff):,} (year x hts8) pairs, "
          f"{tariff['tariff_AVE'].notna().sum():,} with valid tariff_AVE")
    return tariff


# ==================================================================
# Section 2: Weights and RCA
# ==================================================================
def build_tariff_rca(tariff, use_stata_intermediate=False):
    """Build import-weighted tariff and RCA at the SIC-year level.

    Args:
        tariff: Output of build_tariffs().
        use_stata_intermediate: If True, load Stata's tariff_rca_MEX.dta directly.
            Stata's `duplicates drop year sic, force` produces sort-order-dependent
            results that cannot be exactly replicated in Python. Setting this to True
            ensures exact match with Stata output.
    """
    print("  Section 2: Weights and RCA")

    if use_stata_intermediate and STATA_TARIFF_RCA_PATH.exists():
        print("    Loading Stata intermediate: tariff_rca_MEX.dta")
        result = pd.read_stata(STATA_TARIFF_RCA_PATH, convert_dates=False)
        result["sic"] = result["sic"].astype(int)
        result["year"] = result["year"].astype(int)
        # Extract rca1990_MEX
        rca1990 = result[result["year"] == 1990][["sic", "rca_MEX"]].rename(
            columns={"rca_MEX": "rca1990_MEX"}
        )
        result = result.merge(rca1990, on="sic", how="left")
        print(f"    Tariff-RCA: {len(result):,} (year x SIC) pairs, "
              f"{result['sic'].nunique()} industries")
        return result

    # --- 1990 imports ---
    imports = pd.read_stata(IMPORT_HM_PATH, convert_dates=False)
    imports = imports[imports["year"] == 1990].drop(columns=["year"])
    print(f"    1990 imports: {len(imports):,} HTS8 products")

    # Merge tariff x imports (inner join keeps only matched)
    merged = tariff.merge(imports, on="hts8", how="inner")
    merged["hs6"] = merged["hts8"] // 100
    print(f"    Tariff x imports: {len(merged):,} rows")

    # --- 1990 exports ---
    exports = pd.read_csv(EXPORT_HM_PATH)
    exports = exports[exports["Year"] == 1990].copy()
    exports = exports.drop(columns=["Year", "TradeFlowCode"])

    # Clean column names (trailing spaces)
    exports.columns = exports.columns.str.strip()
    exports = exports.rename(columns={
        "MEX in 1000 USD": "mexin1000usd",
        "USA in 1000 USD": "usain1000usd",
        "WLD in 1000 USD": "wldin1000usd",
        "ProductCode": "hs6",
    })
    exports["mexin1000usd"] = exports["mexin1000usd"].fillna(0)
    exports["usain1000usd"] = exports["usain1000usd"].fillna(0)
    exports["wldex"] = exports["wldin1000usd"] - (exports["mexin1000usd"] + exports["usain1000usd"])

    # Pivot ReporterISO3 to wide (wldexAll, wldexMEX)
    exports_wide = exports.pivot_table(
        index="hs6", columns="ReporterISO3", values="wldex", aggfunc="first"
    ).reset_index()
    exports_wide.columns.name = None

    # Rename columns
    col_map = {}
    for c in exports_wide.columns:
        if c == "hs6":
            continue
        col_map[c] = f"wldex{c}"
    exports_wide = exports_wide.rename(columns=col_map)

    if "wldexMEX" not in exports_wide.columns:
        exports_wide["wldexMEX"] = 0
    exports_wide["wldexMEX"] = exports_wide["wldexMEX"].fillna(0)
    exports_wide["wldexROW"] = exports_wide["wldexAll"] - exports_wide["wldexMEX"]

    # Drop hs6 == "9999AA" equivalent
    exports_wide = exports_wide[exports_wide["hs6"] != "9999AA"]
    exports_wide["hs6"] = pd.to_numeric(exports_wide["hs6"], errors="coerce")
    exports_wide = exports_wide.dropna(subset=["hs6"])
    exports_wide["hs6"] = exports_wide["hs6"].astype(int)

    # Merge exports onto tariff-imports
    merged = merged.merge(
        exports_wide[["hs6", "wldexAll", "wldexMEX", "wldexROW"]],
        on="hs6", how="left"  # copy.do keeps all tariff/import products even without exports
    )
    print(f"    After exports merge: {len(merged):,} rows")

    # --- HS6 -> SIC crosswalk ---
    xw = pd.read_stata(HS6_SIC_PATH, convert_dates=False)
    xw = xw.rename(columns={"sic87dd": "sic"})
    xw = xw.dropna(subset=["sic"])
    xw["hs6"] = xw["hs6"].astype(int)
    xw["sic"] = xw["sic"].astype(int)

    # Drop all (sic, hs6) pairs that have duplicates, matching copy.do:
    # duplicates tag (sic hs6), generate(dup)
    # keep if dup==0 & !missing(sic)
    dup_mask = xw.duplicated(subset=["sic", "hs6"], keep=False)
    xw = xw[~dup_mask & xw["sic"].notna()].copy()
    xw = xw[["hs6", "sic", "share"]]

    # joinby hs6 (many-to-many merge)
    merged = merged.merge(xw, on="hs6", how="inner")
    print(f"    After HS6->SIC joinby: {len(merged):,} rows")

    # --- Weighted tariff ---
    # Handle duplicates within (year, sic, hs6)
    dup_counts = merged.groupby(["year", "sic", "hs6"]).cumcount()
    dup_sizes = merged.groupby(["year", "sic", "hs6"])["hts8"].transform("count")

    for col in ["wldexAll", "wldexMEX", "wldexROW"]:
        merged[col] = merged[col] / dup_sizes

    for col in ["tot_im", "mex_im", "wldexAll", "wldexMEX", "wldexROW"]:
        merged[col] = merged[col] * merged["share"]

    # Import weight within (sic, year)
    merged["tot_mex_im"] = merged.groupby(["sic", "year"])["mex_im"].transform("sum")
    merged["weight_im"] = merged["mex_im"] / merged["tot_mex_im"]
    merged["tariff_AVE_wtd"] = merged["tariff_AVE"] * merged["weight_im"]
    # Sum weighted tariff to SIC level (copy.do: bysort sic year: egen w_wtariff = total(w_w * mex_t))
    merged["tariff_AVE_wtd"] = merged.groupby(["sic", "year"])["tariff_AVE_wtd"].transform("sum")

    # --- RCA ---
    merged["wldexMEX_sictot"] = merged.groupby(["year", "sic"])["wldexMEX"].transform("sum")
    merged["wldexROW_sictot"] = merged.groupby(["year", "sic"])["wldexROW"].transform("sum")

    # Collapse to (year, sic) level
    # Stata uses `duplicates drop year sic, force` after `bysort year sic`,
    # keeping the first row. Stata's stable sort preserves the master data order
    # (sorted by hts8), so the first row per (year, sic) is approximately the
    # product with the smallest hts8.
    merged = merged.sort_values(["year", "sic", "hts8"])
    sic_level = merged.drop_duplicates(subset=["year", "sic"]).copy()
    sic_level = sic_level[["year", "sic", "tariff_AVE_wtd", "tariff_AVE",
                            "wldexMEX_sictot", "wldexROW_sictot"]]
    sic_level["wldexMEX_tot"] = sic_level.groupby("year")["wldexMEX_sictot"].transform("sum")
    sic_level["wldexROW_tot"] = sic_level.groupby("year")["wldexROW_sictot"].transform("sum")
    sic_level["rca_MEX"] = (
        (sic_level["wldexMEX_sictot"] / sic_level["wldexROW_sictot"])
        / (sic_level["wldexMEX_tot"] / sic_level["wldexROW_tot"])
    )

    result = sic_level[["year", "sic", "tariff_AVE_wtd", "tariff_AVE", "rca_MEX"]].copy()

    result = result.dropna(subset=["rca_MEX"])

    # Extract rca1990_MEX (fill across years)
    rca1990 = result[result["year"] == 1990][["sic", "rca_MEX"]].rename(
        columns={"rca_MEX": "rca1990_MEX"}
    )
    result = result.merge(rca1990, on="sic", how="left")

    print(f"    Tariff-RCA: {len(result):,} (year x SIC) pairs, "
          f"{result['sic'].nunique()} industries")
    return result


# ==================================================================
# Section 3: Employment (CBP)
# ==================================================================
def build_employment():
    """Load and process County Business Patterns 1986-2008."""
    print("  Section 3: Employment (CBP)")

    cbp_years = list(range(86, 98)) + list(range(1998, 2009))
    dfs = []

    for y in cbp_years:
        yy = str(y) if y >= 1998 else f"{y:02d}"
        path = CBP_DIR / f"cbp{yy}co.txt"
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        yr = y + 1900 if y < 100 else y
        df["year"] = yr
        dfs.append(df)

    cbp = pd.concat(dfs, ignore_index=True)

    # Drop county=999 (statewide totals)
    cbp = cbp[cbp["fipscty"] != 999].copy()
    cbp["county"] = cbp["fipstate"] * 1000 + cbp["fipscty"]

    # Employment flag imputation
    cbp["empflag"] = cbp["empflag"].astype(str).str.strip()
    for flag, val in EMP_FLAG_MAP.items():
        cbp.loc[cbp["empflag"] == flag, "emp"] = val
    # Flag M: median of emp >= 100000
    m_median = cbp.loc[cbp["emp"] >= 100000, "emp"].median()
    cbp.loc[cbp["empflag"] == "M", "emp"] = m_median

    # County total employment (sic == "----" or naics == "------")
    if "sic" in cbp.columns and "naics" in cbp.columns:
        total_mask = (cbp["sic"].astype(str).str.strip() == "----") | \
                     (cbp["naics"].astype(str).str.strip() == "------")
    elif "sic" in cbp.columns:
        total_mask = cbp["sic"].astype(str).str.strip() == "----"
    else:
        total_mask = cbp["naics"].astype(str).str.strip() == "------"

    emp_tot = cbp.loc[total_mask, ["year", "county", "emp"]].copy()
    emp_tot = emp_tot.rename(columns={"emp": "emp_tot"})
    cbp = cbp.merge(emp_tot, on=["year", "county"], how="left")

    # Manufacturing dummy
    if "sic" in cbp.columns:
        sic_num = pd.to_numeric(cbp["sic"], errors="coerce")
    else:
        sic_num = pd.Series(np.nan, index=cbp.index)
    if "naics" in cbp.columns:
        naics_num = pd.to_numeric(cbp["naics"], errors="coerce")
    else:
        naics_num = pd.Series(np.nan, index=cbp.index)

    cbp["manufacturing"] = (
        (sic_num.between(2000, 3999)) | (naics_num.between(310000, 339999))
    ).astype(int)

    # Drop rows with no industry code (recompute masks on merged df)
    sic_num = pd.to_numeric(cbp.get("sic", pd.Series(dtype="object")), errors="coerce")
    naics_num = pd.to_numeric(cbp.get("naics", pd.Series(dtype="object")), errors="coerce")
    has_industry = sic_num.notna() | naics_num.notna()
    is_total = (cbp.get("sic", pd.Series(dtype="object")).astype(str).str.strip() == "----") | \
               (cbp.get("naics", pd.Series(dtype="object")).astype(str).str.strip() == "------")
    cbp = cbp[has_industry | is_total].copy()

    # Mark total vs industry rows (Stata drops non-numeric SIC/NAICS,
    # so industry rows = rows with valid numeric SIC or NAICS codes)
    cbp["is_total"] = is_total.loc[cbp.index].values

    # Keep relevant columns, unify industry code to numeric SIC where possible
    cbp["sic_num"] = pd.to_numeric(cbp.get("sic", pd.Series(dtype="object")), errors="coerce")
    cbp = cbp[["fipstate", "county", "sic_num", "emp", "year", "manufacturing",
               "emp_tot", "is_total"]].copy()
    cbp = cbp.rename(columns={"sic_num": "sic"})

    print(f"    CBP: {len(cbp):,} rows, {cbp['county'].nunique():,} counties, "
          f"{cbp['year'].nunique()} years")
    return cbp


# ==================================================================
# Section 4: Vulnerability Measure
# ==================================================================
def build_vulnerability(cbp, tariff_rca):
    """Compute NAFTA vulnerability at county-year level."""
    print("  Section 4: Vulnerability Measure")

    # 1990 employment by county x SIC
    emp1990 = cbp[cbp["year"] == 1990][["fipstate", "county", "sic", "emp"]].copy()
    emp1990 = emp1990.rename(columns={"emp": "emp1990"})
    emp1990 = emp1990.dropna(subset=["sic"])
    emp1990["sic"] = emp1990["sic"].astype(int)

    # joinby sic with tariff_rca (many-to-many)
    merged = emp1990.merge(tariff_rca, on="sic", how="inner")
    print(f"    Emp1990 x tariff_rca: {len(merged):,} rows")

    # Compute vulnerability per (county, year)
    merged["num"] = merged["emp1990"] * merged["rca_MEX"] * merged["tariff_AVE_wtd"]
    merged["den"] = merged["emp1990"] * merged["rca_MEX"]

    vuln = merged.groupby(["county", "year"]).agg(
        numerator=("num", "sum"),
        denominator=("den", "sum"),
    ).reset_index()
    vuln["vulnerability"] = vuln["numerator"] / vuln["denominator"]

    # vulnerability1990
    v1990 = vuln[vuln["year"] == 1990][["county", "vulnerability"]].rename(
        columns={"vulnerability": "vulnerability1990"}
    )
    # Some counties may not have 1990 data; use mode-like approach
    vuln = vuln.merge(v1990, on="county", how="left")
    # Fill from any year if 1990 missing
    if vuln["vulnerability1990"].isna().any():
        fill = vuln.groupby("county")["vulnerability1990"].transform("first")
        vuln["vulnerability1990"] = vuln["vulnerability1990"].fillna(fill)

    vuln = vuln.drop_duplicates(subset=["county", "year"])

    # Get state from emp1990
    state_map = emp1990.drop_duplicates("county")[["county", "fipstate"]].rename(
        columns={"fipstate": "state"}
    )
    vuln = vuln.merge(state_map, on="county", how="left")

    # Restrict to contiguous 48 + DC (drop Alaska=2, Hawaii=15)
    vuln = vuln[~vuln["state"].isin([2, 15])].copy()

    # Quartiles and scaling
    counties_1990 = vuln[vuln["year"] == 1990][["county", "vulnerability1990"]].drop_duplicates()
    counties_1990["quartile"] = pd.qcut(
        counties_1990["vulnerability1990"], 4, labels=[1, 2, 3, 4]
    ).astype(int)

    # Compute scaling factor
    q4_mean = counties_1990.loc[counties_1990["quartile"] == 4, "vulnerability1990"].mean()
    q1_mean = counties_1990.loc[counties_1990["quartile"] == 1, "vulnerability1990"].mean()
    scale_factor = q4_mean - q1_mean
    counties_1990["vulnerability1990_scaled"] = counties_1990["vulnerability1990"] / scale_factor

    vuln = vuln.merge(
        counties_1990[["county", "quartile", "vulnerability1990_scaled"]],
        on="county", how="left"
    )

    vuln = vuln[["state", "county", "year", "vulnerability", "vulnerability1990",
                 "vulnerability1990_scaled", "quartile"]].copy()

    print(f"    Vulnerability: {vuln['county'].nunique():,} counties, "
          f"scale factor = {scale_factor:.4f}")
    return vuln


# ==================================================================
# Section 5: Controls
# ==================================================================
def build_population():
    """Build population panel 1986-2008."""
    print("    Population ...")

    # --- 1986-1989 (Excel) ---
    pop_80s = []
    for year in range(1986, 1990):
        df = pd.read_excel(DEMO_80S_PATH, sheet_name=str(year), header=5)
        df = df.dropna(subset=["Year of Estimate"])
        df = df.rename(columns={
            "Year of Estimate": "year",
            "FIPS State and County Codes": "county",
        })
        # Age columns for working age (15-64)
        working_cols = [c for c in df.columns if any(
            age in c for age in ["15 to 19", "20 to 24", "25 to 29", "30 to 34",
                                  "35 to 39", "40 to 44", "45 to 49", "50 to 54",
                                  "55 to 59", "60 to 64"]
        )]
        all_age_cols = [c for c in df.columns if c not in
                        ["year", "county", "Race/Sex Indicator"]]

        df["pop_total"] = df[all_age_cols].sum(axis=1)
        df["pop_working"] = df[working_cols].sum(axis=1)

        df["county"] = df["county"].astype(int)
        df["year"] = df["year"].astype(int)
        df["state"] = (df["county"] // 1000).astype(str).str.zfill(2)

        agg = df.groupby(["state", "county", "year"]).agg(
            pop_total=("pop_total", "sum"),
            pop_working=("pop_working", "sum"),
        ).reset_index()
        pop_80s.append(agg)

    pop_80s = pd.concat(pop_80s, ignore_index=True)

    # --- 1990-1999 (.dta files) ---
    pop_90s = []
    for year in range(1990, 2000):
        path = RAW / "demographic" / f"stch{year}.dta"
        df = pd.read_stata(path, convert_categoricals=False)
        df["state"] = df["state"].astype(str).str.zfill(2)
        df["county"] = df["county"].astype(str).str.zfill(5)

        # Total population
        total = df.groupby(["state", "county"])["pop"].sum().reset_index()
        total = total.rename(columns={"pop": "pop_total"})

        # Working age: agegroup 4-13 (15-64)
        working = df[df["agegroup"].between(4, 13)].groupby(
            ["state", "county"]
        )["pop"].sum().reset_index()
        working = working.rename(columns={"pop": "pop_working"})

        merged = total.merge(working, on=["state", "county"], how="left")
        merged["year"] = year
        merged["county"] = merged["county"].astype(int)
        pop_90s.append(merged)

    pop_90s = pd.concat(pop_90s, ignore_index=True)

    # --- 2000-2008 (.dta) ---
    df = pd.read_stata(DEMO_00S_PATH, convert_categoricals=False)
    df = df[~df["yearref"].isin([1, 12])].copy()
    df = df[(df["year"] >= 2000) & (df["year"] <= 2008)].copy()
    df = df[df["agegrp"] != 99].copy()  # Drop totals

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(5)

    total = df.groupby(["year", "state", "county"])["tot_pop"].sum().reset_index()
    total = total.rename(columns={"tot_pop": "pop_total"})

    working = df[df["agegrp"].between(4, 13)].groupby(
        ["year", "state", "county"]
    )["tot_pop"].sum().reset_index()
    working = working.rename(columns={"tot_pop": "pop_working"})

    pop_00s = total.merge(working, on=["year", "state", "county"], how="left")
    pop_00s["county"] = pop_00s["county"].astype(int)

    # --- Combine ---
    pop_all = pd.concat([pop_80s, pop_90s, pop_00s], ignore_index=True)
    pop_all["county"] = pop_all["county"].astype(int)

    # pop1990_total per county
    p1990 = pop_all[pop_all["year"] == 1990][["county", "pop_total"]].rename(
        columns={"pop_total": "pop1990_total"}
    )
    pop_all = pop_all.merge(p1990, on="county", how="left")

    print(f"      {len(pop_all):,} county-years")
    return pop_all


def build_education():
    """Load 1990 county education attainment."""
    print("    Education ...")
    df = pd.read_excel(EDUCATION_PATH, sheet_name="Education 1970 to 2017", header=4)
    df = df.dropna(subset=["FIPS Code"])

    df = df.rename(columns={
        "FIPS Code": "county",
        "Percent of adults with less than a high school diploma, 1990": "less_highschool1990",
        "Percent of adults with a high school diploma only, 1990": "highschool1990",
        "Percent of adults completing some college or associate's degree, 1990": "college1990",
        "Percent of adults with a bachelor's degree or higher, 1990": "bachelor_higher1990",
    })

    df["county"] = pd.to_numeric(df["county"], errors="coerce")
    df = df.dropna(subset=["county"])
    df["county"] = df["county"].astype(int)

    # Drop state-level aggregates (FIPS ending in 000)
    df = df[df["county"] % 1000 != 0].copy()

    # FIPS fixes
    df.loc[df["county"] == 12086, "county"] = 12025
    df.loc[df["county"] == 46102, "county"] = 46113

    # Convert percentages to proportions
    for col in ["less_highschool1990", "highschool1990", "college1990", "bachelor_higher1990"]:
        df[col] = df[col] / 100

    df["state"] = (df["county"] // 1000).astype(str).str.zfill(2)

    edu = df[["county", "state", "less_highschool1990", "highschool1990",
              "college1990", "bachelor_higher1990"]].copy()
    print(f"      {len(edu):,} counties")
    return edu


def build_manushare():
    """Manufacturing share from 1990 CBP."""
    print("    Manufacturing share ...")
    cbp90 = pd.read_csv(CBP_DIR / "cbp90co.txt", encoding="latin1", low_memory=False)
    cbp90 = cbp90[cbp90["fipscty"] != 999].copy()
    cbp90["county"] = cbp90["fipstate"] * 1000 + cbp90["fipscty"]

    # Keep total and manufacturing rows
    # Note: Stata uses raw emp (no flag imputation) for manushare
    cbp90["sic_str"] = cbp90["sic"].astype(str).str.strip()
    subset = cbp90[cbp90["sic_str"].isin(["----", "20--"])].copy()

    subset["label"] = subset["sic_str"].map({"----": "total", "20--": "manufacturing"})

    # Pivot
    pivoted = subset.pivot_table(
        index=["fipstate", "county"], columns="label", values="emp", aggfunc="first"
    ).reset_index()
    pivoted.columns.name = None

    pivoted["manushare1990"] = pivoted.get("manufacturing", 0) / pivoted["total"]
    pivoted = pivoted.rename(columns={
        "fipstate": "state",
        "total": "emp_total1990",
        "manufacturing": "emp_manufacturing1990",
    })

    result = pivoted[["state", "county", "manushare1990",
                       "emp_total1990", "emp_manufacturing1990"]].copy()
    print(f"      {len(result):,} counties")
    return result


def build_income():
    """Load 1989 median household income by county."""
    print("    Income ...")
    # Read CSV (rows 10+ in Stata = skiprows=9 in Python)
    df = pd.read_csv(INCOME_PATH, header=None, skiprows=9)

    # Drop rows without commas in col 0 (non-county rows)
    df = df[df[0].astype(str).str.contains(",", na=False)].copy()
    df = df.dropna(subset=[0]).copy()

    # Parse: "County Name, ST"
    splits = df[0].str.split(",", n=1, expand=True)
    df["county_name"] = splits[0].str.strip()
    df["st_abrv"] = splits[1].str.strip()

    # Income columns: v2=1999, v3=1989, v4=1979, v5=1969
    df["income1989"] = pd.to_numeric(df[1].astype(str).str.replace(",", ""), errors="coerce")

    # Clean county_name to match crosswalk
    df["county_name"] = (
        df["county_name"]
        .str.replace("County", "", regex=False)
        .str.replace("Census Area", "", regex=False)
        .str.replace("Parish", "", regex=False)
        .str.replace("Borough", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace("'", "", regex=False)
    )
    # VA city fix
    va_mask = df["st_abrv"] == "VA"
    df.loc[va_mask, "county_name"] = df.loc[va_mask, "county_name"].str.replace(
        "city", "City", regex=False
    )
    df["county_name"] = df["county_name"].str.strip().str.lower()

    # Specific name fixes (matching Stata)
    name_fixes = {
        "yellowstone national park": "yellowstone nat park",
        "laplata": "la plata",
        "prince george's": "prince georges",
        "queen anne's": "queen annes",
        "st mary's": "st marys",
    }
    for old, new in name_fixes.items():
        df.loc[df["county_name"] == old, "county_name"] = new

    # Dekalb-style fixes
    df.loc[(df["county_name"] == "dekalb") & df["st_abrv"].isin(["AL", "GA", "IL", "MO", "TN"]),
           "county_name"] = "de kalb"
    df.loc[(df["county_name"] == "desoto") & df["st_abrv"].isin(["FL", "MS"]),
           "county_name"] = "de soto"
    df.loc[(df["county_name"] == "dewitt") & (df["st_abrv"] == "TX"),
           "county_name"] = "de witt"
    df.loc[(df["county_name"] == "dupage") & (df["st_abrv"] == "IL"),
           "county_name"] = "du page"
    df.loc[(df["county_name"] == "lagrange") & (df["st_abrv"] == "IN"),
           "county_name"] = "la grange"
    df.loc[(df["county_name"] == "lamoure") & (df["st_abrv"] == "ND"),
           "county_name"] = "la moure"
    df.loc[(df["county_name"] == "mc kean") & (df["st_abrv"] == "PA"),
           "county_name"] = "mckean"
    df.loc[(df["county_name"] == "o'brien") & (df["st_abrv"] == "IA"),
           "county_name"] = "o brien"
    df.loc[(df["county_name"] == "ste genevieve") & (df["st_abrv"] == "MO"),
           "county_name"] = "ste. genevieve"
    df.loc[(df["county_name"] == "debaca") & (df["st_abrv"] == "NM"),
           "county_name"] = "de baca"
    df.loc[(df["county_name"] == "colonial heights city") & (df["st_abrv"] == "VA"),
           "county_name"] = "colonial heights cit"

    df = df.drop_duplicates(subset=["county_name", "st_abrv"])

    # Load FIPS crosswalk
    xw = pd.read_stata(FIPS_XW_PATH, convert_dates=False)
    xw["county_name"] = xw["county_name"].str.strip().str.lower()

    # Merge
    inc = df.merge(xw, on=["county_name", "st_abrv"], how="inner")
    inc = inc.rename(columns={"countyfips": "county", "statefip": "state"})
    inc["county"] = inc["county"].astype(int)

    result = inc[["state", "county", "income1989"]].copy()
    print(f"      {len(result):,} counties matched")
    return result


def build_county_employment(cbp):
    """Extract county-year total employment from CBP.

    Stata sums all industry-specific rows (after destring sic/naics, force;
    drop if missing). We replicate this by summing emp from non-total rows.
    Counties missing from a year (no valid industry rows) will be absent,
    matching Stata's xtbalance behavior.
    """
    industry = cbp[~cbp["is_total"]].copy()
    emp = industry.groupby(["year", "county"])["emp"].sum().reset_index()
    emp["emp"] = emp["emp"].astype(int)
    return emp


def build_controls(cbp):
    """Build all control variables and merge."""
    print("  Section 5: Controls")

    pop = build_population()
    edu = build_education()
    manu = build_manushare()
    inc = build_income()
    emp = build_county_employment(cbp)

    return pop, edu, manu, inc, emp


# ==================================================================
# Section 6: CZ Aggregation
# ==================================================================
def collapse_to_cz(panel):
    """Collapse county panel to commuting zone level with population weights."""
    print("  Section 6: CZ Aggregation")

    cz = pd.read_stata(CZ_XW_PATH, convert_dates=False)
    cz = cz.rename(columns={"cty_fips": "county", "czone": "cz"})
    cz["county"] = cz["county"].astype(int)
    cz["cz"] = cz["cz"].astype(int)
    cz = cz.drop_duplicates(subset=["county"])

    merged = panel.merge(cz, on="county", how="inner")
    n_before = panel["county"].nunique()
    n_after = merged["county"].nunique()
    print(f"    CZ merge: {n_after}/{n_before} counties matched")

    # Population-weighted collapse
    weights = merged["pop1990_total"].fillna(0)

    def weighted_mean(col):
        def _wm(g):
            w = weights.loc[g.index]
            if w.sum() == 0:
                return np.nan
            return np.average(g, weights=w)
        return _wm

    # Ratio variables: pop-weighted mean
    # Level variables: sum
    cz_panel = merged.groupby(["cz", "year"]).apply(
        lambda g: pd.Series({
            "vulnerability1990_scaled": np.average(
                g["vulnerability1990_scaled"].fillna(0),
                weights=g["pop1990_total"].fillna(0)
            ) if g["pop1990_total"].fillna(0).sum() > 0 else np.nan,
            "manushare1990": np.average(
                g["manushare1990"].fillna(0),
                weights=g["pop1990_total"].fillna(0)
            ) if g["pop1990_total"].fillna(0).sum() > 0 else np.nan,
            "pop1990_total": g["pop1990_total"].sum(),
            "emp": g["emp"].sum(),
        }),
        include_groups=False,
    ).reset_index()

    cz_panel = cz_panel.rename(columns={"cz": "czone"})
    print(f"    CZ panel: {len(cz_panel):,} rows, {cz_panel['czone'].nunique()} CZs")
    return cz_panel


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    pipeline_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building NAFTA exposure variables ...\n")

    # Section 1: Tariffs
    tariff = build_tariffs()

    # Section 2: Weights and RCA
    # use_stata_intermediate=True for exact Stata replication (recommended for validation);
    # set to False for fully independent Python computation.
    tariff_rca = build_tariff_rca(tariff, use_stata_intermediate=False)

    # Section 3: Employment
    cbp = build_employment()

    # Section 4: Vulnerability
    vuln = build_vulnerability(cbp, tariff_rca)

    # Section 5: Controls
    pop, edu, manu, inc, emp = build_controls(cbp)

    # --- Assemble county panel ---
    # Stata approach: vulnerability (county x year for tariff years) is merged
    # with employment (county x year 1986-2008). The outer merge on employment
    # provides the 1986-2008 year backbone. Time-invariant controls are filled
    # across all years per county, then the panel is balanced.
    print("\n  Assembling county panel ...")

    # Extract time-invariant vulnerability per county
    vuln_ti = vuln.drop_duplicates("county")[
        ["state", "county", "vulnerability1990_scaled", "quartile"]
    ].copy()

    # Start from employment (county x year, full 1986-2008 backbone)
    panel = emp.copy()

    # Derive state from county FIPS code
    panel["state"] = panel["county"] // 1000

    # Restrict to contiguous 48 + DC (drop Alaska=2, Hawaii=15)
    panel = panel[~panel["state"].isin([2, 15])].copy()

    # Merge vulnerability (time-invariant)
    panel = panel.merge(vuln_ti[["county", "vulnerability1990_scaled", "quartile"]],
                        on="county", how="left")

    # Merge population (time-varying)
    panel = panel.merge(pop[["county", "year", "pop_total", "pop_working", "pop1990_total"]],
                        on=["county", "year"], how="left")

    # Merge education (time-invariant)
    panel = panel.merge(edu[["county", "less_highschool1990", "highschool1990",
                             "college1990", "bachelor_higher1990"]],
                        on="county", how="left")

    # Merge manufacturing share (time-invariant)
    panel = panel.merge(manu[["county", "manushare1990"]], on="county", how="left")

    # Merge income (time-invariant)
    panel = panel.merge(inc[["county", "income1989"]], on="county", how="left")

    # Fill time-invariant vars across years per county
    for col in ["vulnerability1990_scaled", "manushare1990", "pop1990_total"]:
        panel[col] = panel.groupby("county")[col].transform(
            lambda x: x.fillna(x.dropna().iloc[0]) if x.notna().any() else x
        )

    # Balance panel: keep only counties present in all years 1986-2008
    # AND with complete data (non-NaN vulnerability and manushare),
    # matching Stata's xtbalance behavior
    year_range = list(range(1986, 2009))
    n_years = len(year_range)
    county_counts = panel.groupby("county")["year"].nunique()
    balanced_counties = county_counts[county_counts == n_years].index
    panel = panel[panel["county"].isin(balanced_counties)].copy()

    # Drop counties with missing key variables (replicates Stata's xtbalance)
    snap = panel.drop_duplicates("county")
    complete = snap[
        snap["vulnerability1990_scaled"].notna() &
        snap["manushare1990"].notna()
    ]["county"]
    panel = panel[panel["county"].isin(complete)].copy()

    panel = panel.sort_values(["state", "county", "year"]).reset_index(drop=True)

    # Save county-level
    county_path = OUT_DIR / "12_nafta_vars_county.parquet"
    panel.to_parquet(county_path)

    # Section 6: CZ Aggregation
    cz_panel = collapse_to_cz(panel)
    cz_path = OUT_DIR / "12_nafta_vars_cz.parquet"
    cz_panel.to_parquet(cz_path)

    elapsed = time.time() - pipeline_start

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY: NAFTA Exposure Variables")
    print("=" * 72)

    print(f"\n  County panel:")
    print(f"    Shape: {panel.shape[0]:,} rows ({panel['county'].nunique():,} counties "
          f"x {panel['year'].nunique()} years)")
    print(f"    Year range: {panel['year'].min()} - {panel['year'].max()}")
    print(f"    Columns: {panel.columns.tolist()}")
    print(f"    vulnerability1990_scaled: "
          f"mean={panel['vulnerability1990_scaled'].mean():.4f}, "
          f"std={panel['vulnerability1990_scaled'].std():.4f}")
    print(f"    manushare1990: mean={panel['manushare1990'].mean():.4f}")

    print(f"\n  CZ panel:")
    print(f"    Shape: {cz_panel.shape[0]:,} rows ({cz_panel['czone'].nunique()} CZs "
          f"x {cz_panel['year'].nunique()} years)")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Saved to -> {county_path}")
    print(f"              {cz_path}")
    print("=" * 72)
