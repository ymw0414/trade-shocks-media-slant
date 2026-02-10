"""
12c_build_china_shock.py

Prepare the ADH (2013) China shock measure at the commuting-zone level,
following Choi et al. (2024) who use `d_tradeotch_pw_lag` as a control.

Source: David Dorn's website (ddorn.net)
  Autor, Dorn & Hanson (2013) "The China Syndrome" file archive.
  workfile_china.dta contains CZ-level trade exposure, 722 CZs × 2 periods.

Variable: d_tradeotch_pw_lag
  = decadal change in Chinese imports per worker (IV using imports to
    other high-income countries), lagged industry composition weights.
  Period 1990: captures 1990-2000 change.
  Period 2000: captures 2000-2007 change.

We extract the 1990 period (relevant for NAFTA-era analysis) as a
cross-sectional CZ-level measure for merging in step 13.

Inputs:
  - data/raw/econ/adh/adh_extracted/Autor-Dorn-Hanson-ChinaSyndrome-
    FileArchive/dta/workfile_china.dta

Outputs:
  - data/processed/econ/minwoo/12c_china_shock_cz.parquet
"""

import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
ADH_PATH = (BASE_DIR / "data" / "raw" / "econ" / "adh" / "adh_extracted"
            / "Autor-Dorn-Hanson-ChinaSyndrome-FileArchive"
            / "dta" / "workfile_china.dta")
OUT_DIR = BASE_DIR / "data" / "processed" / "econ" / "minwoo"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ADH workfile_china.dta ...")
    df = pd.read_stata(ADH_PATH, convert_dates=False)
    print(f"  {len(df)} rows, {df['czone'].nunique()} CZs, "
          f"periods: {sorted(df['yr'].unique())}")

    # Keep 1990 period (captures 1990-2000 change in Chinese imports)
    df90 = df[df["yr"] == 1990][["czone", "d_tradeotch_pw_lag"]].copy()
    df90 = df90.rename(columns={
        "czone": "cz",
        "d_tradeotch_pw_lag": "china_shock",
    })
    df90["cz"] = df90["cz"].astype(int)

    print(f"\n  1990 period: {len(df90)} CZs")
    print(f"  china_shock (d_tradeotch_pw_lag):")
    print(f"    mean = {df90['china_shock'].mean():.4f}")
    print(f"    std  = {df90['china_shock'].std():.4f}")
    print(f"    min  = {df90['china_shock'].min():.4f}")
    print(f"    max  = {df90['china_shock'].max():.4f}")
    print(f"    p25  = {df90['china_shock'].quantile(0.25):.4f}")
    print(f"    p50  = {df90['china_shock'].quantile(0.50):.4f}")
    print(f"    p75  = {df90['china_shock'].quantile(0.75):.4f}")

    out_path = OUT_DIR / "12c_china_shock_cz.parquet"
    df90.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
