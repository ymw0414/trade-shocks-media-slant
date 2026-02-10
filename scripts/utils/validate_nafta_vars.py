"""Validation script for 12_build_nafta_vars.py against Choi et al. (2024) PUBLIC replication."""
import pandas as pd
import numpy as np

# Our output
my = pd.read_parquet(r'C:\Users\ymw04\Dropbox\shifting_slant\data\processed\econ\minwoo\12_nafta_vars_county.parquet')

# Choi et al. PUBLIC replication ground truth
gt_path = (r'C:\Users\ymw04\Dropbox\shifting_slant\replication\Replication Project'
           r'\Replication Project\Choi et al ARE 2024_Replication Package'
           r'\nafta_politics_replication_submission_PUBLIC\data\working_data\vulnerability.dta')
gt = pd.read_stata(gt_path, convert_dates=False)
gt['county'] = gt['county'].astype(int)

print(f"Our data: {my['county'].nunique()} counties, years {my['year'].min()}-{my['year'].max()}")
print(f"Ground truth: {gt['county'].nunique()} counties, years {gt['year'].min()}-{gt['year'].max()}")
print(f"Ground truth columns: {gt.columns.tolist()}")

# Compare vulnerability at year=1990
my90 = my[my['year'] == 1990].set_index('county')
gt90 = gt[gt['year'] == 1990].set_index('county')

common = sorted(set(my90.index) & set(gt90.index))
print(f"\nCommon counties at 1990: {len(common)}")

# Our vulnerability1990 (unscaled) vs ground truth county_vul_own
my_vuln = my90.loc[common, 'vulnerability1990_scaled'].astype(float)
gt_vuln = gt90.loc[common, 'county_vul_own'].astype(float)

# Check raw correlation (our values are scaled, theirs are not)
# So let's compute unscaled vulnerability from our data
# vulnerability1990_scaled = vulnerability1990 / scale_factor
# We need vulnerability1990 to compare

# Actually, let's check if our vulnerability column exists
if 'vulnerability' in my.columns:
    my_raw = my[my['year'] == 1990].set_index('county')['vulnerability'].astype(float)
elif 'vulnerability1990' in my.columns:
    my_raw = my[my['year'] == 1990].set_index('county')['vulnerability1990'].astype(float)
else:
    # Reconstruct from vulnerability1990_scaled
    # We don't know scale_factor, but correlation doesn't depend on scaling
    my_raw = my_vuln

my_raw_common = my_raw.loc[common]

corr = my_raw_common.corr(gt_vuln)
diff = (my_raw_common - gt_vuln).abs()
print(f"\nCorrelation (our raw vs ground truth county_vul_own): {corr:.6f}")
print(f"Max absolute diff: {diff.max():.6f}")
print(f"Mean absolute diff: {diff.mean():.6f}")

# Also compare scaled vulnerability (rank correlation should be same)
print(f"\nOur vulnerability1990_scaled: min={my_vuln.min():.6f}, median={my_vuln.median():.6f}, max={my_vuln.max():.6f}")
print(f"GT county_vul_own: min={gt_vuln.min():.6f}, median={gt_vuln.median():.6f}, max={gt_vuln.max():.6f}")

# Check ratio (if scaling is the only difference, ratio should be constant)
ratio = my_raw_common / gt_vuln
ratio_valid = ratio.replace([np.inf, -np.inf], np.nan).dropna()
if len(ratio_valid) > 0:
    print(f"\nRatio (ours/GT): mean={ratio_valid.mean():.6f}, std={ratio_valid.std():.6f}")

# Rank correlation (Spearman) - robust to scaling
from scipy.stats import spearmanr
spear_corr, _ = spearmanr(my_raw_common.values, gt_vuln.values)
print(f"Spearman rank correlation: {spear_corr:.6f}")

# County counts
my_counties = set(my['county'].unique())
gt_counties = set(gt['county'].unique())
extra = sorted(my_counties - gt_counties)
missing = sorted(gt_counties - my_counties)
print(f"\nCounties in ours but not GT: {len(extra)}")
print(f"Counties in GT but not ours: {len(missing)}")
