"""Merge Editor & Publisher (ICPSR 30261) data with our newspaper panel."""
import os
import sys
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).resolve().parent / "nlp"))
import pipeline_config as cfg

base = str(cfg.BASE_DIR / 'data' / 'editor_publisher' / 'ICPSR_30261')

# ---- Load E&P datasets ----
ds1 = pd.read_csv(f'{base}/DS0001/30261-0001-Data.tsv', sep='\t', low_memory=False, encoding='latin-1')
ds2 = pd.read_csv(f'{base}/DS0002/30261-0002-Data.tsv', sep='\t', low_memory=False, encoding='latin-1')
ds3 = pd.read_csv(f'{base}/DS0003/30261-0003-Data.tsv', sep='\t', low_memory=False, encoding='latin-1')
ds6 = pd.read_csv(f'{base}/DS0006/30261-0006-Data.tsv', sep='\t', low_memory=False, encoding='latin-1')

years = [1988, 1992, 1996, 2000, 2004]
ep = ds2[ds2['year'].isin(years)].copy()
events = ds3[ds3['year'].isin(years)][['recid', 'cnty90']].copy()
ep = ep.merge(events, on='recid', how='left')
ep = ep.merge(ds1[['permid', 'citypermid', 'papername_constant']], on='permid', how='left')
ep = ep.merge(ds6[['citypermid', 'cityname_constant', 'state']], on='citypermid', how='left')
ep['cnty90_int'] = pd.to_numeric(ep['cnty90'], errors='coerce').astype('Int64')

# ---- Load our panel ----
panel = pd.read_parquet(str(cfg.PANEL_DIR / '14_regression_panel.parquet'))
our_papers = panel[['paper', 'fips']].drop_duplicates().reset_index(drop=True)

# ---- Manual matches (permid) ----
manual = {
    'CHRISTIAN SCIENCE MONITOR': None,
    'USA TODAY': None,
    'Daily Pennsylvanian, The: University of Pennsylvania (Philadelphia, PA)': None,
    'Hunterdon County Democrat (Flemington, NJ)': None,
    'Hunterdon Observer (Flemington, NJ)': None,
    'Miami New Times (FL)': None,
    'Nuevo Herald, El (Miami, FL)': None,
    'Sunday News (Lancaster, PA)': 1323,
    'Sunday Republican (Springfield, MA)': 7616,
    'New Hampshire Sunday News (Manchester, NH)': 6951,
    'New Hampshire Union Leader (Manchester, NH)': 6951,
    'Miami Herald, The (FL)': 4934,
    'Staten Island Advance (NY)': 5662,
    'Virginian-Pilot, The (Norfolk, VA)': 3475,
    'Sun, The (Baltimore, MD)': 7438,
    'State, The (Columbia, SC)': 2886,
    'Times, The (Munster, IN)': 4552,
    'Press of Atlantic City, The (NJ)': 3282,
    'Patriot-News, The (Harrisburg, PA)': 5713,
    'Lewiston Morning Tribune (ID)': 3587,
    'Advocate, The (Baton Rouge, La.)': 852,
    'Chicago Sun-Times': 6924,
}

# ---- Auto-match by FIPS + fuzzy name ----
fips_corrections = {12086: 12025, 36085: 36047, 51710: 51810}


def clean(s):
    s = str(s).lower()
    for rm in [', the', 'the ', '(', ')', ',']:
        s = s.replace(rm, ' ')
    s = re.sub(r'\b[a-z]{2}\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()


match_results = []
for _, row in our_papers.iterrows():
    pname = row['paper']
    fips = row['fips']

    if pname in manual:
        pid = manual[pname]
        mtype = 'manual_none' if pid is None else 'manual'
        match_results.append({'paper': pname, 'ep_permid': pid, 'match_type': mtype})
        continue

    fips_tries = [fips]
    if fips in fips_corrections:
        fips_tries.append(fips_corrections[fips])

    cn1 = clean(pname)
    best_score, best_pid = 0, None

    for try_fips in fips_tries:
        for yr in years:
            cands = ep[(ep['cnty90_int'] == try_fips) & (ep['year'] == yr)]
            for _, erow in cands.iterrows():
                score = SequenceMatcher(None, cn1, clean(erow['papername'])).ratio()
                if score > best_score:
                    best_score = score
                    best_pid = int(erow['permid'])

    if best_score >= 0.4:
        match_results.append({'paper': pname, 'ep_permid': best_pid,
                              'match_type': f'auto_{best_score:.2f}'})
    else:
        match_results.append({'paper': pname, 'ep_permid': None,
                              'match_type': f'failed_{best_score:.2f}'})

mdf = pd.DataFrame(match_results)
matched = mdf[mdf['ep_permid'].notna()]
unmatched = mdf[mdf['ep_permid'].isna()]

print(f'=== MATCH RESULTS ===')
print(f'Matched: {len(matched)} / {len(mdf)}')
print(f'Unmatched: {len(unmatched)}')
for _, r in unmatched.iterrows():
    print(f'  {r["paper"]:60s} ({r["match_type"]})')

# ---- Build E&P panel for matched papers ----
successor = dict(zip(ds1['permid'], ds1['nextpermid']))

ep_panel_rows = []
for _, r in matched.iterrows():
    pid = int(r['ep_permid'])
    paper = r['paper']

    pids_to_try = [pid]
    nxt = successor.get(pid)
    try:
        nxt_int = int(float(nxt)) if pd.notna(nxt) else 0
    except (ValueError, TypeError):
        nxt_int = 0
    if nxt_int > 0:
        pids_to_try.append(nxt_int)

    for yr in years:
        found = False
        for try_pid in pids_to_try:
            row_data = ep[(ep['permid'] == try_pid) & (ep['year'] == yr)]
            if len(row_data) > 0:
                rd = row_data.iloc[0]
                ep_panel_rows.append({
                    'paper': paper,
                    'ep_year': yr,
                    'ep_permid': try_pid,
                    'ep_circ': rd['circ'],
                    'ep_adprice': pd.to_numeric(rd['adprice'], errors='coerce'),
                    'ep_subprice': pd.to_numeric(rd['subprice'], errors='coerce'),
                    'ep_ownerpermid': str(int(float(rd['ownerpermid']))) if pd.notna(rd['ownerpermid']) and str(rd['ownerpermid']).strip() else None,
                    'ep_party_endorsement': rd['party_endorsement'],
                })
                found = True
                break
        if not found:
            ep_panel_rows.append({
                'paper': paper, 'ep_year': yr, 'ep_permid': None,
                'ep_circ': None, 'ep_adprice': None, 'ep_subprice': None,
                'ep_ownerpermid': None, 'ep_party_endorsement': None,
            })

ep_panel = pd.DataFrame(ep_panel_rows)

print(f'\n=== E&P PANEL ===')
print(f'Shape: {ep_panel.shape}')
print(f'Papers: {ep_panel["paper"].nunique()}, Years: {sorted(ep_panel["ep_year"].unique())}')

for yr in years:
    yr_data = ep_panel[ep_panel['ep_year'] == yr]
    n_circ = yr_data['ep_circ'].notna().sum()
    n_ad = yr_data['ep_adprice'].notna().sum()
    n_own = yr_data['ep_ownerpermid'].notna().sum()
    print(f'  {yr}: circ={n_circ}, adprice={n_ad}, owner={n_own} / {len(yr_data)} papers')

# ---- Ownership analysis ----
print(f'\n=== OWNERSHIP (1988 baseline) ===')
own88 = ep_panel[ep_panel['ep_year'] == 1988].dropna(subset=['ep_ownerpermid']).copy()
owner_counts = own88['ep_ownerpermid'].value_counts()
chain_ids = owner_counts[owner_counts > 1].index
own88['is_chain'] = own88['ep_ownerpermid'].isin(chain_ids)
print(f'Unique owners: {len(owner_counts)}')
print(f'Chain papers (owner has >1 paper in sample): {own88["is_chain"].sum()}')
print(f'Independent (single-paper owner in sample): {(~own88["is_chain"]).sum()}')

# But "chain" should be based on ALL E&P papers, not just our sample
ep88_all = ep[ep['year'] == 1988].copy()
def safe_owner_str(x):
    try:
        return str(int(float(x))) if pd.notna(x) and str(x).strip() else None
    except (ValueError, TypeError):
        return None

ep88_all['ownerpermid_str'] = ep88_all['ownerpermid'].apply(safe_owner_str)
all_owner_counts = ep88_all['ownerpermid_str'].value_counts()
chain_ids_all = set(all_owner_counts[all_owner_counts > 1].index)
own88['is_chain_all'] = own88['ep_ownerpermid'].isin(chain_ids_all)
print(f'\nUsing ALL E&P papers for chain definition:')
print(f'Chain papers: {own88["is_chain_all"].sum()}')
print(f'Independent: {(~own88["is_chain_all"]).sum()}')

print(f'\nTop 10 chains (papers in our sample):')
chain_own = own88[own88['is_chain_all']].copy()
for oid, cnt in chain_own['ep_ownerpermid'].value_counts().head(10).items():
    papers = chain_own[chain_own['ep_ownerpermid'] == oid]['paper'].tolist()
    # Get chain name from any E&P paper
    sample_pub = ep88_all[ep88_all['ownerpermid_str'] == oid]['origpublisher'].iloc[0] if len(
        ep88_all[ep88_all['ownerpermid_str'] == oid]) > 0 else 'Unknown'
    total_in_ep = all_owner_counts.get(oid, 0)
    print(f'  Owner {oid} ({total_in_ep} papers total): {papers}')

# ---- Circulation summary ----
print(f'\n=== CIRCULATION SUMMARY ===')
ep_panel['ep_circ'] = pd.to_numeric(ep_panel['ep_circ'], errors='coerce')
circ_wide = ep_panel.pivot(index='paper', columns='ep_year', values='ep_circ')
circ_wide.columns = [f'circ_{y}' for y in circ_wide.columns]
circ_wide = circ_wide.dropna(subset=['circ_1988', 'circ_2004'])
circ_wide['circ_change'] = circ_wide['circ_2004'] - circ_wide['circ_1988']
circ_wide['circ_pct_change'] = circ_wide['circ_change'] / circ_wide['circ_1988'] * 100
print(f'Papers with circ in both 1988 and 2004: {len(circ_wide)}')
print(f'Mean circulation 1988: {circ_wide["circ_1988"].mean():,.0f}')
print(f'Mean circulation 2004: {circ_wide["circ_2004"].mean():,.0f}')
print(f'Mean % change: {circ_wide["circ_pct_change"].mean():.1f}%')
print(f'Median % change: {circ_wide["circ_pct_change"].median():.1f}%')

# ---- Save ----
out_path = 'c:/Users/ymw04/Dropbox/shifting_slant/data/processed/panel/ep_newspaper_panel.parquet'
ep_panel.to_parquet(out_path, index=False)

match_path = 'c:/Users/ymw04/Dropbox/shifting_slant/data/processed/panel/ep_match_table.parquet'
mdf.to_parquet(match_path, index=False)

# Also save chain indicator for easy merge
chain_indicator = own88[['paper', 'ep_ownerpermid', 'is_chain_all']].rename(
    columns={'is_chain_all': 'is_chain'})
chain_path = 'c:/Users/ymw04/Dropbox/shifting_slant/data/processed/panel/ep_chain_indicator.parquet'
chain_indicator.to_parquet(chain_path, index=False)

print(f'\nSaved:')
print(f'  {out_path}')
print(f'  {match_path}')
print(f'  {chain_path}')
