"""Generate compact 2-column newspaper list for the paper."""
import pandas as pd

panel = pd.read_parquet('data/processed/runs/exp_shvocab_cv/panel/14_regression_panel.parquet')
# Drop papers with NaN vulnerability (not in estimation sample)
panel = panel.dropna(subset=['vulnerability1990_scaled'])
newspapers = panel.groupby('paper').agg(
    state=('state', 'first'),
    articles_per_year=('n_articles', 'mean')
).reset_index()
newspapers = newspapers.sort_values(['state', 'paper']).reset_index(drop=True)

# Manual display name mapping
manual = {
    'SACRAMENTO BEE': 'Sacramento Bee',
    'THE SAN FRANCISCO CHRONICLE': 'San Francisco Chronicle',
    'THE SEATTLE TIMES': 'Seattle Times',
    'The Gazette (Cedar Rapids-Iowa City)': 'Gazette (Cedar Rapids)',
    'Gazette, The (Colorado Springs, CO)': 'Gazette (Colorado Springs)',
    'Nuevo Herald, El (Miami, FL)': 'Nuevo Herald',
    'Herald & Review (Decatur, IL)': 'Herald \\& Review (Decatur)',
    'Journal Gazette (Mattoon, IL)': 'Journal Gazette (Mattoon)',
    'Journal Gazette, The (Fort Wayne, IN)': 'Journal Gazette (Fort Wayne)',
    'Times, The (Munster, IN)': 'Times (Munster)',
    'Pantagraph, The (Bloomington, IL)': 'Pantagraph (Bloomington)',
    'Herald-Times (Bloomington, IN)': 'Herald-Times (Bloomington)',
    'Post Register (Idaho Falls, ID)': 'Post Register (Idaho Falls)',
    'Delaware State News (Dover, DE)': 'Delaware State News (Dover)',
    'The Advocate (Baton Rouge, La.)': 'Advocate (Baton Rouge)',
    'Sunday Republican (Springfield, MA)': 'Sunday Republican (Springfield)',
    'Union-News (Springfield, MA)': 'Union-News (Springfield)',
    'Star Tribune: Newspaper of the Twin Cities': 'Star Tribune (Twin Cities)',
    'The Atlanta Journal and The Atlanta Constitution': 'Atlanta Journal-Constitution',
    'Daily Pennsylvanian, The: University of Pennsylvania (Philadelphia, PA)': 'Daily Pennsylvanian',
    'New Hampshire Union Leader (Manchester, NH)': 'New Hampshire Union Leader',
    'Hunterdon County Democrat (Flemington, NJ)': 'Hunterdon County Democrat',
    'Herald-Journal (Spartanburg, SC)': 'Herald-Journal (Spartanburg)',
    'Post-Standard, The (Syracuse, NY)': 'Post-Standard',
    'Times Union, The (Albany, NY)': 'Times Union',
    'Morning Call, The (Allentown, PA)': 'Morning Call',
    'Patriot-News, The (Harrisburg, PA)': 'Patriot-News',
    'Times Leader, The (Wilkes Barre, PA)': 'Times Leader',
    'Intelligencer Journal (Lancaster, PA)': 'Intelligencer Journal (Lancaster)',
    'Lancaster New Era (PA)': 'Lancaster New Era',
    'Sunday News (Lancaster, PA)': 'Sunday News (Lancaster)',
    'Deseret News, The (Salt Lake City, UT)': 'Deseret News',
    'Salt Lake Tribune, The (UT)': 'Salt Lake Tribune',
    'Oregonian, The (Portland, OR)': 'Oregonian',
    'Wenatchee World, The (WA)': 'Wenatchee World',
    'Capital Times, The (Madison, WI)': 'Capital Times',
    'Journal Times, The (Racine, WI)': 'Journal Times',
    'Wisconsin State Journal (Madison, WI)': 'Wisconsin State Journal',
    'Greensboro News & Record': 'Greensboro News \\& Record',
    'The News & Observer': 'News \\& Observer',
    'Worcester Telegram & Gazette (MA)': 'Worcester Telegram \\& Gazette',
    'State, The (Columbia, SC)': 'State (Columbia)',
    'Plain Dealer, The (Cleveland, OH)': 'Plain Dealer',
    'Press of Atlantic City, The (NJ)': 'Press of Atlantic City',
    'The Record (New Jersey)': 'Record (New Jersey)',
    'Daily Breeze (Torrance, CA)': 'Daily Breeze (Torrance)',
    'Daily News of Los Angeles (CA)': 'Daily News of Los Angeles',
    'Press-Enterprise, The (Riverside, CA)': 'Press-Enterprise',
    'Press-Telegram (Long Beach, CA)': 'Press-Telegram (Long Beach)',
    'Redding Record Searchlight (CA)': 'Redding Record Searchlight',
    'San Diego Union-Tribune, The (CA)': 'San Diego Union-Tribune',
    'The Orange County Register': 'Orange County Register',
    'Anchorage Daily News (AK)': 'Anchorage Daily News',
    'Anniston Star, The (AL)': 'Anniston Star',
    'Huntsville Times, The (AL)': 'Huntsville Times',
    'Mobile Register (AL)': 'Mobile Register',
    'The Arizona Daily Star': 'Arizona Daily Star',
    'Fresno Bee, The (CA)': 'Fresno Bee',
    'Modesto Bee, The (CA)': 'Modesto Bee',
    'Pueblo Chieftain, The (CO)': 'Pueblo Chieftain',
    'The Denver Post': 'Denver Post',
    'The Hartford Courant': 'Hartford Courant',
    'The Washington Times': 'Washington Times',
    'Bradenton Herald, The (FL)': 'Bradenton Herald',
    'Fort Pierce Tribune (FL)': 'Fort Pierce Tribune',
    'Miami Herald, The (FL)': 'Miami Herald',
    'Ocala Star-Banner (FL)': 'Ocala Star-Banner',
    'The Palm Beach Post': 'Palm Beach Post',
    'The Tampa Tribune': 'Tampa Tribune',
    'Augusta Chronicle, The (GA)': 'Augusta Chronicle',
    'Lewiston Morning Tribune (ID)': 'Lewiston Morning Tribune',
    'State Journal-Register, The (Springfield, IL)': 'State Journal-Register',
    'Times-Courier (Charleston, IL)': 'Times-Courier (Charleston)',
    'Evansville Courier, The (IN)': 'Evansville Courier',
    'News-Sentinel, The (Fort Wayne, IN)': 'News-Sentinel',
    'Post-Tribune (IN)': 'Post-Tribune',
    'Journal-World (Lawrence, KS)': 'Journal-World (Lawrence)',
    'Wichita Eagle, The (KS)': 'Wichita Eagle',
    'Lexington Herald-Leader (KY)': 'Lexington Herald-Leader',
    'Owensboro Messenger-Inquirer (KY)': 'Owensboro Messenger-Inquirer',
    'The Kentucky Post': 'Kentucky Post',
    'Times-Picayune, The (New Orleans, LA)': 'Times-Picayune',
    'Boston Herald (MA)': 'Boston Herald',
    'Capital, The (Annapolis, MD)': 'Capital (Annapolis)',
    'Sun, The (Baltimore, MD)': 'Sun (Baltimore)',
    'Bangor Daily News (ME)': 'Bangor Daily News',
    'St. Paul Pioneer Press (MN)': 'St.\\ Paul Pioneer Press',
    'St. Louis Post-Dispatch': 'St.\\ Louis Post-Dispatch',
    'The Kansas City Star': 'Kansas City Star',
    'Charlotte Observer, The (NC)': 'Charlotte Observer',
    'Fayetteville Observer, The (NC)': 'Fayetteville Observer',
    'Grand Forks Herald (ND)': 'Grand Forks Herald',
    'Omaha World-Herald (NE)': 'Omaha World-Herald',
    'Star-Ledger, The (Newark, NJ)': 'Star-Ledger',
    'The Buffalo News': 'Buffalo News',
    'Staten Island Advance (NY)': 'Staten Island Advance',
    'Syracuse Herald American (NY)': 'Syracuse Herald American',
    'Syracuse Herald-Journal (NY)': 'Syracuse Herald-Journal',
    'Watertown Daily Times (NY)': 'Watertown Daily Times',
    'Akron Beacon Journal (OH)': 'Akron Beacon Journal',
    'Columbus Dispatch, The (OH)': 'Columbus Dispatch',
    'Dayton Daily News (OH)': 'Dayton Daily News',
    'The Cincinnati Post': 'Cincinnati Post',
    'Lawton Constitution, The (OK)': 'Lawton Constitution',
    'The Daily Oklahoman': 'Daily Oklahoman',
    'Philadelphia Daily News (PA)': 'Philadelphia Daily News',
    'Philadelphia Inquirer, The (PA)': 'Philadelphia Inquirer',
    'Pittsburgh Post-Gazette (PA)': 'Pittsburgh Post-Gazette',
    'York Daily Record (PA)': 'York Daily Record',
    'Providence Journal (RI)': 'Providence Journal',
    'Knoxville News-Sentinel, The (TN)': 'Knoxville News-Sentinel',
    'The Commercial Appeal': 'Commercial Appeal',
    'Austin American-Statesman': 'Austin American-Statesman',
    'Fort Worth Star-Telegram': 'Fort Worth Star-Telegram',
    'Houston Chronicle': 'Houston Chronicle',
    'San Antonio Express-News': 'San Antonio Express-News',
    'The Dallas Morning News': 'Dallas Morning News',
    'Richmond Times-Dispatch': 'Richmond Times-Dispatch',
    'Roanoke Times, The (VA)': 'Roanoke Times',
    'Virginian-Pilot, The (Norfolk, VA)': 'Virginian-Pilot',
    'Seattle Post-Intelligencer': 'Seattle Post-Intelligencer',
    'The News Tribune': 'News Tribune',
    'La Crosse Tribune (WI)': 'La Crosse Tribune',
    'Milwaukee Sentinel': 'Milwaukee Sentinel',
    'The Milwaukee Journal': 'Milwaukee Journal',
    'Chicago Sun-Times': 'Chicago Sun-Times',
    'Journal Star (Peoria, IL)': 'Journal Star (Peoria)',
    'New Haven Register (CT)': 'New Haven Register',
    'St. Petersburg Times': 'St.\\ Petersburg Times',
    'Tulsa World': 'Tulsa World',
    'Rocky Mountain News (CO)': 'Rocky Mountain News',
    'San Jose Mercury News (CA)': 'San Jose Mercury News',
}

rows = []
for i, r in newspapers.iterrows():
    raw = r['paper']
    name = manual.get(raw, raw)
    art = int(round(r['articles_per_year']))
    rows.append((i + 1, name, r['state'], f'{art:,}'))

n = len(rows)
n_cz = panel['cz'].nunique()

# Split into two tables: papers 1-66 and 67-131
# Each table uses 2-column layout internally
split = 66  # first table has papers 1-66
part1 = rows[:split]
part2 = rows[split:]

def make_lines(part_rows):
    """Generate 2-column table rows from a list of (num, name, state, art) tuples."""
    half = (len(part_rows) + 1) // 2
    lines = []
    for j in range(half):
        left = part_rows[j]
        if j + half < len(part_rows):
            right = part_rows[j + half]
            lines.append(
                f'        {left[0]} & {left[1]} & {left[2]} & {left[3]}'
                f' & {right[0]} & {right[1]} & {right[2]} & {right[3]} \\\\'
            )
        else:
            lines.append(
                f'        {left[0]} & {left[1]} & {left[2]} & {left[3]}'
                f' & & & & \\\\'
            )
    return lines

TAB_SPEC = (
    '    \\centering\n'
    '    \\singlespacing\n'
    '    \\scriptsize\n'
    '    \\renewcommand{\\arraystretch}{1.05}\n'
    '    \\begin{tabular}{@{} r l l r @{\\hskip 14pt} r l l r @{}}\n'
    '        \\toprule\n'
    '        & Newspaper & St & Art/Yr & & Newspaper & St & Art/Yr \\\\\n'
    '        \\midrule\n'
)

lines1 = make_lines(part1)
lines2 = make_lines(part2)

table = (
    '% Compact 2-column newspaper list (split across two pages)\n'
    '\\begin{table}[H]\n'
    '    \\begin{threeparttable}\n'
    '    \\caption{Newspapers in the Estimation Sample}\n'
    '    \\label{tab:newspaper_list}\n'
    + TAB_SPEC
    + '\n'.join(lines1) + '\n'
    '        \\bottomrule\n'
    '    \\end{tabular}\n'
    '    \\vspace{4pt}\n'
    '    \\begin{tablenotes}[flushleft]\n'
    '        \\setlength{\\labelsep}{0pt}\n'
    '        \\scriptsize\n'
    f'        \\item \\textit{{Notes:}} The {n} daily newspapers in the estimation sample, sorted alphabetically by state.'
    f' Art/Yr is the mean number of articles per year over each newspaper\'s sample period.'
    f' The newspapers are distributed across {n_cz} commuting zones.'
    ' \\emph{Continued on next page.}\n'
    '    \\end{tablenotes}\n'
    '    \\end{threeparttable}\n'
    '\\end{table}\n'
    '\n'
    '\\begin{table}[H]\n'
    '    \\begin{threeparttable}\n'
    '    \\caption{Newspapers in the Estimation Sample (continued)}\n'
    + TAB_SPEC
    + '\n'.join(lines2) + '\n'
    '        \\bottomrule\n'
    '    \\end{tabular}\n'
    '    \\end{threeparttable}\n'
    '\\end{table}\n'
)

with open('draft/tab_newspaper_list.tex', 'w') as f:
    f.write(table)

print(f'Written {n} newspapers split into 2 tables ({len(lines1)} + {len(lines2)} rows)')
print(f'Across {n_cz} CZs')
