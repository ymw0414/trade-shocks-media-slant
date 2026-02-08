"""
02_standardize_paper_names.py

Derive the analysis sample of newspapers from scratch using raw data
(1987-2004).  Scans all yearly parquets produced by 01, standardizes
paper names to canonical forms, and selects the analysis sample.

Algorithm
---------
Step 1  Scan yearly parquets -> unique (raw_name, year, n_articles).
Step 2  Parse each raw name into (core_name, city, state).
Step 3  Normalize core name and group variants by (normalized_core, state).
        * state=None + 1 matching state group  -> merge
        * state=None + 0 matching state groups -> keep as own group
        * state=None + >1 matching state groups -> DROP (ambiguous)
Step 4  Pick canonical name per group (most-frequent raw variant).
Step 5  Mergers stay separate (different normalized cores).
Step 6  Selection filter:
        * >=100 articles/year in EACH of 1992-1995
        * Exclude wire services and non-newspaper sources

Inputs
------
  data/intermediate/newspapers/yearly/newspapers_{year}.parquet

Outputs
-------
  data/temp/minwoo/02_paper_crosswalk.csv   raw_name -> canonical_name
  data/temp/minwoo/02_target_papers.csv     final analysis sample
"""

import os
import re
import pandas as pd
from pathlib import Path

# ==================================================================
# Paths
# ==================================================================
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
INPUT_DIR = BASE_DIR / "data" / "intermediate" / "newspapers" / "yearly"
OUT_DIR = BASE_DIR / "data" / "temp" / "minwoo"

# ==================================================================
# Constants
# ==================================================================
YEARS = range(1987, 2005)
STABILITY_YEARS = [1992, 1993, 1994, 1995]
MIN_ARTICLES_PER_YEAR = 100

# Wire services / non-newspaper sources to exclude
EXCLUDE_PATTERNS = [
    "Associated Press", "Reuters", "United Press International", "UPI",
    "Agence France", "Xinhua", "TASS", "PR Newswire", "Business Wire",
    "States News Service", "Copley News Service", "Gannett News Service",
    "Knight Ridder", "Newhouse News", "Scripps Howard", "Cox News Service",
    "Tribune News Service", "McClatchy", "Dow Jones",
    "Federal Document", "Federal Information",
    "Congressional Quarterly", "National Journal", "Roll Call", "The Hotline",
]
EXCLUDE_KEYWORDS = ["wire", "newswire", "news service"]

# US state abbreviations
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

# Full state names -> abbreviation (for parentheticals like "(New Jersey)")
STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# Manual state overrides for names the parser can't resolve automatically
MANUAL_STATE = {
    "The Gazette (Cedar Rapids-Iowa City)": "IA",
}

# Non-newspaper sources that pass the article-count filter
EXCLUDE_NON_NEWSPAPER = [
    "All Things Considered",   # NPR radio
    "Weekend Edition",         # NPR radio
    "Morning Edition",         # NPR radio
    "Newsweek",
    "New Scientist",
    "Governing Magazine",
    "Science News",
    "Reason (USA)",
    "Catalyst Chicago",
    "Chicago Reporter",
]

# ==================================================================
# Parsing helpers
# ==================================================================
PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
ARTICLE_SUFFIX_RE = re.compile(r"^(.+?),\s*(The|El)\b(.*)$", re.IGNORECASE)


def _clean_abbrev(s):
    """Strip periods from abbreviations: 'N.J.' -> 'NJ', 'PA.' -> 'PA'."""
    return s.replace(".", "").strip()


def parse_location(loc_str):
    """Parse '(TX)', '(Minneapolis, MN)', '(New Jersey)', '(N.J.)', etc."""
    loc_str = loc_str.strip()
    parts = [p.strip() for p in loc_str.rsplit(",", 1)]

    if len(parts) == 2:
        candidate = _clean_abbrev(parts[1])
        if candidate.upper() in US_STATES:
            return parts[0], candidate.upper()

    # Bare 2-letter abbreviation: "(TX)", "(N.J.)"
    bare = _clean_abbrev(loc_str)
    if bare.upper() in US_STATES:
        return None, bare.upper()

    # Full state name: "(New Jersey)", "(Oklahoma)"
    if loc_str.lower() in STATE_NAMES:
        return None, STATE_NAMES[loc_str.lower()]

    return loc_str, None          # unrecognized -> city=loc, state=None


def parse_paper_name(raw_name):
    """
    Parse a raw paper name -> (core_name, city, state).

    Handles:
      "Houston Chronicle (TX)"          -> ("Houston Chronicle", None, "TX")
      "Buffalo News, The (NY)"          -> ("The Buffalo News", None, "NY")
      "Star Tribune (Minneapolis, MN)"  -> ("Star Tribune", "Minneapolis", "MN")
      "Nuevo Herald, El (Miami, FL)"    -> ("El Nuevo Herald", "Miami", "FL")
      "The Record (New Jersey)"         -> ("The Record", None, "NJ")
      "Chicago Sun-Times"               -> ("Chicago Sun-Times", None, None)
    """
    # Check manual overrides first
    if raw_name in MANUAL_STATE:
        override_state = MANUAL_STATE[raw_name]
        # Still parse to get core_name, but force state
        name = raw_name.strip()
        m = PAREN_RE.match(name)
        if m:
            name = m.group(1).strip()
        m2 = ARTICLE_SUFFIX_RE.match(name)
        if m2:
            name = m2.group(2).title() + " " + m2.group(1).strip()
        return name, None, override_state

    name = raw_name.strip()
    city, state = None, None

    # 1. Extract trailing parenthetical location
    m = PAREN_RE.match(name)
    if m:
        name = m.group(1).strip()
        city, state = parse_location(m.group(2))

    # 2. Handle "Name, The" / "Name, El" -> "The Name" / "El Name"
    m2 = ARTICLE_SUFFIX_RE.match(name)
    if m2:
        article = m2.group(2).title()     # "The" or "El"
        name = article + " " + m2.group(1).strip()

    return name, city, state


def normalize_core(name):
    """
    Normalize a core name for grouping purposes.

    * Title-case if ALL CAPS (len > 3 to skip acronyms like AP)
    * Strip subtitle after ':'
    * Strip leading 'The '
    * Collapse whitespace
    """
    if name == name.upper() and len(name) > 3:
        name = name.title()

    if ":" in name:
        name = name.split(":")[0].strip()

    if name.lower().startswith("the "):
        name = name[4:]

    return " ".join(name.split())


def is_excluded(canonical_name):
    """True if this name matches a wire service / non-newspaper pattern."""
    low = canonical_name.lower()
    for pat in EXCLUDE_PATTERNS:
        if pat.lower() in low:
            return True
    if any(kw in low for kw in EXCLUDE_KEYWORDS):
        return True
    for pat in EXCLUDE_NON_NEWSPAPER:
        if pat.lower() in low:
            return True
    return False


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Step 1: Scan yearly parquets
    # --------------------------------------------------------------
    print("Step 1: Scanning yearly parquets ...")
    records = []

    for year in YEARS:
        path = INPUT_DIR / f"newspapers_{year}.parquet"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping")
            continue

        counts = pd.read_parquet(path, columns=["paper"])["paper"].value_counts()
        for raw_name, n in counts.items():
            records.append((raw_name, year, int(n)))
        print(f"  {year}: {counts.sum():>10,} articles, {len(counts):>5,} unique names")

    raw_df = pd.DataFrame(records, columns=["raw_name", "year", "n_articles"])
    n_unique_raw = raw_df["raw_name"].nunique()
    print(f"\n  Total unique raw names: {n_unique_raw:,}\n")

    # --------------------------------------------------------------
    # Step 2: Parse each raw name
    # --------------------------------------------------------------
    print("Step 2: Parsing paper names ...")
    unique_names = raw_df["raw_name"].unique()

    parsed_rows = []
    for name in unique_names:
        core, city, state = parse_paper_name(name)
        norm = normalize_core(core)
        parsed_rows.append({
            "raw_name": name,
            "core_name": core,
            "city": city,
            "state": state,
            "normalized_core": norm,
        })

    parsed_df = pd.DataFrame(parsed_rows)
    n_with = parsed_df["state"].notna().sum()
    n_without = parsed_df["state"].isna().sum()
    print(f"  Parsed {len(parsed_df):,} unique names "
          f"({n_with:,} with state, {n_without:,} without)\n")

    # --------------------------------------------------------------
    # Step 3: Group by (normalized_core, state)
    # --------------------------------------------------------------
    print("Step 3: Grouping variants ...")

    # Attach total article counts (across all years) to parsed names
    name_totals = raw_df.groupby("raw_name")["n_articles"].sum().reset_index()
    parsed_df = parsed_df.merge(name_totals, on="raw_name")

    has_state = parsed_df[parsed_df["state"].notna()]
    no_state = parsed_df[parsed_df["state"].isna()]

    # Build groups from state-known names
    state_groups = {}          # (normalized_core, state) -> [raw_names]
    for _, row in has_state.iterrows():
        key = (row["normalized_core"], row["state"])
        state_groups.setdefault(key, []).append(row["raw_name"])

    # Resolve state-unknown names
    n_merged = n_dropped = n_own = 0
    dropped_info = []

    for _, row in no_state.iterrows():
        norm = row["normalized_core"]
        matches = [k for k in state_groups if k[0] == norm]

        if len(matches) == 1:
            state_groups[matches[0]].append(row["raw_name"])
            n_merged += 1
        elif len(matches) == 0:
            state_groups.setdefault((norm, None), []).append(row["raw_name"])
            n_own += 1
        else:
            n_dropped += 1
            dropped_info.append((
                row["raw_name"], int(row["n_articles"]),
                [k[1] for k in matches],
            ))

    print(f"  State-known groups: "
          f"{sum(1 for k in state_groups if k[1] is not None):,}")
    print(f"  state=None -> merged (unique match): {n_merged}")
    print(f"  state=None -> own group (no match):  {n_own}")
    print(f"  state=None -> DROPPED (ambiguous):   {n_dropped}")

    if dropped_info:
        print("\n  Dropped ambiguous names (top 20 by articles):")
        for name, total, states in sorted(dropped_info, key=lambda x: -x[1])[:20]:
            print(f"    {name!r:50s}  {total:>10,} articles  states={states}")
    print()

    # --------------------------------------------------------------
    # Step 4: Pick canonical name per group
    # --------------------------------------------------------------
    print("Step 4: Picking canonical names ...")

    name_total_map = name_totals.set_index("raw_name")["n_articles"]

    crosswalk_rows = []
    multi_variant_groups = []

    for (norm, state), raw_names in sorted(state_groups.items()):
        # Sort by total article count descending; most-frequent = canonical
        variants = sorted(raw_names, key=lambda n: -name_total_map.get(n, 0))
        canonical = variants[0]

        for raw in variants:
            crosswalk_rows.append({
                "raw_name": raw,
                "canonical_name": canonical,
                "normalized_core": norm,
                "state": state,
            })

        if len(variants) > 1:
            multi_variant_groups.append((canonical, state, variants))

    crosswalk = pd.DataFrame(crosswalk_rows)
    n_groups = crosswalk["canonical_name"].nunique()
    print(f"  Canonical paper groups: {n_groups:,}")

    # Show multi-variant groups for audit
    if multi_variant_groups:
        print(f"\n  Multi-variant groups ({len(multi_variant_groups)}):")
        for canonical, st, variants in sorted(
            multi_variant_groups, key=lambda x: x[0]
        ):
            others = [v for v in variants if v != canonical]
            print(f"    [{st or '??'}] {canonical!r}")
            for o in others:
                print(f"           <- {o!r}")
    print()

    # Save crosswalk
    crosswalk_path = OUT_DIR / "02_paper_crosswalk.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    print(f"  Saved -> {crosswalk_path.name}")

    # --------------------------------------------------------------
    # Step 5: Verify merger entities are separate
    # --------------------------------------------------------------
    print("\nStep 5: Verifying merger entities stay separate ...")
    merger_checks = [
        ("Atlanta Journal", "Atlanta Journal-Constitution"),
        ("Atlanta Constitution", "Atlanta Journal-Constitution"),
        ("Milwaukee Journal", "Milwaukee Journal Sentinel"),
    ]
    for name_a, name_b in merger_checks:
        canon_a = set(
            crosswalk.loc[
                crosswalk["normalized_core"].str.contains(name_a, case=False)
                & ~crosswalk["normalized_core"].str.contains(name_b, case=False),
                "canonical_name",
            ]
        )
        canon_b = set(
            crosswalk.loc[
                crosswalk["normalized_core"].str.contains(name_b, case=False),
                "canonical_name",
            ]
        )
        if canon_a and canon_b:
            overlap = canon_a & canon_b
            if overlap:
                print(f"  WARNING: '{name_a}' and '{name_b}' share IDs: {overlap}")
            else:
                print(f"  OK: '{name_a}' ({canon_a}) != '{name_b}' ({canon_b})")
        else:
            print(f"  Note: '{name_a}' or '{name_b}' not found in data")
    print()

    # --------------------------------------------------------------
    # Step 6: Selection filter
    # --------------------------------------------------------------
    print("Step 6: Applying selection filter ...")

    # Map raw articles -> canonical names (drops ambiguous records)
    raw_to_canon = crosswalk.set_index("raw_name")["canonical_name"]
    article_df = raw_df[raw_df["raw_name"].isin(crosswalk["raw_name"])].copy()
    article_df["canonical_name"] = article_df["raw_name"].map(raw_to_canon)

    # Yearly aggregates per canonical paper
    yearly = (
        article_df
        .groupby(["canonical_name", "year"])["n_articles"]
        .sum()
        .reset_index()
    )

    # Stability check: >=100 in each of 1992-1995
    stab = yearly[yearly["year"].isin(STABILITY_YEARS)]
    stab_pivot = stab.pivot_table(
        index="canonical_name", columns="year",
        values="n_articles", fill_value=0,
    )
    passes = (stab_pivot >= MIN_ARTICLES_PER_YEAR).all(axis=1)
    stable_papers = set(passes[passes].index)
    print(f"  Stable (>={MIN_ARTICLES_PER_YEAR}/yr in {STABILITY_YEARS}): "
          f"{len(stable_papers)}")

    # Exclude wire / non-newspaper
    excluded = {p for p in stable_papers if is_excluded(p)}
    target_papers = sorted(stable_papers - excluded)
    print(f"  Excluded (wire / non-newspaper): {len(excluded)}")
    if excluded:
        for p in sorted(excluded):
            print(f"    - {p}")
    print(f"  Final target papers: {len(target_papers)}\n")

    # Build summary table
    target_yearly = yearly[yearly["canonical_name"].isin(target_papers)]
    canon_state = crosswalk.drop_duplicates("canonical_name").set_index("canonical_name")

    summary = []
    for paper in target_papers:
        pdata = target_yearly[target_yearly["canonical_name"] == paper]
        row = canon_state.loc[paper]
        summary.append({
            "paper": paper,
            "state": row["state"],
            "normalized_core": row["normalized_core"],
            "years_active": int(pdata["year"].nunique()),
            "total_articles": int(pdata["n_articles"].sum()),
            "avg_articles_per_year": int(round(pdata["n_articles"].mean())),
        })

    target_df = pd.DataFrame(summary)
    target_path = OUT_DIR / "02_target_papers.csv"
    target_df.to_csv(target_path, index=False)
    print(f"  Saved -> {target_path.name}\n")

    # Print full target list
    print("-" * 72)
    print(f"{'Paper':<55s} {'State':>5s} {'Yrs':>4s} {'Total':>10s}")
    print("-" * 72)
    for _, r in target_df.iterrows():
        print(f"  {r['paper']:<53s} {str(r['state']):>5s} "
              f"{r['years_active']:>4d} {r['total_articles']:>10,}")
    print("-" * 72)

    # Final summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Unique raw names scanned:  {n_unique_raw:,}")
    print(f"  Canonical paper groups:    {n_groups:,}")
    print(f"  Ambiguous records dropped: {n_dropped}")
    print(f"  Stable papers (filter):    {len(stable_papers)}")
    print(f"  Final target papers:       {len(target_papers)}")
    print(f"\n  Crosswalk:      {crosswalk_path}")
    print(f"  Target papers:  {target_path}")
    print("=" * 72)
