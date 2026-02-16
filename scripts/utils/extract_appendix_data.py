"""
extract_appendix_data.py

Extract sample construction counts and full LASSO phrase lists
for the paper appendix.

Outputs:
  - output/tables/tab_sample_construction.tex
  - output/tables/tab_full_phrases.tex
  - Prints sample counts to stdout
"""

import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

# ==================================================================
# PART 1: Sample Construction Counts
# ==================================================================
print("=" * 60)
print("PART 1: Sample Construction Counts")
print("=" * 60)

# Step 1: Initial universe — count from preprocessing metadata
preproc_dir = BASE_DIR / "data" / "processed" / "newspapers"

# Try to count from raw article data
raw_dir = BASE_DIR / "data" / "raw" / "newspapers"
for p in sorted(raw_dir.glob("*.parquet")):
    print(f"  Raw file: {p.name}")

# Load the labeled articles (after preprocessing steps 01-04)
labeled_path = preproc_dir / "04_labeled_articles.parquet"
if labeled_path.exists():
    print(f"\nLoading {labeled_path.name} ...")
    labeled = pd.read_parquet(labeled_path, columns=["newspaper_clean", "year", "is_news", "economy_score"])
    print(f"  Total articles (after preprocessing): {len(labeled):,}")
    print(f"  Unique newspapers: {labeled['newspaper_clean'].nunique()}")
    news_only = labeled[labeled["is_news"] == True] if "is_news" in labeled.columns else labeled
    print(f"  News articles only: {len(news_only):,}")
else:
    print(f"  {labeled_path} not found, trying alternatives...")
    # Try other files
    for p in sorted(preproc_dir.glob("*.parquet")):
        print(f"  Processed file: {p.name}")

# Load the geo-linked panel (after step 11)
geo_panel_path = preproc_dir / "11_newspaper_year_panel_geo.parquet"
if geo_panel_path.exists():
    print(f"\nLoading {geo_panel_path.name} ...")
    geo_panel = pd.read_parquet(geo_panel_path)
    print(f"  Shape: {geo_panel.shape}")
    print(f"  Columns: {list(geo_panel.columns)}")
    print(f"  Unique newspapers: {geo_panel['newspaper_clean'].nunique() if 'newspaper_clean' in geo_panel.columns else 'N/A'}")
    # Check for CZ column
    cz_col = [c for c in geo_panel.columns if 'cz' in c.lower() or 'czone' in c.lower()]
    print(f"  CZ columns: {cz_col}")
    if cz_col:
        print(f"  Unique CZs: {geo_panel[cz_col[0]].nunique()}")
    print(f"  Year range: {geo_panel['year'].min()}-{geo_panel['year'].max()}" if 'year' in geo_panel.columns else "")
    n_papers_geo = geo_panel['newspaper_clean'].nunique() if 'newspaper_clean' in geo_panel.columns else None
else:
    print(f"  {geo_panel_path} not found")

# Load the final regression panel (step 14)
panel_path = BASE_DIR / "data" / "processed" / "panel" / "14_regression_panel.parquet"
if panel_path.exists():
    print(f"\nLoading {panel_path.name} ...")
    panel = pd.read_parquet(panel_path)
    print(f"  Shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)}")

    # Find the right column names
    paper_col = [c for c in panel.columns if 'paper' in c.lower() or 'newspaper' in c.lower()][0]
    cz_col = [c for c in panel.columns if 'cz' in c.lower() or 'czone' in c.lower()][0]
    year_col = 'year'

    print(f"\n  Final panel statistics:")
    print(f"    Paper-year observations: {len(panel):,}")
    print(f"    Unique newspapers ({paper_col}): {panel[paper_col].nunique()}")
    print(f"    Unique CZs ({cz_col}): {panel[cz_col].nunique()}")
    print(f"    Year range: {panel[year_col].min()}-{panel[year_col].max()}")
    print(f"    Years: {sorted(panel[year_col].unique())}")

    n_papers_final = panel[paper_col].nunique()
    n_cz_final = panel[cz_col].nunique()
    n_obs_final = len(panel)
    n_years = panel[year_col].nunique()
else:
    print(f"  {panel_path} not found")

# Also check the newspaper-year panel before geo merge (step 10)
step10_dir = preproc_dir
for p in sorted(step10_dir.glob("10_*.parquet")):
    print(f"\nStep 10 file: {p.name}")
    df10 = pd.read_parquet(p)
    print(f"  Shape: {df10.shape}")
    ncol = [c for c in df10.columns if 'newspaper' in c.lower() or 'paper' in c.lower()]
    if ncol:
        print(f"  Unique newspapers: {df10[ncol[0]].nunique()}")

# ==================================================================
# PART 2: Full LASSO Phrase Lists
# ==================================================================
print("\n" + "=" * 60)
print("PART 2: LASSO Phrase Extraction")
print("=" * 60)

# Load vectorizer to get feature names
vec_path = BASE_DIR / "data" / "processed" / "speeches" / "05_feature_vectorizer.joblib"
print(f"Loading vectorizer from {vec_path.name} ...")
vectorizer = joblib.load(vec_path)
all_feature_names = vectorizer.get_feature_names_out()
print(f"  Full features: {len(all_feature_names):,}")

# Map through intersection columns (models operate in intersection space)
MODEL_DIR = BASE_DIR / "data" / "processed" / "speeches" / "models"
intersection_path = MODEL_DIR / "06_intersection_cols.npy"
if intersection_path.exists():
    intersection_cols = np.load(intersection_path)
    # Some indices may exceed vectorizer vocab (newspaper-only features);
    # map valid ones, label others as "UNK"
    n_vocab = len(all_feature_names)
    feature_names = np.array([
        all_feature_names[i] if i < n_vocab else f"feat_{i}"
        for i in intersection_cols
    ])
    print(f"  Intersection features: {len(feature_names):,} "
          f"({(intersection_cols < n_vocab).sum():,} named)")
else:
    feature_names = all_feature_names

# Select representative windows
windows_to_extract = [
    ("99_100", "99+100 (1985--88)"),
    ("103_104", "103+104 (1993--96)"),
]

all_phrases = []

for window_file, window_label in windows_to_extract:
    model_path = MODEL_DIR / f"06_lasso_window_{window_file}.joblib"
    print(f"\nLoading {model_path.name} ...")
    model = joblib.load(model_path)

    coefs = model.coef_[0]
    nonzero_mask = coefs != 0
    n_nonzero = nonzero_mask.sum()
    n_pos = (coefs > 0).sum()
    n_neg = (coefs < 0).sum()
    print(f"  Non-zero: {n_nonzero} ({n_pos} R, {n_neg} D)")

    # Extract all non-zero phrases with coefficients
    nonzero_idx = np.where(nonzero_mask)[0]
    for idx in nonzero_idx:
        all_phrases.append({
            "window": window_label,
            "phrase": feature_names[idx],
            "coefficient": coefs[idx],
            "party": "Republican" if coefs[idx] > 0 else "Democrat",
        })

phrases_df = pd.DataFrame(all_phrases)
phrases_df["abs_coef"] = phrases_df["coefficient"].abs()
phrases_df = phrases_df.sort_values(["window", "party", "abs_coef"], ascending=[True, True, False])

# Save full CSV
csv_path = BASE_DIR / "output" / "tables" / "full_lasso_phrases.csv"
phrases_df.to_csv(csv_path, index=False)
print(f"\nSaved full phrase list -> {csv_path}")

# ==================================================================
# PART 3: Generate LaTeX Tables
# ==================================================================
print("\n" + "=" * 60)
print("PART 3: Generating LaTeX Tables")
print("=" * 60)

# --- Table: Full Phrases (top 30 per party per window) ---
tex_lines = []
tex_lines.append(r"\begin{table}[H]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Top 30 Partisan Phrases by Rolling Window}")
tex_lines.append(r"\label{tab:full_phrases}")
tex_lines.append(r"\footnotesize")

for window_file, window_label in windows_to_extract:
    wdf = phrases_df[phrases_df["window"] == window_label].copy()
    rep = wdf[wdf["party"] == "Republican"].head(30).reset_index(drop=True)
    dem = wdf[wdf["party"] == "Democrat"].head(30).reset_index(drop=True)

    tex_lines.append(r"\vspace{6pt}")
    tex_lines.append(r"\begin{threeparttable}")
    tex_lines.append(r"\begin{tabular}{rlcrl}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"\multicolumn{2}{c}{Republican Phrases} & & \multicolumn{2}{c}{Democratic Phrases} \\")
    tex_lines.append(r"\cmidrule{1-2} \cmidrule{4-5}")
    tex_lines.append(r"Coef. & Phrase & & Coef. & Phrase \\")
    tex_lines.append(r"\midrule")
    tex_lines.append(r"\multicolumn{5}{c}{\textit{Window: " + window_label + r"}} \\[2pt]")

    n_rows = max(len(rep), len(dem))
    for i in range(min(n_rows, 30)):
        r_coef = f"{rep.iloc[i]['coefficient']:.3f}" if i < len(rep) else ""
        r_phrase = rep.iloc[i]["phrase"].replace("_", r"\_") if i < len(rep) else ""
        d_coef = f"{dem.iloc[i]['coefficient']:.3f}" if i < len(dem) else ""
        d_phrase = dem.iloc[i]["phrase"].replace("_", r"\_") if i < len(dem) else ""
        tex_lines.append(f"{r_coef} & {r_phrase} & & {d_coef} & {d_phrase} \\\\")

    # Summary line
    n_rep_total = len(wdf[wdf["party"] == "Republican"])
    n_dem_total = len(wdf[wdf["party"] == "Democrat"])
    tex_lines.append(r"\midrule")
    tex_lines.append(f"\\multicolumn{{5}}{{c}}{{Total non-zero: {n_rep_total} Republican, {n_dem_total} Democratic}} \\\\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{threeparttable}")
    tex_lines.append(r"\vspace{6pt}")

tex_lines.append(r"\begin{minipage}{0.95\textwidth}")
tex_lines.append(r"\footnotesize\textit{Notes:} Each panel shows the 30 phrases with the largest absolute LASSO coefficients for each party in the indicated rolling window. Positive coefficients indicate Republican-associated language; negative coefficients indicate Democratic-associated language. Phrases are Porter-stemmed unigrams or bigrams.")
tex_lines.append(r"\end{minipage}")
tex_lines.append(r"\end{table}")

tex_out = BASE_DIR / "draft" / "tab_full_phrases.tex"
with open(tex_out, "w") as f:
    f.write("\n".join(tex_lines))
print(f"Saved LaTeX phrase table -> {tex_out}")

print("\nDone!")
