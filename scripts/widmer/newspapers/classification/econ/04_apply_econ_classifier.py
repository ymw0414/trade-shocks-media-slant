# -------------------------------------------------------------------------
# FILE: 04_apply_econ_classifier.py
# DESCRIPTION:
#   - Apply trained econ classifier to all newspaper articles (1986–2004)
#   - Output continuous economic intensity measure p_econ
# -------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import joblib
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

CLEAN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"
ECON_DIR  = BASE_DIR / "scripts" / "newspapers" / "classification" / "econ"
OUT_DIR   = BASE_DIR / "data" / "processed" / "newspapers" / "classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = ECON_DIR / "econ_classifier.joblib"
VECT_FILE  = ECON_DIR / "econ_vectorizer.joblib"

OUT_FILE = OUT_DIR / "econ_scores_all_articles.parquet"

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("=" * 70)
    print("04_apply_econ_classifier.py")
    print("Applying econ classifier to all articles")
    print("=" * 70)

    print("Loading classifier and vectorizer...")
    clf = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)

    # --------------------------------------------------
    # Load all cleaned newspaper files
    # --------------------------------------------------
    files = sorted(CLEAN_DIR.glob("newspapers_congress_*_clean.parquet"))

    print(f"Found {len(files)} parquet files")

    results = []

    for f in tqdm(files, desc="Scoring files"):
        df = pd.read_parquet(f)

        if "clean_text" not in df.columns:
            raise ValueError(f"clean_text missing in {f.name}")

        texts = df["clean_text"].fillna("")

        X = vectorizer.transform(texts)
        p_econ = clf.predict_proba(X)[:, 1]

        out = df[["date", "paper"]].copy()
        out["p_econ"] = p_econ

        results.append(out)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_parquet(OUT_FILE)

    print("-" * 70)
    print(f"Saved econ scores: {OUT_FILE}")
    print(f"Total articles scored: {len(final_df):,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
