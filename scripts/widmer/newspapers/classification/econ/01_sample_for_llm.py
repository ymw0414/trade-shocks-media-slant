# -------------------------------------------------------------------------
# FILE: 01_sample_for_llm.py
# DESCRIPTION:
#   - Fast random sampling for LLM labeling
#   - Samples while reading, body text only
# -------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import numpy as np

START_YEAR = 1986
END_YEAR = 1989
SAMPLES_PER_YEAR = 250
RANDOM_SEED = 42

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
CLEAN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"
OUT_DIR = BASE_DIR / "scripts" / "newspapers" / "classification" / "econ"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "01_llm_annotation_sample.csv"

rng = np.random.default_rng(RANDOM_SEED)

def main():

    collected = {y: [] for y in range(START_YEAR, END_YEAR + 1)}
    files = sorted(CLEAN_DIR.glob("newspapers_congress_*_clean.parquet"))

    print("Sampling articles on the fly (body text only)...")

    for f in files:
        df = pd.read_parquet(
            f,
            columns=["date", "clean_text"]
        )

        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

        for year in range(START_YEAR, END_YEAR + 1):
            need = SAMPLES_PER_YEAR - len(collected[year])
            if need <= 0:
                continue

            df_y = df[df["year"] == year]
            if df_y.empty:
                continue

            take = min(need, len(df_y))
            sampled = df_y.sample(
                n=take,
                random_state=int(rng.integers(1e9))
            )

            collected[year].append(sampled)

        if all(len(collected[y]) >= SAMPLES_PER_YEAR for y in collected):
            break

    final_samples = []
    for year in collected:
        final_samples.append(
            pd.concat(collected[year]).sample(
                n=SAMPLES_PER_YEAR,
                random_state=RANDOM_SEED
            )
        )

    sample_df = pd.concat(final_samples).reset_index(drop=True)
    sample_df.insert(0, "article_id", sample_df.index)

    out = sample_df[["article_id", "year", "clean_text"]]
    out.to_csv(OUT_FILE, index=False)

    print(f"Saved sample: {OUT_FILE}")
    print(f"Total samples: {len(out)}")

if __name__ == "__main__":
    main()
