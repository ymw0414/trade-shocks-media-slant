import pandas as pd
import os
from pathlib import Path

BASE = Path(os.environ["SHIFTING_SLANT_DIR"])
FILE = BASE / "data/processed/newspapers/clean/newspapers_congress_110_clean.parquet"

df = pd.read_parquet(FILE)
print(df["clean_text"].str.len().max())
