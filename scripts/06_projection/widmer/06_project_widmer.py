# -------------------------------------------------------------------------
# FILE: 06_project_fixed_ruler.py
# DESC: Project a FIXED reference model (Congress 103) onto newspaper data.
#       This version is optimized for PyCharm 'Run' without CLI arguments.
#       [MODIFICATIONS]
#       - Added internal loop for target congresses (99-110).
#       - Removed mandatory argparse requirements for easier IDE execution.
#       - Uses model from Congress 103 as the 'Fixed Ruler'.
# -------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import re
from pathlib import Path

# --------------------------------------------------
# Settings for IDE Execution (Fixed values)
# --------------------------------------------------

# Set the congresses you want to analyze here
TARGET_CONGRESSES = range(99, 111)  # 99, 100, ..., 110
MODEL_CONGRESS = 103  # The "Fixed Ruler" (NAFTA Period)

# --------------------------------------------------
# Trade Keywords (Stemmed Forms for Vocab Matching)
# --------------------------------------------------

TRADE_STEM_PATTERNS = [
    r"nafta", r"gatt", r"wto", r"uruguay.?round", r"fast.?track",
    r"maquiladora", r"cusfta", r"tariff", r"protect",
    r"custom.?duti", r"import.?quota", r"export.?subsidi",
    r"trade.?barrier", r"free.?trad", r"foreign.?trad",
    r"intern.*?trad", r"global.?trad", r"trade.?agreem",
    r"trade.?polici", r"trade.?deficit", r"trade.?surplus",
    r"trade.?negot"
]
TRADE_REGEX = re.compile(r'\b(' + '|'.join(TRADE_STEM_PATTERNS) + r')', re.IGNORECASE)

SPORTS_REGEX = re.compile(
    r"(baseball|nba|nfl|player|pitcher|quarterback|laker|yankee|red.?sox|coach|roster)",
    re.IGNORECASE
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

if "SHIFTING_SLANT_DIR" not in os.environ:
    # If environment variable is not set in PyCharm, manually set it here
    os.environ["SHIFTING_SLANT_DIR"] = r"C:\Users\ymw04\Dropbox\shifting_slant"

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

NEWS_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"
VOCAB_DIR = BASE_DIR / "data" / "processed" / "shared_vocab"
MODEL_DIR = BASE_DIR / "data" / "processed" / "models" / "widmer"
OUT_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "slant_fixed"

# Load Reference Model (e.g., Congress 103) once
PHI_PATH = MODEL_DIR / f"phi_congress_{MODEL_CONGRESS}.npy"
INTERCEPT_PATH = MODEL_DIR / f"intercept_congress_{MODEL_CONGRESS}.npy"
VOCAB_MODEL_PATH = VOCAB_DIR / f"vocab_shared_congress_{MODEL_CONGRESS}.csv"


# --------------------------------------------------
# Processing Function
# --------------------------------------------------

def run_projection(data_congress):
    """
    Performs slant projection for a specific congress using the fixed ruler.
    """
    print(f"\n--- Projecting Fixed Ruler ({MODEL_CONGRESS}) onto Congress {data_congress} ---")

    # Data paths
    x_data_path = NEWS_DIR / f"X_newspapers_congress_{data_congress}.npz"
    meta_data_path = NEWS_DIR / f"meta_newspapers_congress_{data_congress}.parquet"
    vocab_data_path = NEWS_DIR / f"vocab_newspapers_congress_{data_congress}.csv"
    out_file = OUT_DIR / f"news_slant_ref{MODEL_CONGRESS}_on_data{data_congress}.parquet"

    # 1. Load Data
    if not x_data_path.exists():
        print(f"    [SKIP] Data for Congress {data_congress} not found.")
        return

    x_data = sp.load_npz(x_data_path)
    meta = pd.read_parquet(meta_data_path)
    vocab_data = pd.read_csv(vocab_data_path)["term"].astype(str).tolist()
    data_term_to_idx = {term: idx for idx, term in enumerate(vocab_data)}

    # 2. Load Model & Align
    phi = np.load(PHI_PATH)
    intercept = float(np.load(INTERCEPT_PATH))
    model_terms = pd.read_csv(VOCAB_MODEL_PATH)["bigram"].astype(str).tolist()

    data_indices = []
    model_indices = []
    for m_idx, term in enumerate(model_terms):
        if term in data_term_to_idx:
            data_indices.append(data_term_to_idx[term])
            model_indices.append(m_idx)

    # 3. Calculation
    if data_indices:
        x_matched = x_data[:, data_indices]
        phi_matched = phi[model_indices]
        slant = x_matched.dot(phi_matched) + intercept
        used_terms = np.asarray(x_matched.sum(axis=1)).ravel()
    else:
        slant = np.full(x_data.shape[0], intercept)
        used_terms = np.zeros(x_data.shape[0])

    # 4. Trade Classification
    trade_term_indices = [i for i, t in enumerate(vocab_data)
                          if TRADE_REGEX.search(t) and not SPORTS_REGEX.search(t)]

    if trade_term_indices:
        trade_vec = np.zeros((len(vocab_data), 1))
        trade_vec[trade_term_indices] = 1
        is_trade = (x_data.dot(trade_vec).ravel() > 0).astype(int)
    else:
        is_trade = np.zeros(len(meta), dtype=int)

    # 5. Save
    meta = meta.copy()
    meta["slant"] = slant
    meta["used_terms"] = used_terms
    meta["is_trade"] = is_trade
    meta["ref_congress"] = MODEL_CONGRESS

    meta.to_parquet(out_file)
    print(f"    - Saved: {out_file.name} | Trade articles: {is_trade.sum():,}")


# --------------------------------------------------
# Main Execution Loop
# --------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch projection using Fixed Ruler: Congress {MODEL_CONGRESS}")

    for congress in TARGET_CONGRESSES:
        run_projection(congress)

    print("\n>>> All tasks completed successfully.")


if __name__ == "__main__":
    main()