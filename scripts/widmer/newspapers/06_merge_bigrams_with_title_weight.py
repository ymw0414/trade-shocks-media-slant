# -------------------------------------------------------------------------
# FILE: 06_merge_bigrams_with_title_weight.py
# DESCRIPTION:
#   - Merge body and title bigram matrices
#   - Apply title weight AFTER bigram construction
#   - Final X = X_body + w * X_title
# -------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
import scipy.sparse as sp

# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--congress", type=int, required=True)
parser.add_argument("--title_weight", type=float, default=3.0)
args = parser.parse_args()

CONGRESS = args.congress
W = args.title_weight

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

BIGRAM_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "bigrams"
OUT_DIR    = BASE_DIR / "data" / "processed" / "newspapers" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

X_BODY_FILE  = BIGRAM_DIR / f"X_body_congress_{CONGRESS}.npz"
X_TITLE_FILE = BIGRAM_DIR / f"X_title_congress_{CONGRESS}.npz"

OUT_X = OUT_DIR / f"X_newspapers_congress_{CONGRESS}_titleW{int(W)}.npz"

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    print("=" * 70)
    print("06_merge_bigrams_with_title_weight.py")
    print(f"Congress     : {CONGRESS}")
    print(f"Title weight : {W}")
    print("=" * 70)

    print("Loading bigram matrices...")
    X_body  = sp.load_npz(X_BODY_FILE)
    X_title = sp.load_npz(X_TITLE_FILE)

    if X_body.shape != X_title.shape:
        raise ValueError("Body and title matrices have different shapes")

    print("Applying title weight and merging...")
    X_final = X_body + W * X_title

    print("Saving final matrix...")
    sp.save_npz(OUT_X, X_final)

    print("-" * 70)
    print(f"Final X shape: {X_final.shape}")
    print(f"Saved to: {OUT_X}")
    print("=" * 70)

if __name__ == "__main__":
    main()
