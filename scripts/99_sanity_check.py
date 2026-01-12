# -------------------------------------------------------------------------
# FILE: create_classification_pipeline.py
# DESCRIPTION:
#   - Create classification pipeline structure for newspaper analysis
#   - Generates folders and empty (or templated) Python files
# -------------------------------------------------------------------------

import os
from pathlib import Path

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])
CLASS_DIR = BASE_DIR / "scripts" / "newspapers" / "classification"

FILES = {
    "01a_unsupervised_clustering.py": """# Unsupervised clustering (TF-IDF + KMeans)
# Purpose: discover section-like structure (pre-1990 only)
""",
    "01b_cluster_interpretation.md": """# Cluster Interpretation Log

Record conservative human interpretations here.

Example:
- Cluster 2: ECON (clear)
- Cluster 5: CRIME (clear)
- Cluster 7: AMBIGUOUS (drop)
""",
    "02_train_econ_classifier.py": """# Train supervised econ vs non-econ classifier
# Input: clusters approved in 01b
""",
    "03_apply_econ_classifier.py": """# Apply trained classifier to full 1986–2004 sample
""",
    "04_filter_econ_articles.py": """# Filter econ articles for downstream slant analysis
"""
}

def main():
    print("Creating classification pipeline structure...")

    CLASS_DIR.mkdir(parents=True, exist_ok=True)

    for fname, content in FILES.items():
        fpath = CLASS_DIR / fname
        if fpath.exists():
            print(f"Exists, skipping: {fpath.name}")
            continue

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Created: {fpath.name}")

    print("Done.")

if __name__ == "__main__":
    main()
