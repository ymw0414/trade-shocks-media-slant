# -------------------------------------------------------------------------
# FILE: 03_train_econ_classifier.py
# -------------------------------------------------------------------------

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

BASE_DIR = Path(os.environ["SHIFTING_SLANT_DIR"])

ECON_DIR  = BASE_DIR / "scripts" / "newspapers" / "classification" / "econ"
CLEAN_DIR = BASE_DIR / "data" / "processed" / "newspapers" / "clean"

LABEL_CHATGPT = ECON_DIR / "02_llm_labels_chatgpt_thinking.csv"
LABEL_GEMINI  = ECON_DIR / "02_llm_labels_gemini_thinking.csv"
SAMPLE_FILE   = ECON_DIR / "01_llm_annotation_sample.csv"

OUT_MODEL = ECON_DIR / "econ_classifier.joblib"
OUT_VECT  = ECON_DIR / "econ_vectorizer.joblib"

def main():

    print("=" * 70)
    print("03_train_econ_classifier.py")
    print("Training econ vs non-econ classifier")
    print("=" * 70)

    # --------------------------------------------------
    # Load labels and fix types
    # --------------------------------------------------
    df_c = pd.read_csv(LABEL_CHATGPT)
    df_g = pd.read_csv(LABEL_GEMINI)

    df_c["article_id"] = df_c["article_id"].astype(int)
    df_g["article_id"] = df_g["article_id"].astype(int)

    df = df_c.merge(df_g, on="article_id", suffixes=("_c", "_g"))

    # Keep only agreement
    df = df[df["label_c"] == df["label_g"]].copy()
    df["y"] = (df["label_c"] == "ECON").astype(int)

    print(f"Labeled (agreement) samples: {len(df):,}")
    print(f"Econ share: {df['y'].mean():.3f}")

    # --------------------------------------------------
    # Load text (same article_id type!)
    # --------------------------------------------------
    text_df = pd.read_csv(SAMPLE_FILE)
    text_df["article_id"] = text_df["article_id"].astype(int)

    df = df.merge(
        text_df[["article_id", "clean_text"]],
        on="article_id",
        how="inner"
    )

    texts = df["clean_text"].fillna("")
    y = df["y"].values

    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=50_000
    )

    X = vectorizer.fit_transform(texts)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    print("Validation performance:")
    print(classification_report(y_te, clf.predict(X_te)))

    joblib.dump(clf, OUT_MODEL)
    joblib.dump(vectorizer, OUT_VECT)

    print("Saved model and vectorizer.")

if __name__ == "__main__":
    main()
