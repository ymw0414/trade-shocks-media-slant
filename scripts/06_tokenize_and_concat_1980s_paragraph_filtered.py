"""
06_tokenize_and_concat_1980s_paragraph_filtered.py

This script tokenizes the *paragraph-filtered* 1980s dataset.
The input file (speeches_clean_1980s.parquet) already contains:
    - Only 97th–100th Congress speeches (1981–1989)
    - Paragraph-filtered speeches (>=2 sentences AND >=200 chars)

Pipeline:
    1. Load parquet in streaming batches
    2. Tokenize each batch into chunk_i/
    3. Save final merged dataset into tokenized_1980s_paragraph_filtered_full/

Outputs:
    data/processed/tokenized_1980s_paragraph_filtered_streaming/chunk_*
    data/processed/tokenized_1980s_paragraph_filtered_full/
"""

import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import RobertaTokenizerFast
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE = Path("C:/Users/ymw04/Dropbox/shifting_slant")

DATA_PATH = BASE / "data/processed/speeches_clean_1980s.parquet"  # already filtered
CHUNK_DIR = BASE / "data/processed/tokenized_1980s_paragraph_filtered_streaming"
FINAL_DIR = BASE / "data/processed/tokenized_1980s_paragraph_filtered_full"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------
print("Loading tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def tokenize_batch(batch):
    return tokenizer(
        batch["speech"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


# ---------------------------------------------------------
# Step 1: Stream parquet → tokenize chunks
# ---------------------------------------------------------
print("Opening parquet...")
parquet = pq.ParquetFile(DATA_PATH)

CHUNK_SIZE = 20000
chunk_id = 0
chunk_paths = []

for batch in parquet.iter_batches(batch_size=CHUNK_SIZE):
    print(f"Processing chunk {chunk_id}...")

    df = batch.to_pandas()

    # labels
    df["labels"] = df["party"].map({"D": 0, "R": 1})

    ds = Dataset.from_pandas(df[["speech", "labels"]])
    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.remove_columns(["speech"])

    out_path = CHUNK_DIR / f"chunk_{chunk_id}"
    ds.save_to_disk(str(out_path))
    chunk_paths.append(out_path)

    print(f"Saved chunk {chunk_id} → {out_path}")

    del df, ds
    chunk_id += 1

print(f"Finished tokenizing into {len(chunk_paths)} chunks.")


# ---------------------------------------------------------
# Step 2: Load chunks → concat → save final dataset
# ---------------------------------------------------------
print("Loading chunks and concatenating...")

datasets = [load_from_disk(str(p)) for p in chunk_paths]
merged = concatenate_datasets(datasets)

print("Saving merged dataset...")
merged.save_to_disk(str(FINAL_DIR))

print("Done.")
print(f"Final dataset saved to: {FINAL_DIR}")
