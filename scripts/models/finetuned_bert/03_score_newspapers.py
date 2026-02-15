"""
03_score_newspapers.py -- Score newspaper articles with fine-tuned BERT.

For each congress:
  1. Load the fine-tuned model for that window
  2. Load + sample newspaper articles (same seed as SBERT)
  3. Compute slant = logit_R - logit_D
  4. Save article-level slant scores
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.finetuned_bert.config import (
    MAX_SEQ_LENGTH, EVAL_BATCH_SIZE,
    SCORE_CONGRESSES, NEWSPAPER_SAMPLE_PER_CONGRESS,
    RAW_NEWSPAPERS, MODEL_DIR, OUTPUT_DIR,
    get_windows, congress_to_years,
)


class ArticleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    windows = get_windows()
    all_slant = []

    for (prev_cong, curr_cong) in windows:
        print(f"\n{'='*60}")
        print(f"Congress {curr_cong} (window {prev_cong},{curr_cong})")

        # Load model
        model_path = MODEL_DIR / f"window_{prev_cong}_{curr_cong}"
        if not model_path.exists():
            print(f"  Model not found at {model_path}, skipping")
            continue

        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()

        # Load newspapers
        year1, year2 = congress_to_years(curr_cong)
        dfs = []
        for yr in [year1, year2]:
            path = RAW_NEWSPAPERS / f"newspapers_{yr}.parquet"
            if not path.exists():
                print(f"  WARNING: {path.name} not found")
                continue
            df_yr = pd.read_parquet(path)
            df_yr["year"] = yr
            dfs.append(df_yr)

        if not dfs:
            print(f"  No newspaper data, skipping")
            continue

        articles = pd.concat(dfs, ignore_index=True)
        articles["congress"] = curr_cong

        # Sample (same seed as SBERT)
        n_sample = min(NEWSPAPER_SAMPLE_PER_CONGRESS, len(articles))
        articles = articles.sample(n=n_sample, random_state=42).reset_index(drop=True)
        print(f"  Articles: {len(articles):,}")

        # Combine title + text
        texts = (
            articles["title"].fillna("") + " " + articles["text"].fillna("")
        ).str.strip().tolist()

        # Inference
        ds = ArticleDataset(texts, tokenizer, MAX_SEQ_LENGTH)
        loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

        all_logits = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                all_logits.append(outputs.logits.cpu().numpy())

        logits = np.vstack(all_logits)  # (n_articles, 2)
        slant = logits[:, 1] - logits[:, 0]  # R logit - D logit

        # Save
        result = articles[["date", "paper", "title", "word_count", "year", "congress"]].copy()
        result["slant"] = slant.astype(np.float32)
        result["prob_R"] = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy().astype(np.float32)
        result.to_parquet(OUTPUT_DIR / f"article_slant_cong_{curr_cong}.parquet", index=False)

        print(f"  Slant: mean={slant.mean():.4f}, std={slant.std():.4f}, "
              f"min={slant.min():.4f}, max={slant.max():.4f}")
        print(f"  P(R): mean={result['prob_R'].mean():.4f}")

        all_slant.append(result)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Combined
    if all_slant:
        combined = pd.concat(all_slant, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "article_slant_all.parquet", index=False)
        print(f"\n{'='*60}")
        print(f"Total articles scored: {len(combined):,}")
        print(f"Saved: article_slant_all.parquet")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
