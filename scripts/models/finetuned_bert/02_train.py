"""
02_train.py -- Fine-tune BERT for R/D speech classification per window.

For each 2-congress window:
  1. Load train/val splits from 01_prepare_data.py
  2. Fine-tune bert-base-uncased with classification head
  3. Log accuracy/F1 per epoch
  4. Save best model (by val loss)
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.finetuned_bert.config import (
    MODEL_NAME, MAX_SEQ_LENGTH, NUM_LABELS,
    LEARNING_RATE, BATCH_SIZE, EVAL_BATCH_SIZE,
    NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY,
    DATA_DIR, MODEL_DIR, get_windows,
)


class SpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_one_window(prev_cong, curr_cong, tokenizer, device):
    """Fine-tune BERT for one window."""
    print(f"\n{'='*60}")
    print(f"Window ({prev_cong}, {curr_cong})")
    print(f"{'='*60}")

    # Load data
    train_df = pd.read_parquet(DATA_DIR / f"train_{prev_cong}_{curr_cong}.parquet")
    val_df = pd.read_parquet(DATA_DIR / f"val_{prev_cong}_{curr_cong}.parquet")

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()

    print(f"  Train: {len(train_texts):,} (R={sum(train_labels):,}, D={len(train_labels)-sum(train_labels):,})")
    print(f"  Val:   {len(val_texts):,}")

    # Datasets
    train_ds = SpeechDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
    val_ds = SpeechDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Model (fresh for each window)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    best_val_loss = float("inf")
    save_dir = MODEL_DIR / f"window_{prev_cong}_{curr_cong}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().numpy())
        val_loss /= len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"acc={acc:.4f}, F1={f1:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    print(f"  Best val_loss={best_val_loss:.4f}, saved to {save_dir.name}")
    return {"window": f"({prev_cong},{curr_cong})", "best_val_loss": best_val_loss,
            "final_acc": acc, "final_f1": f1}


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    results = []
    for prev_cong, curr_cong in get_windows():
        res = train_one_window(prev_cong, curr_cong, tokenizer, device)
        results.append(res)
        # Free GPU memory between windows
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['window']}: val_loss={r['best_val_loss']:.4f}, "
              f"acc={r['final_acc']:.4f}, F1={r['final_f1']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
