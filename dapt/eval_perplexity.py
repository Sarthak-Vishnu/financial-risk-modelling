"""
Evaluate MLM perplexity of any model checkpoint on dapt_data/val.jsonl.

Usage:
    # Pre-DAPT baseline
    python dapt/eval_perplexity.py

    # Post-DAPT checkpoint
    python dapt/eval_perplexity.py --model_path dapt_checkpoints/best
"""

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_VAL = ROOT / "dapt_data" / "val.jsonl"
MAX_LENGTH = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--val_file", type=str, default=str(DEFAULT_VAL))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model : {args.model_path}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path).eval().to(device)

    val = load_dataset("json", data_files={"val": args.val_file})["val"]
    val = val.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH, padding=False),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collator)

    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            total_loss += loss.item()
            steps += 1

    perplexity = math.exp(total_loss / steps)
    print(f"\nVal perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
