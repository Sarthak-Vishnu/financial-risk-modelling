"""
Phase 2 — Step 1: Chunk Item 1A corpus into MLM-ready sequences.

Reads sp500_1A pickle files, splits text into sentence-boundary-aware
chunks of ≤510 tokens, and writes train/val JSONL files.

Output:
    dapt_data/train.jsonl  — filing_date < 2025-01-01
    dapt_data/val.jsonl    — 2025-01-01 ≤ filing_date < 2026-01-01

Each line: {"text": "..."}

Usage:
    python dapt/chunk_corpus.py
"""

import json
import os
import pickle
import re
from pathlib import Path

import nltk
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

ROOT = Path(__file__).resolve().parent.parent
SP500_DIR = ROOT / "datasets" / "sp500_1A"
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"
OUT_DIR = ROOT / "dapt_data"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_TOKENS = 510  # leaves room for [CLS] and [SEP]


def sentence_pack(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Pack sentences greedily into chunks of at most max_tokens tokens."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))

        if sent_len > max_tokens:
            # Sentence itself exceeds limit — flush current buffer then split by token window
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            token_ids = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(token_ids), max_tokens):
                chunks.append(tokenizer.decode(token_ids[i : i + max_tokens], skip_special_tokens=True))
        elif current_len + sent_len > max_tokens:
            chunks.append(" ".join(current))
            current, current_len = [sent], sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def parse_filename(fname: str):
    """Extract (ticker, year) from '{TICKER}_{YEAR}.pickle'."""
    m = re.match(r"^(.+)_(\d{4})\.pickle$", fname)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature table...")
    ft = pd.read_parquet(FEATURE_TABLE)
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    # Build lookup: (ticker, fiscal_year) -> filing_date
    lookup = {
        (row.ticker, int(row.fiscal_year)): row.filing_date
        for row in ft.itertuples()
    }

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    files = sorted(SP500_DIR.glob("*.pickle"))
    print(f"Processing {len(files)} pickle files...")

    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    skipped = []

    train_chunks = val_chunks = 0

    with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
        for fpath in tqdm(files, desc="Chunking"):
            ticker, year = parse_filename(fpath.name)
            if ticker is None:
                skipped.append(fpath.name)
                continue

            filing_date = lookup.get((ticker, year))
            if filing_date is None:
                skipped.append(fpath.name)
                continue

            # Only train and val splits — exclude test (filing_date >= 2026)
            if filing_date >= pd.Timestamp("2026-01-01"):
                continue

            is_val = filing_date >= pd.Timestamp("2025-01-01")

            with open(fpath, "rb") as fh:
                text = pickle.load(fh)

            if not isinstance(text, str) or len(text.strip()) < 50:
                skipped.append(fpath.name)
                continue

            chunks = sentence_pack(text, tokenizer)

            out_file = f_val if is_val else f_train
            for chunk in chunks:
                out_file.write(json.dumps({"text": chunk}) + "\n")

            if is_val:
                val_chunks += len(chunks)
            else:
                train_chunks += len(chunks)

    print(f"\nDone.")
    print(f"  Train sequences : {train_chunks:,}")
    print(f"  Val sequences   : {val_chunks:,}")
    print(f"  Skipped files   : {len(skipped)}")
    if skipped:
        print(f"  First 10 skipped: {skipped[:10]}")
    print(f"  Output -> {OUT_DIR}")


if __name__ == "__main__":
    main()
