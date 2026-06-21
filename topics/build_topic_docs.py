"""
Phase 4 — Increment 1: build the BERTopic document set.

Re-derives Item-1A risk-factor *paragraph* documents from the sp500_1A pickles and tags each with
firm/sector/year metadata + a temporal split. This mirrors the unit extraction in
`contrastive/build_pairs.py`, with ONE deliberate difference: it keeps the **2026 test** filings
(build_pairs dropped them) so the Phase 5 forward-looking benchmark can score test.

Reuses the exact same preprocessing as the rest of the pipeline (no divergence):
  - `dapt.chunk_corpus.paragraph_split`        — one risk factor per blank-line paragraph
  - `contrastive.build_pairs.normalize_text`   — numbers -> [NUM], dates -> [DATE]
  - `contrastive.build_pairs.parse_filename`   — "<TICKER>_<YEAR>.pickle"

Split (by filing_date, report §3.1.1):
  train  : filing_date <  2025-01-01   (2006–2024)
  val    : 2025-01-01 <= filing_date < 2026-01-01
  test   : filing_date >= 2026-01-01

Output (topics/data/):
  topic_docs.jsonl — one line per paragraph:
    {uid, text, ticker, cik, fiscal_year, sic2, filing_date, split}

Usage:
    python topics/build_topic_docs.py                 # full corpus
    python topics/build_topic_docs.py --max_files 50  # toy run
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dapt.chunk_corpus import paragraph_split
from contrastive.build_pairs import normalize_text, parse_filename

SP500_DIR = ROOT / "datasets" / "sp500_1A"
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"
OUT_DIR = ROOT / "topics" / "data"

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2026-01-01")


def split_of(filing_date: pd.Timestamp) -> str:
    if filing_date < TRAIN_END:
        return "train"
    if filing_date < VAL_END:
        return "val"
    return "test"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--max_files", type=int, default=None, help="Toy run: limit pickles processed")
    p.add_argument("--min_para_words", type=int, default=20, help="Drop paragraphs shorter than this")
    p.add_argument("--blank_ticker", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading feature table...")
    ft = pd.read_parquet(FEATURE_TABLE)
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    lookup = {
        (row.ticker, int(row.fiscal_year)): (int(row.cik), str(row.sic)[:2], row.filing_date)
        for row in ft.itertuples()
    }

    files = sorted(SP500_DIR.glob("*.pickle"))
    if args.max_files:
        files = files[: args.max_files]
    print(f"Processing {len(files)} pickle files...")

    out_path = out_dir / "topic_docs.jsonl"
    n_docs, skipped = 0, 0
    splits, by_year = Counter(), Counter()
    filings = set()
    with open(out_path, "w") as out:
        for fpath in files:
            ticker, year = parse_filename(fpath.name)
            if ticker is None or (ticker, year) not in lookup:
                skipped += 1
                continue
            cik, sic2, filing_date = lookup[(ticker, year)]
            split = split_of(filing_date)

            with open(fpath, "rb") as fh:
                text = pickle.load(fh)
            if not isinstance(text, str) or len(text.strip()) < 50:
                skipped += 1
                continue

            for idx, para in enumerate(paragraph_split(text)):
                norm = normalize_text(para, ticker, args.blank_ticker)
                if len(norm.split()) < args.min_para_words:
                    continue
                rec = {
                    "uid": f"{ticker}_{year}_{idx}",
                    "text": norm, "ticker": ticker, "cik": cik,
                    "fiscal_year": year, "sic2": sic2,
                    "filing_date": filing_date.strftime("%Y-%m-%d"), "split": split,
                }
                out.write(json.dumps(rec) + "\n")
                n_docs += 1
                splits[split] += 1
                by_year[year] += 1
                filings.add((ticker, year))

    print(f"\nDone. Docs: {n_docs:,} | skipped files: {skipped} | distinct filings: {len(filings):,}")
    print(f"  splits : {dict(splits)}")
    print(f"  years  : {dict(sorted(by_year.items()))}")
    print(f"  Output -> {out_path}")


if __name__ == "__main__":
    main()
