"""
Phase 4 — Increment 1: validate the BERTopic document set (pre-GPU sanity, report Table 3).

Checks `topics/data/topic_docs.jsonl` before any GPU run:
  1. every doc joins to a feature_table row on (ticker, fiscal_year)
  2. split labels are consistent with filing_date (train<2025 / val 2025 / test>=2026)
  3. the 2026 TEST split is present and non-empty (the whole reason this set is re-extracted)
  4. uids are unique
  5. no empty / too-short documents
and prints a distribution report (per-split / per-year / per-sector counts, filing coverage).

Exit code 0 = all checks pass; 1 = a check failed.

Usage:
    python topics/validate_docs.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = ROOT / "topics" / "data" / "topic_docs.jsonl"
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2026-01-01")


def expected_split(d):
    if d < TRAIN_END:
        return "train"
    if d < VAL_END:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="?", default=str(DEFAULT_DOCS), help="path to topic_docs.jsonl")
    args = ap.parse_args()

    ft = pd.read_parquet(FEATURE_TABLE)
    ft_keys = {(r.ticker, int(r.fiscal_year)) for r in ft.itertuples()}

    docs = [json.loads(l) for l in open(args.docs)]
    print(f"Loaded {len(docs):,} docs from {args.docs}\n")

    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

    # 1. feature_table join
    unjoined = [d for d in docs if (d["ticker"], d["fiscal_year"]) not in ft_keys]
    check("every doc joins feature_table", not unjoined, f"{len(unjoined)} unjoined")

    # 2. split consistency
    bad_split = [d for d in docs
                 if expected_split(pd.Timestamp(d["filing_date"])) != d["split"]]
    check("split label matches filing_date", not bad_split, f"{len(bad_split)} mismatched")

    # 3. test split present
    n_test = sum(1 for d in docs if d["split"] == "test")
    check("2026 test split present", n_test > 0, f"{n_test} test docs")

    # 4. unique uids
    uids = [d["uid"] for d in docs]
    dup = len(uids) - len(set(uids))
    check("uids unique", dup == 0, f"{dup} duplicates")

    # 5. non-empty text
    empty = [d for d in docs if len(d["text"].split()) < 20]
    check("no doc under 20 words", not empty, f"{len(empty)} too short")

    # distribution report
    splits = Counter(d["split"] for d in docs)
    years = Counter(d["fiscal_year"] for d in docs)
    sics = Counter(d["sic2"] for d in docs)
    filings = {(d["ticker"], d["fiscal_year"]) for d in docs}
    print("\nDistribution:")
    print(f"  splits  : {dict(splits)}")
    print(f"  filings : {len(filings):,}")
    print(f"  years   : {dict(sorted(years.items()))}")
    print(f"  top sectors (sic2): {sics.most_common(10)}")

    ok = all(checks)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
