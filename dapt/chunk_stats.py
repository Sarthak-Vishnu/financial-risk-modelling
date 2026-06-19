"""
Report fill-rate statistics for a chunked JSONL file (one {"text": ...} per line).

Used to compare packing strategies: a higher mean tokens/chunk and a higher
fraction of near-full chunks means less wasted token budget — which matters for
all-mpnet-base-v2's mean pooling (padding dilutes the mean).

Usage:
    python dapt/chunk_stats.py dapt_data/train.jsonl
    python dapt/chunk_stats.py dapt_data_para/train.jsonl
"""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_TOKENS = 510
NEAR_FULL = 480  # chunks at/above this are considered well-filled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=str, help="Path to chunked JSONL file")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.jsonl)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lengths = []
    with open(path) as f:
        for line in f:
            text = json.loads(line)["text"]
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))

    n = len(lengths)
    near_full = sum(1 for l in lengths if l >= NEAR_FULL)

    print(f"File           : {path}")
    print(f"Chunks         : {n:,}")
    print(f"Mean tokens    : {statistics.mean(lengths):.1f}")
    print(f"Median tokens  : {statistics.median(lengths):.1f}")
    print(f"Min / Max      : {min(lengths)} / {max(lengths)}")
    print(f"% >= {NEAR_FULL} tokens : {100 * near_full / n:.1f}%  (fill-rate proxy, cap {MAX_TOKENS})")


if __name__ == "__main__":
    main()
