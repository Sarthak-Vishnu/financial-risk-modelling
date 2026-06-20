"""
Phase 3 — Increment 1: validate contrastive pairs before any GPU training.

Implements the report's Phase 3 risk mitigation (Table 3): unit-test pair construction,
verify SIC groupings, and inspect positive/negative distributions. Run this on the toy
(`--max_files`) output first, then on the full output.

Checks (raise AssertionError on failure):
  - No pair crosses the train/val boundary (all fiscal years < 2025).
  - sector  : anchor/positive are DIFFERENT firms, SAME sic2 + SAME fiscal year.
  - chrono  : anchor/positive are the SAME firm, SAME sic2, ADJACENT fiscal years.
  - lexical : anchor/positive are the SAME firm+year and share tokens (spans overlap).

Then prints a distribution report (counts per view, per-sector spread, year gaps, units/firm).

Usage:
    python contrastive/validate_pairs.py
    python contrastive/validate_pairs.py --pairs_dir contrastive/pairs
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "contrastive" / "pairs"
TRAIN_END_YEAR = 2025


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def validate_sector(pairs):
    if not pairs:
        print("  [WARN] no sector pairs to check")
        return
    bad_firm = sum(p["anchor_firm"] == p["positive_firm"] for p in pairs)
    bad_sic = sum(p["anchor_sic2"] != p["positive_sic2"] for p in pairs)
    bad_fy = sum(p["anchor_fy"] != p["positive_fy"] for p in pairs)
    check("sector: anchor/positive are different firms", bad_firm == 0, f"{bad_firm} same-firm pairs")
    check("sector: same 2-digit SIC", bad_sic == 0, f"{bad_sic} cross-sic pairs")
    check("sector: same fiscal year", bad_fy == 0, f"{bad_fy} cross-year pairs")


def validate_chrono(pairs):
    if not pairs:
        print("  [WARN] no chrono pairs to check")
        return
    bad_firm = sum(p["anchor_firm"] != p["positive_firm"] for p in pairs)
    bad_sic = sum(p["anchor_sic2"] != p["positive_sic2"] for p in pairs)
    bad_gap = sum(abs(p["anchor_fy"] - p["positive_fy"]) != 1 for p in pairs)
    check("chrono: same firm", bad_firm == 0, f"{bad_firm} cross-firm pairs")
    check("chrono: same sector", bad_sic == 0, f"{bad_sic} cross-sic pairs")
    check("chrono: adjacent fiscal years", bad_gap == 0, f"{bad_gap} non-adjacent pairs")


def validate_lexical(pairs):
    if not pairs:
        print("  [WARN] no lexical pairs to check")
        return
    bad_firm = sum(p["anchor_firm"] != p["positive_firm"] for p in pairs)
    bad_fy = sum(p["anchor_fy"] != p["positive_fy"] for p in pairs)
    no_overlap = sum(not (set(p["anchor_text"].split()) & set(p["positive_text"].split())) for p in pairs)
    check("lexical: same firm", bad_firm == 0, f"{bad_firm} cross-firm pairs")
    check("lexical: same fiscal year", bad_fy == 0, f"{bad_fy} cross-year pairs")
    check("lexical: spans overlap (share tokens)", no_overlap == 0, f"{no_overlap} disjoint pairs")


def validate_no_leakage(all_pairs):
    bad = sum(1 for p in all_pairs if p["anchor_fy"] >= TRAIN_END_YEAR or p["positive_fy"] >= TRAIN_END_YEAR)
    check(f"no pair touches fiscal_year >= {TRAIN_END_YEAR} (train/val leakage)", bad == 0, f"{bad} leaking pairs")


def report(units, lexical, chrono, sector):
    print("\n=== Distribution report ===")
    splits = Counter(u["split"] for u in units)
    print(f"Units: {len(units):,}  (train={splits.get('train',0):,}, val={splits.get('val',0):,})")
    per_firm = Counter(u["ticker"] for u in units if u["split"] == "train")
    if per_firm:
        vals = list(per_firm.values())
        print(f"Train units/firm: firms={len(per_firm)}, median={int(statistics.median(vals))}, "
              f"min={min(vals)}, max={max(vals)}")

    print(f"\nPairs: lexical={len(lexical):,}  chrono={len(chrono):,}  sector={len(sector):,}")

    if sector:
        by_sic = Counter(p["anchor_sic2"] for p in sector)
        groups = {(p["anchor_sic2"], p["anchor_fy"]) for p in sector}
        print(f"  sector: {len(groups)} distinct (sic2,year) groups; top sic2 by pairs: "
              f"{', '.join(f'{s}:{n}' for s, n in by_sic.most_common(5))}")
    if chrono:
        gaps = Counter(abs(p["anchor_fy"] - p["positive_fy"]) for p in chrono)
        firms = {p["anchor_firm"] for p in chrono}
        print(f"  chrono: {len(firms)} firms; year-gap distribution: {dict(gaps)}")
    if lexical:
        aw = statistics.mean(len(p["anchor_text"].split()) for p in lexical)
        pw = statistics.mean(len(p["positive_text"].split()) for p in lexical)
        print(f"  lexical: mean span words anchor={aw:.0f}, positive={pw:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", type=str, default=str(DEFAULT_DIR))
    args = ap.parse_args()
    d = Path(args.pairs_dir)

    units = load_jsonl(d / "units.jsonl")
    lexical = load_jsonl(d / "pairs_lexical.jsonl")
    chrono = load_jsonl(d / "pairs_chrono.jsonl")
    sector = load_jsonl(d / "pairs_sector.jsonl")
    all_pairs = lexical + chrono + sector

    print(f"Loaded {len(units):,} units and {len(all_pairs):,} pairs from {d}\n")
    print("=== Correctness checks ===")
    validate_no_leakage(all_pairs)
    validate_lexical(lexical)
    validate_chrono(chrono)
    validate_sector(sector)

    report(units, lexical, chrono, sector)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
