"""
Phase 3 — Increment 1: contrastive pair construction.

Re-derives risk-factor *paragraph* units from the Item 1A pickles, tags each with
firm/sector/year metadata (joined from feature_table.parquet), normalises shortcut
cues, and generates positive pairs for three contrastive views:

  - lexical       : two random nested spans of the SAME paragraph (Chiu et al. 2025).
  - chronological : same firm, ADJACENT years (t, t+1), risk factors matched by
                    TF-IDF cosine -> captures temporal persistence of a specific risk.
  - sector        : two paragraphs from DIFFERENT firms sharing the same 2-digit SIC
                    and fiscal year (soft positive). Same-(sic2,fy) items are the labels
                    used later for label-aware false-negative masking (Khosla 2020).

All pairs are built from the TRAIN split only (filing_date < 2025-01-01); 2025 filings
are emitted as val units (monitoring) but not paired. Test filings (>= 2026) are excluded.

Normalisation (shortcut prevention, Literature_agent_phase3.md Q2):
  numbers -> [NUM] (SEC-BERT-NUM), dates -> [DATE]. Firm/ticker blanking is left OFF by
  default: naive ticker string-replace mangles common-word tickers (e.g. "A", "ALL", "IT");
  proper probabilistic entity blanking is deferred to Increment 2.

Output (contrastive/pairs/):
  units.jsonl, pairs_lexical.jsonl, pairs_chrono.jsonl, pairs_sector.jsonl

Usage:
    python contrastive/build_pairs.py                 # full corpus
    python contrastive/build_pairs.py --max_files 50  # toy run for validation
"""

import argparse
import json
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dapt.chunk_corpus import paragraph_split  # reuse Phase 2 paragraph splitter

SP500_DIR = ROOT / "datasets" / "sp500_1A"
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"
OUT_DIR = ROOT / "contrastive" / "pairs"

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2026-01-01")

# --- normalisation -----------------------------------------------------------
_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
           r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_PATTERNS = [
    re.compile(rf"{_MONTHS}\.?\s+\d{{1,2}},?\s+\d{{4}}", re.I),  # March 3, 2024
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                  # 03/03/2024
    re.compile(r"\b(?:19|20)\d{2}\b"),                          # bare 4-digit year
]
_NUM_PATTERN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_WS = re.compile(r"\s+")


def normalize_text(text: str, ticker: str | None = None, blank_ticker: bool = False) -> str:
    for pat in _DATE_PATTERNS:
        text = pat.sub("[DATE]", text)
    text = _NUM_PATTERN.sub("[NUM]", text)
    if blank_ticker and ticker and len(ticker) >= 2:
        text = re.sub(rf"\b{re.escape(ticker)}\b", "[COMPANY]", text)
    return _WS.sub(" ", text).strip()


def parse_filename(fname: str):
    m = re.match(r"^(.+)_(\d{4})\.pickle$", fname)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


# --- pair builders -----------------------------------------------------------
def lexical_spans(words: list[str], rng: random.Random) -> tuple[str, str]:
    """Two random nested spans [0:b] and [a:n] with a < b (overlap = words[a:b])."""
    n = len(words)
    a = rng.randint(1, n - 2)
    b = rng.randint(a + 1, n - 1)
    return " ".join(words[:b]), " ".join(words[a:])


def _rec(view, a_text, p_text, a, p):
    """Flat pair record carrying per-side tags for label-aware masking in training."""
    return {
        "view": view,
        "anchor_text": a_text,
        "positive_text": p_text,
        "anchor_firm": a["ticker"], "positive_firm": p["ticker"],
        "anchor_sic2": a["sic2"], "positive_sic2": p["sic2"],
        "anchor_fy": a["fiscal_year"], "positive_fy": p["fiscal_year"],
    }


def build_lexical(units, rng, min_words):
    pairs = []
    for u in units:
        words = u["text"].split()
        if len(words) < min_words:
            continue
        s1, s2 = lexical_spans(words, rng)
        pairs.append(_rec("lexical", s1, s2, u, u))
    return pairs


def build_chrono(units, rng, sim_threshold):
    by_firm = defaultdict(lambda: defaultdict(list))
    for u in units:
        by_firm[u["ticker"]][u["fiscal_year"]].append(u)
    pairs = []
    for firm, by_year in by_firm.items():
        years = sorted(by_year)
        for t in years:
            if t + 1 not in by_year:
                continue
            A, B = by_year[t], by_year[t + 1]
            ta, tb = [u["text"] for u in A], [u["text"] for u in B]
            try:
                vec = TfidfVectorizer().fit(ta + tb)
            except ValueError:
                continue  # empty vocabulary
            sims = cosine_similarity(vec.transform(ta), vec.transform(tb))
            for i, ua in enumerate(A):
                j = int(sims[i].argmax())
                if sims[i][j] >= sim_threshold:
                    pairs.append(_rec("chrono", ua["text"], B[j]["text"], ua, B[j]))
    return pairs


def build_sector(units, rng, pairs_per_group):
    by_group = defaultdict(list)
    for u in units:
        by_group[(u["sic2"], u["fiscal_year"])].append(u)
    pairs = []
    for (sic2, fy), group in by_group.items():
        by_firm = defaultdict(list)
        for u in group:
            by_firm[u["ticker"]].append(u)
        firms = list(by_firm)
        if len(firms) < 2:
            continue
        seen = set()
        attempts = 0
        while len(seen) < pairs_per_group and attempts < pairs_per_group * 5:
            attempts += 1
            f1, f2 = rng.sample(firms, 2)
            u1, u2 = rng.choice(by_firm[f1]), rng.choice(by_firm[f2])
            key = (u1["uid"], u2["uid"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(_rec("sector", u1["text"], u2["text"], u1, u2))
    return pairs


# --- main --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--max_files", type=int, default=None, help="Toy run: limit pickles processed")
    p.add_argument("--min_para_words", type=int, default=20, help="Drop paragraphs shorter than this")
    p.add_argument("--min_lexical_words", type=int, default=24, help="Min words to form a lexical pair")
    p.add_argument("--chrono_sim_threshold", type=float, default=0.5, help="TF-IDF cosine to match risk factors")
    p.add_argument("--sector_pairs_per_group", type=int, default=50)
    p.add_argument("--blank_ticker", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
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

    units = []
    skipped = 0
    for fpath in files:
        ticker, year = parse_filename(fpath.name)
        if ticker is None or (ticker, year) not in lookup:
            skipped += 1
            continue
        cik, sic2, filing_date = lookup[(ticker, year)]
        if filing_date >= VAL_END:
            continue  # test held out
        split = "train" if filing_date < TRAIN_END else "val"

        with open(fpath, "rb") as fh:
            text = pickle.load(fh)
        if not isinstance(text, str) or len(text.strip()) < 50:
            skipped += 1
            continue

        for idx, para in enumerate(paragraph_split(text)):
            norm = normalize_text(para, ticker, args.blank_ticker)
            if len(norm.split()) < args.min_para_words:
                continue
            units.append({
                "uid": f"{ticker}_{year}_{idx}",
                "text": norm, "ticker": ticker, "cik": cik,
                "fiscal_year": year, "sic2": sic2,
                "filing_date": filing_date.strftime("%Y-%m-%d"), "split": split,
            })

    train_units = [u for u in units if u["split"] == "train"]
    print(f"Units: {len(units):,} total ({len(train_units):,} train) | skipped files: {skipped}")

    print("Building pairs (train only)...")
    lexical = build_lexical(train_units, rng, args.min_lexical_words)
    chrono = build_chrono(train_units, rng, args.chrono_sim_threshold)
    sector = build_sector(train_units, rng, args.sector_pairs_per_group)

    def dump(name, rows):
        path = out_dir / name
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return path

    dump("units.jsonl", units)
    dump("pairs_lexical.jsonl", lexical)
    dump("pairs_chrono.jsonl", chrono)
    dump("pairs_sector.jsonl", sector)

    print("\nDone.")
    print(f"  lexical pairs : {len(lexical):,}")
    print(f"  chrono pairs  : {len(chrono):,}")
    print(f"  sector pairs  : {len(sector):,}")
    print(f"  Output -> {out_dir}")


if __name__ == "__main__":
    main()
