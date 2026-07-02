"""
Stage C — earnings-call tone features (pilot).

Management *tone* in earnings calls is genuinely semantic (unlike legalese risk factors), so it is the
text source where dense encoders and finance-sentiment lexicons have the most upside for volatility.

We compute cheap, interpretable per-call features and attach them to each 10-K filing by taking the
most recent call STRICTLY BEFORE the filing_date (known at prediction time -> leakage-free):

  call_n_words            length (log)              very short / very long calls differ
  call_uncertainty_dens   uncertainty-word density  hedging -> higher forward vol
  call_negative_dens      LM-negative density        negative tone -> higher vol
  call_positive_dens      LM-positive density
  call_risk_dens          risk-term density          (shared RISK_TERMS lexicon)
  call_qa_uncertainty     uncertainty density in Q&A only (if speaker split available)
  call_days_before_filing recency of the matched call

Calls currently cover 2025-2026 only, so this is a PILOT scored on val-2025. If it lifts IC we backfill
2006-2024 before committing to the full panel.

Output: datasets/call_features.parquet, keyed (ticker, filing_date).
Run (after the call JSONs are downloaded under datasets/calls/):
    python dataset_config/build_call_features.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
CALLS_DIR = DATA / "calls"
FEATURE_TABLE = DATA / "feature_table.parquet"
OUT_PATH = DATA / "call_features.parquet"

MAX_LOOKBACK_DAYS = 200  # only attach a call within this many days before the filing

# Compact Loughran-McDonald-flavoured lexicons (pilot). Swap for the full LM master list for the thesis.
UNCERTAINTY = set("""approximate approximately uncertain uncertainty uncertainties may might could
possibly possible probable likely unlikely risk risks believe believes anticipate anticipates expect
expects estimate estimates assume assumes depend depends fluctuate fluctuations volatile volatility
contingent indefinite unpredictable unknown variability tentative caution preliminary""".split())
NEGATIVE = set("""adverse adversely loss losses decline declines decrease decreased weak weaker weakness
litigation lawsuit default impairment shortfall downturn recession deteriorate deterioration negative
negatively fail failed failure unable difficult challenging headwind concern concerns slowdown disrupt
disruption restructuring writedown bankruptcy delinquency""".split())
POSITIVE = set("""strong stronger strength growth grow grew improve improved improvement gain gains
profit profitable record robust favorable favourable opportunity opportunities momentum outperform
exceed exceeded beat solid healthy expand expansion confident confidence success successful""".split())
RISK_TERMS = set("""risk risks risky uncertain uncertainty adverse litigation lawsuit loss liability
default impairment decline disruption failure penalty downturn recession volatility competition
regulatory negative material harm bankruptcy insolvency restructuring writedown shortfall""".split())

_WORD = re.compile(r"[A-Za-z']+")


def _density(tokens, lex):
    if not tokens:
        return np.nan
    return sum(t in lex for t in tokens) / len(tokens)


def _qa_from_text(text):
    """Best-effort Q&A segment of a flat transcript string (everything after the operator opens Q&A)."""
    m = re.search(r"question[-\s]and[-\s]answer session", text, re.I)
    if not m:
        m = re.search(r"(?:we['’]?ll|we will)\s+(?:now\s+)?(?:begin|open|take|move to).{0,40}question",
                      text, re.I)
    return text[m.start():] if m else ""


def _extract(obj):
    """Return (full_text, qa_text). Handles a 'transcript' string and/or 'transcript_split' turns."""
    full, qa = "", ""
    if isinstance(obj, dict):
        split = obj.get("transcript_split") or obj.get("transcript_list") or obj.get("split")
        if isinstance(split, list) and split:
            parts, qa_parts, in_qa = [], [], False
            for turn in split:
                if isinstance(turn, dict):
                    spk = str(turn.get("speaker", ""))
                    txt = str(turn.get("text", turn.get("content", "")))
                else:
                    spk, txt = "", str(turn)
                parts.append(txt)
                if re.search(r"question|analyst|operator", spk, re.I) or re.search(
                        r"question-and-answer|q&a|first question", txt, re.I):
                    in_qa = True
                if in_qa:
                    qa_parts.append(txt)
            return " ".join(parts), " ".join(qa_parts)
        full = str(obj.get("transcript") or obj.get("content") or obj.get("text") or "")
    elif isinstance(obj, str):
        full = obj
    return full, _qa_from_text(full)


def _date_of(obj, fname):
    for k in ("date", "call_date", "datetime"):
        if isinstance(obj, dict) and obj.get(k):
            try:
                return pd.to_datetime(obj[k])
            except Exception:
                pass
    m = re.search(r"(20\d{2})[-_]?(\d{2})?[-_]?(\d{2})?", fname)
    if m:
        try:
            return pd.to_datetime("-".join(x for x in m.groups() if x))
        except Exception:
            return pd.NaT
    return pd.NaT


def _ticker_of(obj, fname):
    if isinstance(obj, dict):
        for k in ("ticker", "symbol"):
            if obj.get(k):
                return str(obj[k]).upper().strip()
    m = re.match(r"([A-Za-z.\-]+)[_\.]", Path(fname).name)
    return m.group(1).upper().strip() if m else None


def call_records():
    rows = []
    files = sorted(CALLS_DIR.rglob("*.json")) if CALLS_DIR.exists() else []
    for fp in files:
        try:
            obj = json.load(open(fp))
        except Exception:
            continue
        if isinstance(obj, list):  # some dumps wrap a single call in a list
            obj = obj[0] if obj else {}
        full, qa = _extract(obj)
        toks = [w.lower() for w in _WORD.findall(full)]
        if len(toks) < 50:
            continue
        qa_toks = [w.lower() for w in _WORD.findall(qa)]
        rows.append({
            "ticker": _ticker_of(obj, fp.name),
            "call_date": _date_of(obj, fp.name),
            "call_n_words": float(len(toks)),
            "call_uncertainty_dens": _density(toks, UNCERTAINTY),
            "call_negative_dens": _density(toks, NEGATIVE),
            "call_positive_dens": _density(toks, POSITIVE),
            "call_risk_dens": _density(toks, RISK_TERMS),
            "call_qa_uncertainty": _density(qa_toks, UNCERTAINTY) if qa_toks else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    calls = call_records()
    if calls.empty:
        raise SystemExit(f"No usable call JSONs under {CALLS_DIR} — download calls/ first.")
    calls = calls.dropna(subset=["ticker", "call_date"]).sort_values("call_date")
    print(f"Parsed {len(calls):,} calls | {calls.ticker.nunique()} tickers | "
          f"{calls.call_date.min().date()}..{calls.call_date.max().date()}")

    ft = pd.read_parquet(FEATURE_TABLE)[["ticker", "filing_date"]].copy()
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["ticker"] = ft["ticker"].str.upper().str.strip()

    # most recent call strictly before each filing, within MAX_LOOKBACK_DAYS (leakage-free)
    out = []
    by_tkr = {t: g.sort_values("call_date") for t, g in calls.groupby("ticker")}
    feat_cols = ["call_n_words", "call_uncertainty_dens", "call_negative_dens",
                 "call_positive_dens", "call_risk_dens", "call_qa_uncertainty"]
    for r in ft.itertuples():
        g = by_tkr.get(r.ticker)
        rec = {"ticker": r.ticker, "filing_date": r.filing_date}
        if g is not None:
            prior = g[g.call_date < r.filing_date]
            if len(prior):
                c = prior.iloc[-1]
                gap = (r.filing_date - c.call_date).days
                if gap <= MAX_LOOKBACK_DAYS:
                    for fc in feat_cols:
                        rec[fc] = c[fc]
                    rec["call_n_words"] = np.log1p(rec["call_n_words"])
                    rec["call_days_before_filing"] = float(gap)
        out.append(rec)

    res = pd.DataFrame(out)
    res.to_parquet(OUT_PATH, index=False)
    cov = res[[c for c in res.columns if c.startswith("call_")]].notna().mean()
    matched = res["call_uncertainty_dens"].notna().sum()
    print(f"\nSaved {OUT_PATH}  ({len(res):,} filings, {matched:,} matched to a call)")
    print("Coverage (non-null fraction):")
    print(cov.to_string())


if __name__ == "__main__":
    main()
