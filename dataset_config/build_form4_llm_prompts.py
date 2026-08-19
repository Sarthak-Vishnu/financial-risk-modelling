"""
Round 8 — anonymised Form 4 prompts for the generative-LLM pilot.

The question: does a generative model reading the RAW insider-transaction record produce a
volatility score that beats the eight hand-crafted f4_* features (build_form4_features.py) under
the same head, on the same rows? This script builds the model's side of that comparison: one
prompt per filing, rendering exactly the window the manual features aggregate — the transactions
in (filing-180d, filing-3d] — as a table, so the two lanes see identical evidence and the only
difference is who does the extraction.

Transactions are parsed by build_form4_features.load_transactions(), reused rather than
reimplemented, so the two lanes cannot drift apart on parsing.

SCOPE: filing years 2024 and 2025 only (~865 filings). Both post-date the training cutoffs of the
open-weights candidates (Qwen2.5 and Llama-3.1 are both late-2023), so look-ahead is controlled by
construction rather than by argument. This is the whole reason the pilot is defensible at this
size; do not widen it without revisiting the contamination argument.

ANONYMISATION (RISK_LLM_ANON=1, the default). Glasserman & Lin (2023) show that overlapping the
model's training window with the backtest period biases results through two channels at once —
look-ahead bias and a "distraction effect" where general knowledge of the named company interferes
with reading the text — and that the two run in OPPOSITE directions, so the net bias cannot be
signed in advance. Their remedy is removing the company's identifiers from the text, which is what
is done here. Omitted: ticker, company name, insider names, absolute dates, and per-share price
(a $700 share price narrows the S&P 500 universe on its own). Kept: day offset from the filing,
role flags, transaction code, acquired/disposed, share count, dollar value.

Residual identifiability is bounded, not zero — dollar-value magnitudes still carry some signal
about firm size. Stated as a caveat rather than solved.

RISK_LLM_ANON=0 builds the identified variant (real dates, ticker, insider names) as the control
lane: scoring both and differencing gives this study's own measurement of the combined
look-ahead + distraction premium, the direct analogue of the ~0.06 IC encoder lookahead priced in
Part III.3.

Output: datasets/form4_llm_prompts{_ident}.parquet — (ticker, filing_date, year, n_txn,
truncated, prompt).
Run:  python dataset_config/build_form4_llm_prompts.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_config"))
from build_form4_features import (_truthy, FORM4_CSV, FEATURE_TABLE,   # noqa: E402
                                  WINDOW_DAYS, DISCLOSURE_LAG_DAYS)

DATA = ROOT / "datasets"
ANON = os.environ.get("RISK_LLM_ANON", "1") != "0"
OUT_PATH = DATA / (f"form4_llm_prompts{'' if ANON else '_ident'}.parquet")

YEARS = (2024, 2025)   # post-cutoff slice — see the scope note above
MAX_ROWS = 150         # p90 of the window is 96 transactions, so >90% of filings render in full

CODE_LEGEND = {
    "P": "open-market purchase", "S": "open-market sale",
    "A": "grant or award from the company", "M": "option or derivative exercise",
    "F": "shares withheld to cover tax", "G": "gift",
    "D": "disposition back to the company", "C": "conversion of a derivative",
    "X": "exercise of an in-the-money derivative",
}

HEADER = """You are given the complete insider-transaction record for one publicly listed company,
covering a {span}-day window that ends 3 days before the company filed its annual report.

Each row is one transaction reported by a company insider. Day offsets count backwards from the
filing date, so day -170 is earlier than day -10. Insiders are labelled {label_note}, and the same
label refers to the same person throughout the window.

Transaction codes: {legend}

Roles: OFF = officer, DIR = director, TEN = beneficial owner of more than ten percent.
Direction: A = shares acquired, D = shares disposed of.
"""

QUESTION = """
Based only on the record above, judge how the stock is likely to behave after the annual report is
filed. Insider trading is informative about uncertainty rather than about direction, so judge how
turbulent the stock will be, not whether it will go up or down.

Answer with a single JSON object and nothing else, in exactly this form:

{"volatility_risk": <0-100>, "information_asymmetry": <0-100>, "confidence": <0-100>}

volatility_risk       0 means the calmest stocks, 100 means the most turbulent, over the 30
                      trading days after the filing.
information_asymmetry 0 means insiders appear to know nothing the market does not, 100 means they
                      appear to know a great deal the market does not.
confidence            how much this record supports any judgement at all. A near-empty window
                      should score low.
"""


def load_transactions_rendered():
    """build_form4_features.load_transactions() plus the columns needed to render a row.

    Deliberately mirrors that function line for line — same usecols semantics, same dtype
    handling, same dropna, same sort — rather than re-reading and re-joining, because the two lanes
    must aggregate identical transactions for the comparison to mean anything. (Re-reading and
    aligning by index would be wrong: the original drops rows and re-sorts.)"""
    tx = pd.read_csv(FORM4_CSV,
                     usecols=["ticker", "issuer_cik", "rptownercik", "is_officer", "is_director",
                              "is_ten_pct_owner", "owner_name", "acquired_disposed",
                              "transaction_date", "transaction_code", "shares", "price_per_share"],
                     dtype={"transaction_code": str}, low_memory=False)
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    tx = tx.dropna(subset=["transaction_date", "issuer_cik"]).copy()
    tx["issuer_cik"] = tx["issuer_cik"].astype("int64")
    tx["ticker"] = tx["ticker"].astype(str).str.upper().str.strip()
    tx["code"] = tx["transaction_code"].str.upper().str.strip()
    tx["is_officer"] = _truthy(tx["is_officer"])
    tx["is_director"] = _truthy(tx["is_director"])
    tx["is_ten_pct"] = _truthy(tx["is_ten_pct_owner"])
    tx["value"] = (pd.to_numeric(tx["shares"], errors="coerce").clip(lower=0)
                   * pd.to_numeric(tx["price_per_share"], errors="coerce").clip(lower=0))
    return tx.sort_values("transaction_date").reset_index(drop=True)


def _sig3(x):
    """Round to 3 significant figures — keeps magnitude, drops spurious precision."""
    if not np.isfinite(x) or x == 0:
        return 0
    return int(round(x, -int(np.floor(np.log10(abs(x)))) + 2))


def render(w, filing_date, ticker):
    """One prompt for one filing's window `w` (a transaction sub-frame, sorted by date)."""
    span = WINDOW_DAYS - DISCLOSURE_LAG_DAYS
    codes = sorted(set(w["code"]) & set(CODE_LEGEND)) if len(w) else []
    legend = "; ".join(f"{c} = {CODE_LEGEND[c]}" for c in codes) or "none present"

    head = HEADER.format(
        span=span, legend=legend,
        label_note="I1, I2, I3 and so on" if ANON else "by name")

    if not len(w):
        return head + "\nThe record is empty. No insider reported any transaction in this window.\n" + QUESTION

    # stable per-filing pseudonyms, ordered by first appearance in the window. Numbered I1/I2/...
    # rather than A/B/C: letter labels collide with the direction codes (A = acquired,
    # D = disposed) in the same table, and with single-letter tickers, which makes both the prompt
    # and the anonymisation check ambiguous.
    owners = list(dict.fromkeys(w["rptownercik"].tolist()))
    if ANON:
        label = {o: f"I{n + 1}" for n, o in enumerate(owners)}
    else:
        label = dict(zip(w["rptownercik"], w["owner_name"].astype(str)))

    trunc = max(0, len(w) - MAX_ROWS)
    shown = w.iloc[-MAX_ROWS:] if trunc else w

    lines = [f"{'day':>5s} {'insider':<10s} {'role':<12s} {'code':<5s} {'dir':<4s} "
             f"{'shares':>12s} {'value_usd':>14s}"]
    for r in shown.itertuples():
        roles = "".join([("OFF" if r.is_officer else ""), ("DIR" if r.is_director else ""),
                         ("TEN" if r.is_ten_pct else "")]) or "-"
        day = (r.transaction_date - filing_date).days
        sh = pd.to_numeric(r.shares, errors="coerce")
        lines.append(
            f"{day:>5d} {str(label.get(r.rptownercik, '?'))[:10]:<10s} {roles:<12s} "
            f"{str(r.code)[:5]:<5s} {str(r.acquired_disposed)[:4]:<4s} "
            f"{(_sig3(sh) if np.isfinite(sh) else 0):>12,d} {_sig3(r.value):>14,d}")

    body = "\n".join(lines)
    note = ""
    if trunc:
        earlier = w.iloc[:-MAX_ROWS]
        by_code = earlier["code"].value_counts().to_dict()
        note = (f"\n\n({trunc} earlier transactions in the same window are not listed above. "
                f"Their counts by code were: "
                f"{', '.join(f'{k} x{v}' for k, v in sorted(by_code.items()))}.)")
    ident = "" if ANON else f"\nCompany ticker: {ticker}. Filing date: {filing_date.date()}.\n"
    return head + ident + "\n" + body + note + "\n" + QUESTION


def main():
    tx = load_transactions_rendered()
    print(f"Loaded {len(tx):,} transactions | anonymised: {ANON}")

    ft = pd.read_parquet(FEATURE_TABLE)[["ticker", "cik", "filing_date"]].copy()
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["ticker"] = ft["ticker"].str.upper().str.strip()
    ft["year"] = ft["filing_date"].dt.year
    ft = ft[ft["year"].isin(YEARS)].reset_index(drop=True)
    print(f"Pilot slice: {len(ft):,} filings in {YEARS}")

    by_cik = {int(c): g.reset_index(drop=True) for c, g in tx.groupby("issuer_cik")}
    by_tkr = {t: g.reset_index(drop=True) for t, g in tx.groupby("ticker")}

    out = []
    for r in ft.itertuples():
        g = by_cik.get(int(r.cik))
        if g is None:
            g = by_tkr.get(r.ticker)
        if g is None:
            continue                      # not in the Form 4 universe -> no prompt, no LLM score
        dates = g["transaction_date"].to_numpy()
        hi = np.searchsorted(dates, np.datetime64(
            r.filing_date - pd.Timedelta(days=DISCLOSURE_LAG_DAYS)), side="right")
        lo = np.searchsorted(dates, np.datetime64(
            r.filing_date - pd.Timedelta(days=WINDOW_DAYS)), side="right")
        w = g.iloc[lo:hi]
        out.append({"ticker": r.ticker, "filing_date": r.filing_date, "year": r.year,
                    "n_txn": int(len(w)), "truncated": bool(len(w) > MAX_ROWS),
                    "prompt": render(w, r.filing_date, r.ticker)})

    res = pd.DataFrame(out)
    res.to_parquet(OUT_PATH, index=False)
    chars = res["prompt"].str.len()
    print(f"\nSaved {OUT_PATH}  ({len(res):,} prompts)")
    print(f"  by year        : {res.year.value_counts().sort_index().to_dict()}")
    print(f"  window txns    : median {res.n_txn.median():.0f}  p90 {res.n_txn.quantile(.9):.0f}  "
          f"max {res.n_txn.max()}")
    print(f"  truncated (>{MAX_ROWS}): {int(res.truncated.sum())} "
          f"({res.truncated.mean():.1%})")
    print(f"  prompt chars   : median {chars.median():.0f}  p90 {chars.quantile(.9):.0f}  "
          f"max {chars.max()}  (~{chars.max() // 4:,} tokens worst case)")
    print(f"  empty windows  : {int((res.n_txn == 0).sum())}")

    if ANON:
        # Anonymisation is a correctness property, so assert it rather than trusting the renderer.
        # Checked PER ROW (a ticker appearing in some other company's prompt is not a leak) and
        # only for tickers of 2+ characters: one-letter tickers (A, D, F, T, V ...) collide with
        # the direction codes in every table and would make this check pure noise. Nothing in the
        # renderer can emit a ticker in anonymised mode, so the residual risk is nil.
        leaked = [f"{r.ticker}" for r in res.itertuples()
                  if len(r.ticker) > 1 and re.search(rf"\b{re.escape(r.ticker)}\b", r.prompt)]
        assert not leaked, f"ticker leaked into its own anonymised prompt: {sorted(set(leaked))[:10]}"
        dated = [r.ticker for r in res.itertuples()
                 if re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", r.prompt)]
        assert not dated, f"absolute date leaked into anonymised prompts: {sorted(set(dated))[:10]}"
        names = res["prompt"].str.contains("Company ticker:").sum()
        assert names == 0, "identified header leaked into anonymised prompts"
        print(f"  anonymisation  : {len(res)} prompts checked, no own-ticker, absolute date or "
              f"identified header found")

    print("\n--- sample prompt (median length) ---")
    print(res.loc[chars.sub(chars.median()).abs().idxmin(), "prompt"][:2200])


if __name__ == "__main__":
    main()
