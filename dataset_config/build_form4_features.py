"""
Stage E — Form 4 insider-trading features.

Insider-trading activity proxies information asymmetry, and information asymmetry predicts
volatility (not returns — Rule 2 of the data-integration memo). Every feature below is framed as
intensity/dispersion, not alpha direction.

For each filing we aggregate the firm's Form 4 non-derivative transactions over a trailing window
ending 3 days before the filing date (Form 4s are public within 2 business days, so the window is
deploy-realistic and leakage-free):

  f4_n_trades          log1p count of all transactions in the window
  f4_n_insiders        log1p distinct reporting owners trading in the window
  f4_net_dir_cnt       (P - S) / (P + S) by trade count      (open-market codes P/S only)
  f4_net_dir_val       (P - S) / (P + S) by dollar value
  f4_disagreement      fraction of trading insiders on the minority P/S side (dispersion)
  f4_officer_sell_frac share of officer P/S trades that are sales
  f4_abn_intensity     log ratio of the window trade rate to the firm's trailing-365d rate
  f4_opp_frac          fraction of P/S trades that are opportunistic (Cohen-Malloy-Pomorski 2012:
                       routine = same insider traded the same calendar month in >=3 prior years)

A zero-trade window for a firm with Form 4 history is a valid observation (counts 0, direction
ratios NaN); firms absent from the Form 4 universe get all-NaN rows.

Source: datasets/form4/form4_transactions.csv — SEC insider-transactions structured flat files
(collection documented in dataset_collection_discussion/data_collection_readme.md; 1.67M rows,
645 S&P 500 firms, 2006-2025).

Filings are matched to issuers by CIK first (stable across ticker renames — the P0-a lesson),
with a ticker fallback for rows lacking a CIK.

Output: datasets/form4_features.parquet, keyed (ticker, filing_date).
Run (after form4/ is downloaded):
    python dataset_config/build_form4_features.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
FORM4_CSV = DATA / "form4" / "form4_transactions.csv"
# P0-a: prefer the corrected filing<->label join (same convention as phase5/eval_common.py)
_FT_FIXED = DATA / "feature_table_fixed.parquet"
FEATURE_TABLE = _FT_FIXED if _FT_FIXED.exists() else DATA / "feature_table.parquet"
OUT_PATH = DATA / "form4_features.parquet"

WINDOW_DAYS = 180        # trailing aggregation window before the filing
DISCLOSURE_LAG_DAYS = 3  # Form 4s file within 2 business days; end the window 3d before filing
BASELINE_DAYS = 365      # baseline rate window for abnormal intensity (ends at window start)
ROUTINE_MIN_YEARS = 3    # CMP 2012: routine trade = same insider, same calendar month, >=3 prior years

FEAT_COLS = ["f4_n_trades", "f4_n_insiders", "f4_net_dir_cnt", "f4_net_dir_val",
             "f4_disagreement", "f4_officer_sell_frac", "f4_abn_intensity", "f4_opp_frac"]


def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(("1", "true", "1.0", "yes"))


def load_transactions():
    tx = pd.read_csv(FORM4_CSV, usecols=["ticker", "issuer_cik", "rptownercik", "is_officer",
                                         "transaction_date", "transaction_code", "shares",
                                         "price_per_share"],
                     dtype={"transaction_code": str}, low_memory=False)
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    tx = tx.dropna(subset=["transaction_date", "issuer_cik"]).copy()
    tx["issuer_cik"] = tx["issuer_cik"].astype("int64")
    tx["ticker"] = tx["ticker"].astype(str).str.upper().str.strip()
    tx["code"] = tx["transaction_code"].str.upper().str.strip()
    tx["is_officer"] = _truthy(tx["is_officer"])
    tx["value"] = (pd.to_numeric(tx["shares"], errors="coerce").clip(lower=0)
                   * pd.to_numeric(tx["price_per_share"], errors="coerce").clip(lower=0))
    return tx.sort_values("transaction_date").reset_index(drop=True)


def routine_history(tx):
    """(issuer_cik, rptownercik, month) -> set of years with >=1 open-market P/S trade."""
    ps = tx[tx["code"].isin(("P", "S"))]
    hist = defaultdict(set)
    for icik, ocik, d in zip(ps["issuer_cik"], ps["rptownercik"], ps["transaction_date"]):
        hist[(icik, ocik, d.month)].add(d.year)
    return hist


def window_features(g, filing_date, hist, icik):
    """Features over (filing_date - WINDOW_DAYS, filing_date - DISCLOSURE_LAG_DAYS] for one firm's
    transactions g (sorted by date). Returns a dict (zero-trade windows are valid observations)."""
    w_end = filing_date - pd.Timedelta(days=DISCLOSURE_LAG_DAYS)
    w_start = filing_date - pd.Timedelta(days=WINDOW_DAYS)
    dates = g["transaction_date"].to_numpy()
    lo, hi = np.searchsorted(dates, np.datetime64(w_start), side="right"), \
        np.searchsorted(dates, np.datetime64(w_end), side="right")
    w = g.iloc[lo:hi]

    # abnormal intensity: window rate vs the firm's rate over the year ending at the window start
    b_lo = np.searchsorted(dates, np.datetime64(w_start - pd.Timedelta(days=BASELINE_DAYS)),
                           side="right")
    n_base = lo - b_lo
    expected = n_base * (WINDOW_DAYS - DISCLOSURE_LAG_DAYS) / BASELINE_DAYS
    rec = {
        "f4_n_trades": float(np.log1p(len(w))),
        "f4_n_insiders": float(np.log1p(w["rptownercik"].nunique())),
        "f4_abn_intensity": float(np.log((len(w) + 1.0) / (expected + 1.0))),
        "f4_net_dir_cnt": np.nan, "f4_net_dir_val": np.nan, "f4_disagreement": np.nan,
        "f4_officer_sell_frac": np.nan, "f4_opp_frac": np.nan,
    }

    ps = w[w["code"].isin(("P", "S"))]
    if len(ps):
        buys, sells = (ps["code"] == "P"), (ps["code"] == "S")
        rec["f4_net_dir_cnt"] = float((buys.sum() - sells.sum()) / len(ps))
        val = ps["value"].fillna(0.0)
        tot = float(val.sum())
        if tot > 0:
            rec["f4_net_dir_val"] = float((val[buys].sum() - val[sells].sum()) / tot)
        # dispersion: classify each insider by their net side in the window; minority fraction
        net = ps.assign(sgn=np.where(buys, 1, -1)).groupby("rptownercik")["sgn"].sum()
        b, s = int((net > 0).sum()), int((net < 0).sum())
        if b + s > 0:
            rec["f4_disagreement"] = min(b, s) / (b + s)
        off = ps[ps["is_officer"]]
        if len(off):
            rec["f4_officer_sell_frac"] = float((off["code"] == "S").mean())
        opp = [len([y for y in hist.get((icik, o, d.month), ()) if y < d.year])
               < ROUTINE_MIN_YEARS
               for o, d in zip(ps["rptownercik"], ps["transaction_date"])]
        rec["f4_opp_frac"] = float(np.mean(opp))
    return rec


def main():
    if not FORM4_CSV.exists():
        raise SystemExit(f"{FORM4_CSV} not found — download form4/ first.")
    tx = load_transactions()
    print(f"Loaded {len(tx):,} transactions | {tx.issuer_cik.nunique()} issuers | "
          f"{tx.transaction_date.min().date()}..{tx.transaction_date.max().date()}")
    hist = routine_history(tx)

    ft = pd.read_parquet(FEATURE_TABLE)[["ticker", "cik", "filing_date"]].copy()
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["ticker"] = ft["ticker"].str.upper().str.strip()

    # CIK join first (survives ticker renames), ticker fallback
    by_cik = {int(c): g.reset_index(drop=True) for c, g in tx.groupby("issuer_cik")}
    by_tkr = {t: g.reset_index(drop=True) for t, g in tx.groupby("ticker")}
    tkr_cik = tx.groupby("ticker")["issuer_cik"].first().to_dict()

    out = []
    for r in ft.itertuples():
        rec = {"ticker": r.ticker, "filing_date": r.filing_date}
        g, icik = by_cik.get(int(r.cik)), int(r.cik)
        if g is None:
            g, icik = by_tkr.get(r.ticker), tkr_cik.get(r.ticker)
        if g is not None:
            rec.update(window_features(g, r.filing_date, hist, icik))
        out.append(rec)

    res = pd.DataFrame(out)
    res.to_parquet(OUT_PATH, index=False)
    matched = res["f4_n_trades"].notna()
    active = matched & (res["f4_n_trades"] > 0)
    print(f"\nSaved {OUT_PATH}  ({len(res):,} filings, {int(matched.sum()):,} in the Form 4 "
          f"universe, {int(active.sum()):,} with window activity)")
    print("Coverage (non-null fraction):")
    print(res[FEAT_COLS].notna().mean().to_string())
    print("\nFilings per filing year (universe-matched / window-active):")
    yr = res["filing_date"].dt.year
    tab = pd.DataFrame({"filings": yr.value_counts(), "matched": yr[matched].value_counts(),
                        "active": yr[active].value_counts()})
    print(tab.fillna(0).astype(int).sort_index().to_string())
    print("\nFiscal-2020 sanity (medians should move vs other years):")
    chk = res.assign(y=yr).groupby("y")[["f4_abn_intensity", "f4_disagreement"]].median()
    print(chk.loc[chk.index.isin(range(2018, 2023))].to_string())


if __name__ == "__main__":
    main()
