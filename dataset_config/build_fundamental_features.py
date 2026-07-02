"""
Stage A (fundamentals) — Compustat firm fundamentals per filing.

Standard cross-sectional volatility predictors from annual fundamentals:
  log_market_cap   size            log(mkvalt) or log(prcc_f * csho)   small firms are more volatile
  leverage         lt / at         debt load                          higher leverage -> higher vol
  book_to_market   ceq / mktcap    value vs growth                    value firms behave differently
  roa              ni / at         profitability                      unprofitable firms more volatile

These are known at filing time (fiscal-year-end fundamentals reported in the 10-K), so joining them to
the contemporaneous filing to predict FORWARD vol is leakage-free.

Join path: compustat (gvkey, fiscal_year) -> feature_table (gvkey, fiscal_year) to attach (ticker,
filing_date), so the output shares the SAME key as market_features.parquet.

Output: datasets/fundamental_features.parquet, keyed (ticker, filing_date).
Run (after compustat_fundamentals.csv is on the cluster):
    python dataset_config/build_fundamental_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
# HF repo keeps this under compustat/; tolerate either compustat/ or a flat datasets/ layout.
COMPUSTAT = next((p for p in [DATA / "compustat" / "compustat_fundamentals.csv",
                              DATA / "compustat_fundamentals.csv"] if p.exists()),
                 DATA / "compustat" / "compustat_fundamentals.csv")
FEATURE_TABLE = DATA / "feature_table.parquet"
OUT_PATH = DATA / "fundamental_features.parquet"

FEATS = ["log_market_cap", "leverage", "book_to_market", "roa"]


def _col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    if not COMPUSTAT.exists():
        raise SystemExit(f"{COMPUSTAT} not found — upload it to datasets/ first.")
    cs = pd.read_csv(COMPUSTAT)
    print(f"Compustat: {len(cs):,} rows, cols={list(cs.columns)}")

    gk = _col(cs, "gvkey")
    yr = _col(cs, "fyear", "fiscal_year", "year")
    if yr is None:  # derive from datadate
        dd = _col(cs, "datadate")
        cs["fyear"] = pd.to_datetime(cs[dd]).dt.year
        yr = "fyear"
    mkvalt = _col(cs, "mkvalt")
    prcc = _col(cs, "prcc_f", "prcc")
    csho = _col(cs, "csho")
    lt, at, ceq, ni = _col(cs, "lt"), _col(cs, "at"), _col(cs, "ceq"), _col(cs, "ni")

    # market cap: prefer mkvalt, fall back to price * shares
    mc = cs[mkvalt].astype(float) if mkvalt else pd.Series(np.nan, index=cs.index)
    if prcc and csho:
        alt = cs[prcc].astype(float) * cs[csho].astype(float)
        mc = mc.where(mc.notna() & (mc > 0), alt)
    cs["market_cap"] = mc.where(mc > 0)

    cs["log_market_cap"] = np.log(cs["market_cap"])
    cs["leverage"] = cs[lt].astype(float) / cs[at].astype(float) if lt and at else np.nan
    cs["book_to_market"] = cs[ceq].astype(float) / cs["market_cap"] if ceq else np.nan
    cs["roa"] = cs[ni].astype(float) / cs[at].astype(float) if ni and at else np.nan
    cs = cs.replace([np.inf, -np.inf], np.nan)

    cs["gvkey_i"] = pd.to_numeric(cs[gk], errors="coerce").astype("Int64")
    cs["fy"] = pd.to_numeric(cs[yr], errors="coerce").astype("Int64")
    fund = (cs[["gvkey_i", "fy"] + FEATS]
            .dropna(subset=["gvkey_i", "fy"])
            .drop_duplicates(["gvkey_i", "fy"]))

    ft = pd.read_parquet(FEATURE_TABLE)[["ticker", "filing_date", "gvkey", "fiscal_year"]].copy()
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["gvkey_i"] = pd.to_numeric(ft["gvkey"], errors="coerce").astype("Int64")
    ft["fy"] = pd.to_numeric(ft["fiscal_year"], errors="coerce").astype("Int64")

    out = (ft.merge(fund, on=["gvkey_i", "fy"], how="left")[["ticker", "filing_date"] + FEATS])
    out.to_parquet(OUT_PATH, index=False)
    cov = out[FEATS].notna().mean()
    print(f"\nSaved {OUT_PATH}  ({len(out):,} rows)")
    print("Coverage (non-null fraction):")
    print(cov.to_string())


if __name__ == "__main__":
    main()
