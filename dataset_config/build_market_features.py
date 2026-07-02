"""
Stage A — structured market/price features per filing (the strong quant baseline).

For every filing in feature_table.parquet we compute, from daily prices in a TRAILING window
ending strictly BEFORE filing_date (so they are known at prediction time), the standard
cross-sectional predictors of forward volatility:

  realised vol      rvol_21 / rvol_63 / rvol_126 / rvol_252   (annualised std of log-returns)
  vol dynamics      vol_of_vol (std of rolling-21d vol, 252d), rvol_ratio_21_252 (term structure)
  distribution      ret_skew_252, ret_kurt_252, max_drawdown_252
  momentum          ret_21 / ret_63 / ret_252  (cumulative trailing log-return)
  market            beta_252, idio_vol_252      (vs equal-weight panel market)
  liquidity*        adv_usd_252 (avg dollar volume), amihud_252, turnover_252   *needs volume

Mirrors the price-loading + windowing logic in compute_volatility_labels.py. Volume-dependent
features are emitted only if the price source carries a volume column; otherwise they are NaN.

Output: datasets/market_features.parquet, keyed by (ticker, filing_date), one row per filing.
Merge into the panel with build_feature_table.py (or directly in the Phase-5 harness).

Run (after the two price CSVs are on the cluster under datasets/):
    python dataset_config/build_market_features.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
FEATURE_TABLE = DATA / "feature_table.parquet"
# HF repo keeps price CSVs under raw/; tolerate either raw/ or a flat datasets/ layout.
FINSABER_PATH = next((p for p in [DATA / "raw" / "all_sp500_prices_2000_2024_delisted_include.csv",
                                  DATA / "all_sp500_prices_2000_2024_delisted_include.csv"] if p.exists()),
                     DATA / "raw" / "all_sp500_prices_2000_2024_delisted_include.csv")
CRSP_PATH = next((p for p in [DATA / "raw" / "crsp_2025_daily.csv",
                              DATA / "crsp_2025_daily.csv"] if p.exists()),
                 DATA / "raw" / "crsp_2025_daily.csv")
OUT_PATH = DATA / "market_features.parquet"

ANNUALIZE = math.sqrt(252)
HORIZONS = [21, 63, 126, 252]
LONG = 252  # trailing window for distribution / market / liquidity features


# ── price loading ──────────────────────────────────────────────────────────────
def _first_col(df, candidates):
    """Return the first column in `candidates` present in df, else None."""
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_prices():
    """Per-ticker DataFrame(index=date) with columns log_ret and (optional) dollar_vol."""
    frames = []

    # FINSABER 2000-2024 (daily OHLCV)
    if FINSABER_PATH.exists():
        head = pd.read_csv(FINSABER_PATH, nrows=1)
        vcol = _first_col(head, ["volume", "adjusted_volume", "vol"])
        ccol = _first_col(head, ["close", "adjusted_close", "adj_close"])
        usecols = ["date", "symbol", "adjusted_close"] + ([vcol] if vcol else [])
        if ccol and ccol not in usecols:
            usecols.append(ccol)
        fin = pd.read_csv(FINSABER_PATH, usecols=lambda c: c in usecols)
        fin["date"] = pd.to_datetime(fin["date"])
        fin["ticker"] = fin["symbol"].str.upper().str.strip()
        fin = fin.sort_values(["ticker", "date"])
        fin["log_ret"] = fin.groupby("ticker")["adjusted_close"].transform(lambda x: np.log(x / x.shift(1)))
        if vcol:
            price_for_vol = fin[ccol] if ccol else fin["adjusted_close"]
            fin["dollar_vol"] = fin[vcol].astype(float) * price_for_vol.astype(float)
        else:
            fin["dollar_vol"] = np.nan
        frames.append(fin[["ticker", "date", "log_ret", "dollar_vol"]])
        print(f"FINSABER: {len(fin):,} rows | volume col: {vcol or 'NONE'}")
    else:
        print(f"WARNING: {FINSABER_PATH.name} not found — transfer it to datasets/")

    # CRSP 2025
    if CRSP_PATH.exists():
        head = pd.read_csv(CRSP_PATH, nrows=1)
        vcol = _first_col(head, ["DlyVol", "DlyDollarVol", "vol"])
        pcol = _first_col(head, ["DlyPrc", "DlyClose", "prc"])
        usecols = ["Ticker", "DlyCalDt", "DlyRet"] + [c for c in (vcol, pcol) if c]
        crsp = pd.read_csv(CRSP_PATH, usecols=lambda c: c in usecols)
        crsp = crsp.rename(columns={"Ticker": "ticker", "DlyCalDt": "date", "DlyRet": "dlyret"})
        crsp["date"] = pd.to_datetime(crsp["date"])
        crsp["ticker"] = crsp["ticker"].str.upper().str.strip()
        crsp = crsp[crsp["dlyret"].notna()].copy()
        crsp["log_ret"] = np.log1p(crsp["dlyret"])
        if vcol and pcol:
            crsp["dollar_vol"] = crsp[vcol].abs().astype(float) * crsp[pcol].abs().astype(float)
        elif vcol:
            crsp["dollar_vol"] = crsp[vcol].abs().astype(float)
        else:
            crsp["dollar_vol"] = np.nan
        frames.append(crsp[["ticker", "date", "log_ret", "dollar_vol"]])
        print(f"CRSP 2025: {len(crsp):,} rows | volume col: {vcol or 'NONE'}")
    else:
        print(f"WARNING: {CRSP_PATH.name} not found — 2025 filings will lack recent prices")

    if not frames:
        raise SystemExit("No price sources found. Transfer the CSVs into datasets/ and re-run.")

    px = (pd.concat(frames, ignore_index=True)
          .replace([np.inf, -np.inf], np.nan)
          .dropna(subset=["log_ret"])
          .sort_values(["ticker", "date"])
          .drop_duplicates(["ticker", "date"]))
    return px


def market_index(px):
    """Equal-weight daily market log-return across the whole panel (for beta / idio vol)."""
    return px.groupby("date")["log_ret"].mean().sort_index()


# ── per-filing feature computation ───────────────────────────────────────────────
def features_for(series, dvol, mkt, filing_date):
    """series, dvol: ascending DatetimeIndex up to filing_date. mkt: market return series."""
    r = series[series.index < filing_date]
    out = {}
    if len(r) < 21:
        return out  # not enough history

    for h in HORIZONS:
        w = r.iloc[-h:]
        out[f"rvol_{h}"] = float(w.std()) * ANNUALIZE if len(w) >= max(10, h // 2) else np.nan
        if h in (21, 63, 252):
            out[f"ret_{h}"] = float(w.sum())  # trailing cumulative log-return (momentum)

    long = r.iloc[-LONG:]
    if len(long) >= 60:
        # vol dynamics
        roll21 = r.rolling(21).std().iloc[-LONG:].dropna()
        out["vol_of_vol"] = float(roll21.std()) * ANNUALIZE if len(roll21) > 10 else np.nan
        if out.get("rvol_252"):
            out["rvol_ratio_21_252"] = out["rvol_21"] / out["rvol_252"]
        # distribution
        out["ret_skew_252"] = float(long.skew())
        out["ret_kurt_252"] = float(long.kurt())
        # max drawdown on cumulative log-price
        cum = long.cumsum()
        out["max_drawdown_252"] = float((cum - cum.cummax()).min())
        # market beta + idiosyncratic vol (align on dates)
        m = mkt.reindex(long.index).dropna()
        common = long.index.intersection(m.index)
        if len(common) >= 60:
            x = m.loc[common].to_numpy()
            y = long.loc[common].to_numpy()
            vx = x.var()
            if vx > 0:
                beta = float(np.cov(x, y)[0, 1] / vx)
                out["beta_252"] = beta
                resid = y - beta * x
                out["idio_vol_252"] = float(resid.std()) * ANNUALIZE

    # liquidity (needs dollar volume)
    dv = dvol[dvol.index < filing_date].iloc[-LONG:] if dvol is not None else None
    if dv is not None and dv.notna().sum() >= 60:
        out["adv_usd_252"] = float(dv.mean())
        rl = r.reindex(dv.index)
        amih = (rl.abs() / dv.replace(0, np.nan)).dropna()
        out["amihud_252"] = float(amih.mean()) * 1e6 if len(amih) else np.nan
    return out


def main():
    ft = pd.read_parquet(FEATURE_TABLE)[["ticker", "filing_date"]].copy()
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["tkr"] = ft["ticker"].str.upper().str.strip()
    print(f"Filings to feature: {len(ft):,}")

    px = load_prices()
    mkt = market_index(px)
    ret_by = {t: g.set_index("date")["log_ret"].sort_index() for t, g in px.groupby("ticker")}
    dv_by = {t: g.set_index("date")["dollar_vol"].sort_index() for t, g in px.groupby("ticker")}
    print(f"Tickers in price panel: {len(ret_by):,}")

    rows = []
    miss = 0
    for r in tqdm(ft.itertuples(), total=len(ft), desc="market-features"):
        s = ret_by.get(r.tkr)
        if s is None:
            miss += 1
            rows.append({"ticker": r.ticker, "filing_date": r.filing_date})
            continue
        feat = features_for(s, dv_by.get(r.tkr), mkt, r.filing_date)
        feat.update({"ticker": r.ticker, "filing_date": r.filing_date})
        rows.append(feat)

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    feat_cols = [c for c in out.columns if c not in ("ticker", "filing_date")]
    cov = out[feat_cols].notna().mean().sort_values()
    print(f"\nNo price match for {miss:,} filings")
    print(f"Saved {OUT_PATH}  ({len(out):,} rows, {len(feat_cols)} features)")
    print("Feature coverage (non-null fraction):")
    print(cov.to_string())


if __name__ == "__main__":
    main()
