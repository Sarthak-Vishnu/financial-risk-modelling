"""
#17 Horizon robustness — volatility labels at 20/60/90 trading days.

Same definition as the study's 30-day labels (compute_volatility_labels.py), at three more
horizons: vol = std(daily log-returns, ddof=1) over the H trading days strictly before / strictly
after the filing date, annualised by sqrt(252). Prices come from the same two sources the
structured features use (build_market_features.load_prices: FINSABER 2000-2024 + CRSP 2025).

The 30-day labels in feature_table_fixed.parquet are NEVER touched — they are the study's frozen
anchor. This script instead recomputes H=30 as a VALIDATION lane and reports agreement with the
stored labels (certifying this reimplementation before any 20/60/90 number is trusted), then
saves only the new horizons.

Output: datasets/volatility_labels_horizons.parquet, keyed (ticker, filing_date), columns
lagged_vol_{20,60,90}d / fwd_vol_{20,60,90}d.
Run:  python dataset_config/compute_horizon_labels.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_market_features import load_prices

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"
_FT_FIXED = DATA / "feature_table_fixed.parquet"
FEATURE_TABLE = _FT_FIXED if _FT_FIXED.exists() else DATA / "feature_table.parquet"
OUT_PATH = DATA / "volatility_labels_horizons.parquet"

HORIZONS = [20, 60, 90]
ANNUALIZE = math.sqrt(252)
TOL = 1e-5  # stored 30d labels are rounded to 6dp
STALE_DAYS = np.timedelta64(10, "D")  # window must touch the filing date within this margin


def window_vols(dates, rets, filing_date, horizon):
    """(lagged, fwd) over the `horizon` trading days strictly before/after filing_date;
    NaN where fewer than `horizon` days exist. Mirrors compute_volatility_labels.compute_vol,
    plus a staleness guard: if the firm's price history ends before (or starts after) the filing,
    the nearest `horizon` returns are not the filing-adjacent window — no label, not a stale one."""
    d = np.datetime64(filing_date)
    lo = np.searchsorted(dates, d, side="left")    # dates[:lo]  < filing_date
    hi = np.searchsorted(dates, d, side="right")   # dates[hi:]  > filing_date
    lagged = (float(np.std(rets[lo - horizon:lo], ddof=1)) * ANNUALIZE
              if lo >= horizon and d - dates[lo - 1] <= STALE_DAYS else np.nan)
    fwd = (float(np.std(rets[hi:hi + horizon], ddof=1)) * ANNUALIZE
           if len(rets) - hi >= horizon and dates[hi] - d <= STALE_DAYS else np.nan)
    return lagged, fwd


def main():
    px = load_prices()
    series = {t: (g["date"].to_numpy(dtype="datetime64[ns]"), g["log_ret"].to_numpy(dtype=float))
              for t, g in px.groupby("ticker")}
    print(f"Price series: {len(series):,} tickers")

    ft = pd.read_parquet(FEATURE_TABLE)[["ticker", "filing_date",
                                         "lagged_vol_30d", "fwd_vol_30d"]].copy()
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["ticker"] = ft["ticker"].str.upper().str.strip()

    out, chk30 = [], []
    for r in ft.itertuples():
        rec = {"ticker": r.ticker, "filing_date": r.filing_date}
        s = series.get(r.ticker)
        for h in HORIZONS:
            lag, fwd = window_vols(*s, r.filing_date, h) if s else (np.nan, np.nan)
            rec[f"lagged_vol_{h}d"], rec[f"fwd_vol_{h}d"] = lag, fwd
        out.append(rec)
        l30, f30 = window_vols(*s, r.filing_date, 30) if s else (np.nan, np.nan)
        chk30.append((l30, f30))

    res = pd.DataFrame(out)

    # ---- validation lane: recomputed H=30 must agree with the stored (frozen) labels ----
    c = pd.DataFrame(chk30, columns=["lag30", "fwd30"])
    print("\nH=30 reimplementation check vs stored labels (rows where both sides exist;")
    print("tickers without cluster price data — the external-fallback pair — drop out here):")
    for new, old in [("lag30", "lagged_vol_30d"), ("fwd30", "fwd_vol_30d")]:
        both = c[new].notna() & ft[old].notna().reset_index(drop=True)
        diff = c.loc[both, new].to_numpy() - ft.reset_index(drop=True).loc[both, old].to_numpy()
        within = float(np.mean(np.abs(diff) <= TOL))
        corr = float(np.corrcoef(c.loc[both, new], ft.loc[both, old])[0, 1])
        print(f"  {old:15s} n={int(both.sum()):,}  corr={corr:.6f}  "
              f"within {TOL:g}: {within:.4f}  max|diff|={float(np.abs(diff).max()):.2e}")
        assert corr > 0.9999, f"{old}: reimplementation disagrees with stored labels"

    res.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}  ({len(res):,} filings)")

    print("\nCoverage (non-null fwd label fraction) by filing year:")
    yr = res["filing_date"].dt.year
    cov = pd.DataFrame({f"h{h}": res[f"fwd_vol_{h}d"].notna().groupby(yr).mean()
                        for h in HORIZONS})
    print(cov.round(3).to_string())

    print("\nSanity: cross-horizon correlation of forward labels (should be high but < 1):")
    m = res[[f"fwd_vol_{h}d" for h in HORIZONS]].copy()
    m["fwd_vol_30d"] = ft["fwd_vol_30d"].to_numpy()
    print(m.corr().round(3).to_string())

    print("\nFiscal-2020 sanity (median fwd vol should be elevated at every horizon):")
    med = res.assign(y=yr).groupby("y")[[f"fwd_vol_{h}d" for h in HORIZONS]].median()
    print(med.loc[med.index.isin(range(2018, 2023))].round(3).to_string())


if __name__ == "__main__":
    main()
