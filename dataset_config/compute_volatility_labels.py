# compute_volatility_labels.py
# Computes lagged_vol_30d and fwd_vol_30d for every filing in filings_index.csv
# where has_pickle=True.
#
# Price sources (in priority order):
#   1. FINSABER all_sp500_prices_2000_2024_delisted_include.csv  (2000-2024)
#   2. WRDS CRSP crsp_2025_daily.csv                             (2025)
#   3. yfinance fallback for ETN / PARA (no PERMNO in CCM)
#
# Volatility definition:
#   vol = std(log_returns over 30 trading days) * sqrt(252)   [annualised]
#   lagged_vol_30d : 30 trading days strictly BEFORE filing_date
#   fwd_vol_30d    : 30 trading days strictly AFTER  filing_date
#
# Output: datasets/volatility_labels.csv
# Run:    python scripts/compute_volatility_labels.py

import math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
FILINGS_PATH  = r"D:\UoE AI\Dissertation\IPP Draft\datasets\filings_index.csv"
FINSABER_PATH = r"D:\UoE AI\Dissertation\IPP Draft\datasets\all_sp500_prices_2000_2024_delisted_include.csv"
CRSP_PATH     = r"D:\UoE AI\Dissertation\IPP Draft\datasets\crsp_2025_daily.csv"
OUT_PATH      = r"D:\UoE AI\Dissertation\IPP Draft\datasets\volatility_labels.csv"

WINDOW    = 30              # trading-day window
ANNUALIZE = math.sqrt(252)  # annualisation factor

# Tickers with no PERMNO in CCM that have Item 1A pickles -> yfinance fallback
YFINANCE_TICKERS = {'ETN', 'PARA'}


# ── Helper ─────────────────────────────────────────────────────────────────────
def compute_vol(series: pd.Series, filing_date: pd.Timestamp, window: int = 30):
    """
    series : pd.Series, DatetimeIndex ascending, values = daily log-returns
    Returns (lagged_vol, fwd_vol) as Python floats; np.nan if < window days.
    """
    # Ensure 1-D Series (guards against yfinance returning DataFrame column)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    dates  = series.index
    before = dates[dates < filing_date]
    after  = dates[dates > filing_date]

    lagged_vol = float(series.loc[before[-window:]].std()) * ANNUALIZE \
                 if len(before) >= window else np.nan
    fwd_vol    = float(series.loc[after[:window]].std()) * ANNUALIZE \
                 if len(after)  >= window else np.nan

    return lagged_vol, fwd_vol


# ── 1. Load filings (pickle only) ─────────────────────────────────────────────
print("Loading filings index...")
filings = pd.read_csv(FILINGS_PATH)
filings = filings[filings['has_pickle'] == True].copy()
filings['filing_date'] = pd.to_datetime(filings['filing_date'])
print(f"  Filings with pickle : {len(filings):,}")


# ── 2. Load FINSABER (2000-2024) ──────────────────────────────────────────────
print("\nLoading FINSABER prices (2000-2024)...")
fin = pd.read_csv(FINSABER_PATH, usecols=['date', 'adjusted_close', 'symbol'])
fin['date']   = pd.to_datetime(fin['date'])
fin['symbol'] = fin['symbol'].str.upper().str.strip()
fin = fin.sort_values(['symbol', 'date'])

# Log-return from adjusted close
fin['log_ret'] = fin.groupby('symbol')['adjusted_close'].transform(
    lambda x: np.log(x / x.shift(1))
)
# Drop NaN and -inf/-inf rows (adjusted_close == 0 produces log(0) = -inf)
fin = fin[fin['log_ret'].notna() & np.isfinite(fin['log_ret'])][['symbol', 'date', 'log_ret']]
fin.rename(columns={'symbol': 'ticker'}, inplace=True)
print(f"  Rows after log-ret  : {len(fin):,}  |  Tickers: {fin['ticker'].nunique()}")


# ── 3. Load CRSP 2025 ─────────────────────────────────────────────────────────
print("\nLoading CRSP 2025 daily...")
crsp = pd.read_csv(CRSP_PATH, usecols=['Ticker', 'DlyCalDt', 'DlyRet'])
crsp.columns = ['ticker', 'date', 'dlyret']
crsp['date']   = pd.to_datetime(crsp['date'])
crsp['ticker'] = crsp['ticker'].str.upper().str.strip()
crsp = crsp[crsp['dlyret'].notna()].copy()
crsp['log_ret'] = np.log(1 + crsp['dlyret'])
crsp = crsp[['ticker', 'date', 'log_ret']]
print(f"  Rows               : {len(crsp):,}  |  Tickers: {crsp['ticker'].nunique()}")


# ── 4. Combine into a single lookup dict ──────────────────────────────────────
print("\nMerging price sources...")
prices = pd.concat([fin, crsp], ignore_index=True)
prices = (prices
          .sort_values(['ticker', 'date'])
          .drop_duplicates(subset=['ticker', 'date']))

# dict: ticker (upper) -> pd.Series(log_ret, index=DatetimeIndex)
price_dict = {
    tkr: grp.set_index('date')['log_ret']
    for tkr, grp in prices.groupby('ticker')
}
print(f"  Unique tickers in price dict: {len(price_dict):,}")


# ── 5. yfinance fallback for ETN / PARA ───────────────────────────────────────
yf_needed = YFINANCE_TICKERS & set(filings['ticker'].str.upper())
if yf_needed:
    import yfinance as yf
    print(f"\nFetching yfinance for: {yf_needed}")
    for tkr in sorted(yf_needed):
        hist = yf.download(tkr, start='2005-01-01', end='2026-06-01',
                           auto_adjust=True, progress=False)
        if hist.empty:
            print(f"  {tkr}: no data returned")
            continue
        # Newer yfinance returns hist['Close'] as a DataFrame; squeeze to Series
        close = hist['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = np.log(close / close.shift(1)).dropna()
        s = s[np.isfinite(s)]          # drop any -inf from zero prices
        s.index = pd.to_datetime(s.index)
        s.index.name = 'date'
        s.name = 'log_ret'
        price_dict[tkr] = s.sort_index()
        print(f"  {tkr}: {len(s):,} rows  ({s.index[0].date()} -> {s.index[-1].date()})")


# ── 6. Compute volatility for every filing ────────────────────────────────────
print("\nComputing volatility windows...")
records = []

for _, row in tqdm(filings.iterrows(), total=len(filings), desc="Vol"):
    ticker       = str(row['ticker']).upper().strip()
    filing_date  = row['filing_date']

    if ticker in price_dict:
        lagged_vol, fwd_vol = compute_vol(price_dict[ticker], filing_date, WINDOW)
    else:
        lagged_vol, fwd_vol = np.nan, np.nan

    records.append({
        'ticker':          row['ticker'],
        'cik':             row['cik'],
        'sic':             row['sic'],
        'sic_description': row['sic_description'],
        'fiscal_year':     row['fiscal_year'],
        'filing_date':     row['filing_date'].date(),
        'report_date':     row['report_date'],
        'lagged_vol_30d':  round(lagged_vol, 6) if pd.notna(lagged_vol) else np.nan,
        'fwd_vol_30d':     round(fwd_vol,    6) if pd.notna(fwd_vol)    else np.nan,
    })

df = pd.DataFrame(records)
df = df.sort_values(['ticker', 'filing_date']).reset_index(drop=True)


# ── 7. Summary ────────────────────────────────────────────────────────────────
total         = len(df)
has_both      = (df['lagged_vol_30d'].notna() & df['fwd_vol_30d'].notna()).sum()
lagged_only   = (df['lagged_vol_30d'].notna() & df['fwd_vol_30d'].isna()).sum()
fwd_only      = (df['lagged_vol_30d'].isna()  & df['fwd_vol_30d'].notna()).sum()
neither       = (df['lagged_vol_30d'].isna()  & df['fwd_vol_30d'].isna()).sum()

print(f"\nTotal filings processed : {total:,}")
print(f"  Both vol computed     : {has_both:,}")
print(f"  Lagged only           : {lagged_only:,}  (fwd window crosses data gap)")
print(f"  Forward only          : {fwd_only:,}")
print(f"  Neither (no prices)   : {neither:,}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
print(df.head(10).to_string(index=False))
