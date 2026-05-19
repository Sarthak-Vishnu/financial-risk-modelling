# build_feature_table.py
# Assembles the Phase 1 feature table by joining:
#   filings_index.csv   -> ticker, cik, sic, filing_date, report_date, fiscal_year
#   permno_linkage.csv  -> permno, permco, gvkey
#   volatility_labels.csv -> lagged_vol_30d, fwd_vol_30d
#
# Embedding and topic_vector columns are left as NaN placeholders
# to be filled in Phase 2 (DAPT embeddings) and Phase 4 (BERTopic).
#
# Output: datasets/feature_table.parquet  (+ feature_table_preview.csv for inspection)
# Run:    python scripts/build_feature_table.py

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
FILINGS_PATH  = r"D:\UoE AI\Dissertation\IPP Draft\datasets\filings_index.csv"
PERMNO_PATH   = r"D:\UoE AI\Dissertation\IPP Draft\datasets\permno_linkage.csv"
VOL_PATH      = r"D:\UoE AI\Dissertation\IPP Draft\datasets\volatility_labels.csv"
OUT_PARQUET   = r"D:\UoE AI\Dissertation\IPP Draft\datasets\feature_table.parquet"
OUT_CSV       = r"D:\UoE AI\Dissertation\IPP Draft\datasets\feature_table_preview.csv"


# ── 1. Load filings (pickle only — these are the rows we model) ───────────────
print("Loading filings index...")
filings = pd.read_csv(FILINGS_PATH)
filings = filings[filings['has_pickle'] == True].copy()
filings['filing_date'] = pd.to_datetime(filings['filing_date'])
filings['report_date'] = pd.to_datetime(filings['report_date'])
filings['cik'] = filings['cik'].astype(int)
print(f"  Filings with pickle : {len(filings):,}")


# ── 2. Load PERMNO linkage ────────────────────────────────────────────────────
print("\nLoading PERMNO linkage...")
permno = pd.read_csv(PERMNO_PATH)
permno.columns = permno.columns.str.lower().str.strip()
permno['cik'] = permno['cik'].astype(int)

# Keep only the columns we need
permno = permno[['cik', 'permno', 'permco', 'gvkey', 'linktype', 'linkprim']].copy()
permno['permno'] = pd.to_numeric(permno['permno'], errors='coerce')

# Deduplicate by CIK — some CIKs appear twice (dual-class shares e.g. GOOGL/GOOG,
# BRK.A/BRK.B). Keep the row with a valid PERMNO; if both valid, keep first.
permno = permno.sort_values('permno', na_position='last').drop_duplicates(
    subset='cik', keep='first'
)
print(f"  PERMNO rows (dedup) : {len(permno):,}")


# ── 3. Load volatility labels ─────────────────────────────────────────────────
print("\nLoading volatility labels...")
vol = pd.read_csv(VOL_PATH)
vol['filing_date'] = pd.to_datetime(vol['filing_date'])
vol['cik'] = vol['cik'].astype(int)
vol = vol[['ticker', 'cik', 'filing_date', 'lagged_vol_30d', 'fwd_vol_30d']].copy()
print(f"  Vol rows            : {len(vol):,}")


# ── 4. Join everything ────────────────────────────────────────────────────────
print("\nJoining tables...")

# Start from filings as the spine
df = filings[['ticker', 'cik', 'sic', 'sic_description',
              'fiscal_year', 'filing_date', 'report_date']].copy()

# Join PERMNO on CIK
df = df.merge(permno, on='cik', how='left')

# Join vol on ticker + CIK + filing_date (3-key join avoids duplicates from
# dual-class shares that share a CIK but file under different ticker rows)
df = df.merge(vol, on=['ticker', 'cik', 'filing_date'], how='left')


# ── 5. Add Phase 2/4 placeholder columns ──────────────────────────────────────
# These will be populated in later phases:
#   embedding     -> dense risk embedding from DAPT BERT   (Phase 2)
#   topic_vector  -> BERTopic topic exposure vector        (Phase 4)
df['embedding']    = np.nan   # placeholder: will become list/array per row
df['topic_vector'] = np.nan   # placeholder: will become list/array per row


# ── 6. Column order & types ───────────────────────────────────────────────────
col_order = [
    'ticker', 'cik', 'permno', 'permco', 'gvkey',
    'sic', 'sic_description',
    'fiscal_year', 'filing_date', 'report_date',
    'lagged_vol_30d', 'fwd_vol_30d',
    'embedding', 'topic_vector',
    'linktype', 'linkprim',
]
df = df[col_order].sort_values(['ticker', 'filing_date']).reset_index(drop=True)

# Cast types
df['cik']         = df['cik'].astype(int)
df['fiscal_year'] = df['fiscal_year'].astype(int)
df['sic']         = df['sic'].astype(str)
df['permno']      = pd.to_numeric(df['permno'], errors='coerce')


# ── 7. Summary ────────────────────────────────────────────────────────────────
total      = len(df)
has_permno = df['permno'].notna().sum()
has_both_vol = (df['lagged_vol_30d'].notna() & df['fwd_vol_30d'].notna()).sum()

print(f"\nFeature table shape : {df.shape}")
print(f"Total rows          : {total:,}")
print(f"With PERMNO         : {has_permno:,}  ({has_permno/total*100:.1f}%)")
print(f"With both vol       : {has_both_vol:,}  ({has_both_vol/total*100:.1f}%)")
print(f"Date range          : {df['filing_date'].min().date()} -> {df['filing_date'].max().date()}")
print(f"Unique tickers      : {df['ticker'].nunique()}")
print(f"Unique SIC codes    : {df['sic'].nunique()}")

print("\nColumn dtypes:")
print(df.dtypes)


# ── 8. Save ───────────────────────────────────────────────────────────────────
# Parquet: compact, fast, preserves types (primary output)
df.drop(columns=['embedding', 'topic_vector']).to_parquet(OUT_PARQUET, index=False)

# CSV preview: human-readable, without placeholder columns
df.drop(columns=['embedding', 'topic_vector']).to_csv(OUT_CSV, index=False)

print(f"\nSaved -> {OUT_PARQUET}")
print(f"Saved -> {OUT_CSV}  (preview without placeholder cols)")
print("\nFirst 10 rows:")
print(df[['ticker', 'cik', 'permno', 'sic', 'fiscal_year',
          'filing_date', 'lagged_vol_30d', 'fwd_vol_30d']].head(10).to_string(index=False))
