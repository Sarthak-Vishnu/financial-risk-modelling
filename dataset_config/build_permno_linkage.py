# build_permno_linkage.py
# Joins CCM linking table with sp500_union_constituents to produce
# a clean ticker -> CIK -> PERMNO mapping saved as permno_linkage.csv
#
# Run: python scripts/build_permno_linkage.py

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
CCM_PATH         = r"D:\UoE AI\Dissertation\IPP Draft\datasets\ccm_linking_table.csv"
CONSTITUENTS_PATH= r"D:\UoE AI\Dissertation\IPP Draft\datasets\sp500_union_constituents(1).csv"
OUT_PATH         = r"D:\UoE AI\Dissertation\IPP Draft\datasets\permno_linkage.csv"

# ── 1. Load CCM linking table ──────────────────────────────────────────────────
print("Loading CCM linking table...")
ccm = pd.read_csv(CCM_PATH, dtype=str)
ccm.columns = ccm.columns.str.lower().str.strip()
print(f"  Raw rows: {len(ccm):,}  |  Columns: {list(ccm.columns)}")

# ── 2. Clean CIK column ────────────────────────────────────────────────────────
# CIK is zero-padded string e.g. "0000320193" -> 320193
ccm = ccm[ccm['cik'].notna() & (ccm['cik'].str.strip() != '')]
ccm['cik_int'] = ccm['cik'].str.strip().str.lstrip('0').replace('', '0').astype(int)
print(f"  After dropping empty CIK: {len(ccm):,} rows")

# ── 3. Clean LPERMNO ──────────────────────────────────────────────────────────
ccm['lpermno'] = pd.to_numeric(ccm['lpermno'], errors='coerce')
ccm = ccm[ccm['lpermno'].notna()]
ccm['lpermno'] = ccm['lpermno'].astype(int)

# ── 4. Prefer best link per CIK ───────────────────────────────────────────────
# Priority: (a) LINKPRIM == P  (b) still active (LINKENDDT == 'E')  (c) latest LINKDT

# Flag active links (LINKENDDT = 'E' means no end date = still valid)
ccm['is_active'] = (ccm['linkenddt'].str.upper().str.strip() == 'E').astype(int)

# Flag primary links
ccm['is_primary'] = (ccm['linkprim'].str.upper().str.strip() == 'P').astype(int)

# Sort so best link floats to top per CIK
ccm = ccm.sort_values(
    ['cik_int', 'is_primary', 'is_active', 'linkdt'],
    ascending=[True, False, False, False]
)

# Keep best link per CIK (first row after sort)
ccm_best = ccm.drop_duplicates(subset='cik_int', keep='first')
print(f"  After deduplication (best link per CIK): {len(ccm_best):,} rows")

# ── 5. Load S&P 500 constituents ──────────────────────────────────────────────
print("\nLoading S&P 500 constituents...")
constituents = pd.read_csv(CONSTITUENTS_PATH, dtype=str)
constituents.columns = constituents.columns.str.lower().str.strip()
# Columns: cik, symbol
constituents['cik_int'] = constituents['cik'].str.strip().astype(int)
print(f"  Constituents: {len(constituents):,} rows")

# ── 6. Join ────────────────────────────────────────────────────────────────────
print("\nJoining constituents with CCM...")
merged = constituents.merge(
    ccm_best[['cik_int', 'lpermno', 'lpermco', 'sic', 'tic', 'gvkey',
              'linktype', 'linkprim', 'linkdt', 'linkenddt']],
    on='cik_int',
    how='left'
)

# ── 7. Report coverage ────────────────────────────────────────────────────────
total     = len(merged)
matched   = merged['lpermno'].notna().sum()
unmatched = merged['lpermno'].isna().sum()

print(f"\n  Total tickers in constituents : {total}")
print(f"  Matched with PERMNO           : {matched} ({matched/total*100:.1f}%)")
print(f"  Unmatched (no PERMNO found)   : {unmatched}")

if unmatched > 0:
    print("\n  Unmatched tickers:")
    print(merged[merged['lpermno'].isna()][['symbol', 'cik']].to_string(index=False))

# ── 8. Save ────────────────────────────────────────────────────────────────────
out = merged[['symbol', 'cik_int', 'lpermno', 'lpermco', 'sic',
              'gvkey', 'linktype', 'linkprim', 'linkdt', 'linkenddt']].copy()
out.columns = ['ticker', 'cik', 'permno', 'permco', 'sic_ccm',
               'gvkey', 'linktype', 'linkprim', 'linkdt', 'linkenddt']

out.to_csv(OUT_PATH, index=False)
print(f"\nSaved -> {OUT_PATH}")
print(out.head(10).to_string(index=False))
