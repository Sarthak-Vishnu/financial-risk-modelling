# investigate_unmatched.py
# Looks up the 5 unmatched tickers in CCM by ticker symbol
# to find their PERMNO via alternative routes.
#
# Run: python scripts/investigate_unmatched.py

import pandas as pd

CCM_PATH = r"D:\UoE AI\Dissertation\IPP Draft\datasets\ccm_linking_table.csv"

ccm = pd.read_csv(CCM_PATH, dtype=str)
ccm.columns = ccm.columns.str.lower().str.strip()

UNMATCHED = {
    'ETN':  1551182,
    'ACV':  1636289,
    'PARA':  813828,
    'DAY':  1725057,
    'CPWR':  827099,
}

print("=" * 70)
print("LOOKUP BY TICKER SYMBOL in CCM")
print("=" * 70)
for ticker, cik in UNMATCHED.items():
    hits = ccm[ccm['tic'].str.upper().str.strip() == ticker]
    print(f"\n[{ticker}]  our CIK={cik}")
    if len(hits):
        print(hits[['gvkey','tic','cik','lpermno','linkprim','linktype','linkdt','linkenddt']].to_string(index=False))
    else:
        print("  -> NOT found by ticker either")

print("\n" + "=" * 70)
print("LOOKUP BY CIK (partial / parent entity search)")
print("=" * 70)
for ticker, cik in UNMATCHED.items():
    cik_str = str(cik).zfill(10)
    hits = ccm[ccm['cik'].str.strip() == cik_str]
    print(f"\n[{ticker}]  CIK={cik_str}")
    if len(hits):
        print(hits[['gvkey','tic','cik','lpermno','linkprim','linktype','linkdt','linkenddt']].to_string(index=False))
    else:
        print("  -> NOT found by exact CIK either")
