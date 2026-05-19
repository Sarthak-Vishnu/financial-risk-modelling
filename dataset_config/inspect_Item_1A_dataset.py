# inspect_dataset.py
# Quick inspection of the sp500_1A pickle dataset.
# Run in VSCode terminal: python inspect_dataset.py

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, pickle, glob
import pandas as pd
from collections import defaultdict

DATASET_DIR = r"D:\UoE AI\Dissertation\IPP Draft\datasets\sp500_1A"

# --------------------------------------------------------------------------
# STEP 1 -- inspect a few sample files
# --------------------------------------------------------------------------
samples = ["AAPL_2006.pickle", "BAC_2020.pickle", "AMZN_2024.pickle"]

print("=" * 70)
print("STEP 1 -- INSPECT SAMPLE FILES")
print("=" * 70)

for fname in samples:
    fpath = os.path.join(DATASET_DIR, fname)
    if not os.path.exists(fpath):
        print(f"\n[SKIP] {fname} not found.")
        continue

    with open(fpath, "rb") as f:
        obj = pickle.load(f)

    print(f"\n{'-'*60}")
    print(f"File : {fname}")
    print(f"Type : {type(obj)}")

    if isinstance(obj, dict):
        print(f"Keys : {list(obj.keys())}")
        for k, v in obj.items():
            if isinstance(v, str):
                print(f"  [{k}] str  len={len(v)}  preview={v[:150]!r}")
            elif isinstance(v, (list, tuple)):
                print(f"  [{k}] {type(v).__name__}  len={len(v)}  first={str(v[0])[:100]!r}")
            elif isinstance(v, pd.DataFrame):
                print(f"  [{k}] DataFrame  shape={v.shape}  cols={list(v.columns)}")
                print(v.head(3).to_string())
            else:
                print(f"  [{k}] {type(v).__name__}  val={str(v)[:150]}")

    elif isinstance(obj, pd.DataFrame):
        print(f"Shape   : {obj.shape}")
        print(f"Columns : {list(obj.columns)}")
        print(obj.head(3).to_string())

    elif isinstance(obj, str):
        print(f"String len={len(obj)}  preview={obj[:300]!r}")

    elif isinstance(obj, (list, tuple)):
        print(f"Len={len(obj)}  item_type={type(obj[0]).__name__}")
        print(f"First item={str(obj[0])[:300]!r}")

    else:
        print(f"Value : {str(obj)[:300]}")

# --------------------------------------------------------------------------
# STEP 2 -- dataset-wide statistics
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 -- DATASET-WIDE STATISTICS")
print("=" * 70)

all_files = glob.glob(os.path.join(DATASET_DIR, "*.pickle"))
print(f"\nTotal pickle files : {len(all_files)}")

tickers = set()
years = set()
ticker_years = defaultdict(list)

for fp in all_files:
    base = os.path.basename(fp).replace(".pickle", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2:
        ticker, year = parts
        tickers.add(ticker)
        years.add(int(year))
        ticker_years[ticker].append(int(year))

print(f"Unique tickers     : {len(tickers)}")
print(f"Year range         : {min(years)} to {max(years)}")

year_counts = defaultdict(int)
for fp in all_files:
    base = os.path.basename(fp).replace(".pickle", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2:
        year_counts[int(parts[1])] += 1

print("\nFiles per year:")
for yr in sorted(year_counts):
    bar = "#" * (year_counts[yr] // 5)
    print(f"  {yr}: {year_counts[yr]:4d}  {bar}")

print("\nSample tickers and year coverage (first 10):")
for ticker in sorted(tickers)[:10]:
    yr_list = sorted(ticker_years[ticker])
    full = list(range(min(yr_list), max(yr_list) + 1))
    missing = [y for y in full if y not in yr_list]
    print(f"  {ticker:<8} {yr_list}  missing={missing if missing else 'none'}")

# --------------------------------------------------------------------------
# STEP 3 -- temporal split alignment
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 -- TEMPORAL SPLIT (per report design)")
print("=" * 70)

train = [fp for fp in all_files
         if int(os.path.basename(fp).replace(".pickle","").rsplit("_",1)[1]) <= 2024]
val   = [fp for fp in all_files
         if os.path.basename(fp).replace(".pickle","").rsplit("_",1)[1] == "2025"]

print(f"\nTrain (<=2024) : {len(train)} files")
print(f"Val   (2025)   : {len(val)} files")
print(f"Test  (2026)   : not scraped yet")
print("\n[Done]")
