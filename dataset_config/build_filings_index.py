# build_filings_index.py
# Queries the EDGAR submissions API for each CIK in sp500_union_constituents
# to extract: SIC code, SIC description, and 10-K filing dates (2006-2025).
#
# Output: datasets/filings_index.csv
# Columns: ticker, cik, sic, sic_description, fiscal_year, filing_date, report_date
#
# Run: python scripts/build_filings_index.py

import os, time, json
import requests
import pandas as pd
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
CONSTITUENTS_PATH = r"D:\UoE AI\Dissertation\IPP Draft\datasets\sp500_union_constituents(1).csv"
PICKLE_DIR        = r"D:\UoE AI\Dissertation\IPP Draft\datasets\sp500_1A"
OUT_PATH          = r"D:\UoE AI\Dissertation\IPP Draft\datasets\filings_index.csv"

# EDGAR requires a descriptive User-Agent — fill in your details
HEADERS = {
    "User-Agent": "University of Edinburgh s2880814@ed.ac.uk",
    "Accept-Encoding": "gzip, deflate",
}

YEAR_MIN = 2006
YEAR_MAX = 2025
SLEEP_BETWEEN_REQUESTS = 0.12   # stay well under EDGAR's 10 req/s limit


# ── Helper: fetch JSON from EDGAR with retry ───────────────────────────────────
def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                print(f"  [rate-limited] sleeping 5s...")
                time.sleep(5)
        except Exception as e:
            print(f"  [error] {e} — retry {attempt+1}")
            time.sleep(2)
    return None


# ── Helper: extract 10-K rows from a filings block ────────────────────────────
def extract_10k_rows(filings_block):
    """
    filings_block is the dict under filings.recent or a fetched older-batch JSON.
    Returns list of dicts with keys: filing_date, report_date, accession.
    """
    rows = []
    forms       = filings_block.get("form", [])
    dates       = filings_block.get("filingDate", [])
    report_dates= filings_block.get("reportDate", [])
    accessions  = filings_block.get("accessionNumber", [])

    for form, fdate, rdate, acc in zip(forms, dates, report_dates, accessions):
        if form in ("10-K", "10-K405"):          # 10-K405 is the pre-2003 variant
            rows.append({
                "filing_date":  fdate,
                "report_date":  rdate,
                "accession":    acc,
            })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load constituents
    constituents = pd.read_csv(CONSTITUENTS_PATH, dtype=str)
    constituents.columns = constituents.columns.str.lower().str.strip()
    # Columns: cik, symbol

    # Build set of (ticker, year) pairs we actually have pickle files for
    pickle_pairs = set()
    for fname in os.listdir(PICKLE_DIR):
        if fname.endswith(".pickle"):
            parts = fname.replace(".pickle", "").rsplit("_", 1)
            if len(parts) == 2:
                pickle_pairs.add((parts[0], int(parts[1])))

    records = []
    errors  = []

    for _, row in tqdm(constituents.iterrows(), total=len(constituents),
                       desc="Querying EDGAR"):
        ticker = row["symbol"].strip()
        cik    = row["cik"].strip()
        cik_padded = cik.zfill(10)

        # ── Fetch main submissions JSON ────────────────────────────────────────
        url  = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        data = fetch_json(url)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if data is None:
            errors.append({"ticker": ticker, "cik": cik, "reason": "fetch_failed"})
            continue

        sic      = data.get("sic", "")
        sic_desc = data.get("sicDescription", "")

        # ── Collect 10-K filings from recent block ─────────────────────────────
        all_10k = extract_10k_rows(data.get("filings", {}).get("recent", {}))

        # ── Fetch older filing batches if they exist ───────────────────────────
        older_files = data.get("filings", {}).get("files", [])
        for older in older_files:
            older_url  = f"https://data.sec.gov/submissions/{older['name']}"
            older_data = fetch_json(older_url)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if older_data:
                all_10k.extend(extract_10k_rows(older_data))

        # ── Filter to YEAR_MIN–YEAR_MAX and match against pickle corpus ────────
        for filing in all_10k:
            rdate = filing["report_date"]
            if not rdate or len(rdate) < 4:
                continue
            fiscal_year = int(rdate[:4])
            if fiscal_year < YEAR_MIN or fiscal_year > YEAR_MAX:
                continue

            records.append({
                "ticker":          ticker,
                "cik":             int(cik),
                "sic":             sic,
                "sic_description": sic_desc,
                "fiscal_year":     fiscal_year,
                "filing_date":     filing["filing_date"],
                "report_date":     filing["report_date"],
                "has_pickle":      (ticker, fiscal_year) in pickle_pairs,
            })

    # ── Save ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    if df.empty:
        print("\n[ERROR] No records collected — check HEADERS / network.")
        return

    # Sort by ticker, fiscal_year
    df = df.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)

    # Summary
    total        = len(df)
    with_pickle  = df["has_pickle"].sum()
    without      = total - with_pickle
    print(f"\nTotal 10-K filings found (2006-2025) : {total}")
    print(f"  With matching pickle file          : {with_pickle}")
    print(f"  No pickle file (filing exists but  ")
    print(f"    text not scraped)                : {without}")

    if errors:
        print(f"\nFailed CIKs ({len(errors)}):")
        for e in errors:
            print(f"  {e}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
