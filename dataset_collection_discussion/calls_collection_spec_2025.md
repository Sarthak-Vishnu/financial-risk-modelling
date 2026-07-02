# Earnings-call collection spec — 2025 pre-filing set (confirmation step)

## Why
The combined call-anchored gate (`phase5/call_combined_gate.py`) showed earnings-call **tone adds
+0.039 IC** on top of the full structured block (≈ the +0.041 that 10-K text adds in Stage D) — but the
calls already collected **postdate** the 10-Ks, so they can't be used filing-anchored. To confirm at the
filing level we need, for each 2025 10-K, the **annual / Q4 earnings call that PRECEDES it**.

This is the *small* confirmation collection (406 calls). Backfill 2006–2024 only AFTER this confirms.

## Exactly which calls (target list)
`datasets/calls_to_collect_2025.csv` — one row per 2025 filing, columns:

| column | meaning |
|---|---|
| ticker, cik | firm |
| fiscal_year | fiscal year the 10-K covers (348 are FY2024, 58 are FY2025) |
| filing_date | the 10-K filing date (prediction anchor) |
| **request_year** | year to request = `fiscal_year` |
| **request_quarter** | `4` (the annual / Q4 call that reports the fiscal year) |
| window_start, window_end | the call's `date` should fall in this window (≈ filing_date−100d … filing_date−1d); use it to verify the returned call truly precedes the filing |

Target = the Q4 call for `fiscal_year`, which for a Dec year-end firm is announced ~late-Jan/Feb of the
filing year, i.e. just before the 10-K. (We already hold 274 call files; the script will skip any already
present in the window — only fetch the gaps.)

## How (API Ninjas — same source as the existing calls)
Endpoint: `GET https://api.api-ninjas.com/v1/earningstranscript`
Header: `X-Api-Key: <YOUR_KEY>`
Params: `ticker=<TICKER>&year=<request_year>&quarter=4`

Response fields to KEEP (match the existing schema exactly so the parser works unchanged):
`date, ticker, cik, year, quarter, transcript`

Save path (mirror current layout):
`datasets/calls/firm/json/<ticker_lower>/<request_year>/calls_<TICKER>_<request_year>_4.json`

### Validation before saving each file
- The returned `date` must be **< filing_date** and **≥ window_start** (a call dated after the filing is
  leakage — discard it). The CSV gives the window per ticker.
- `transcript` length > 200 chars (skip empty/placeholder responses).
- 58 FY2025 firms (non-Dec year-ends, filed mid-2025) may not have a Q4-2025 call yet — those will simply
  return empty; that's expected, skip them.

## After collection
```bash
python dataset_config/build_call_features.py   # now matches at the filing level (expect ~300+ matches)
python phase5/run_fusion.py                     # add the +calls row to the val-2025 ladder
```
Decision gate: if `structured + ... + calls` beats the no-calls model on val-2025, commit to the
2006–2024 backfill; otherwise stop here with the documented call-anchored evidence.
