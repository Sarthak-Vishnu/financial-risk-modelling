# GDELT Data Collection — README
**Project:** Hybrid Topic and Domain-Adaptive Modelling for Financial Risk and Forecasting
**Student:** S2880814 | **Supervisor:** Prof Tiejun Ma
**Dataset:** GDELT (Global Database of Events, Language, and Tone)

---

## What GDELT Is

GDELT is a real-time open data platform that monitors the world's news media — broadcast, print, and web — across 215 languages. Every article is processed through NLP to extract:
- **Who** is mentioned (organisations, persons, locations)
- **What tone** the article takes (positive, negative, neutral — scored numerically)
- **What themes** appear (economic, financial, political, environmental — coded categories)

For this dissertation, GDELT is used as a **company-specific news sentiment signal** — aggregated daily per S&P 500 company and aligned by PERMNO for joining with volatility labels and SEC filings.

---

## Why GDELT (not RavenPack or Bloomberg)

RavenPack is the standard financial-research news sentiment dataset and is available via WRDS. However:

> **University of Edinburgh's WRDS subscription does not include RavenPack.**
> Confirmed: the WRDS page (`wrds-www.wharton.upenn.edu/pages/get-data/ravenpack-news-analytics-40/`) returns *"Sorry, you do not have access to this content"* for this account.

GDELT is the closest freely available alternative. It is used extensively in financial research (e.g., Calomiris & Mamaysky 2019; Ke, Kelly & Xiu 2019 use similar news-sentiment approaches). It covers the same news sources and provides equivalent tone/sentiment signals.

---

## GDELT Dataset Variants

| Variant | What it contains | Update cadence | Coverage |
|---------|-----------------|----------------|----------|
| **GDELT 1.0 GKG** | Per-article organisations + tone + themes. Daily ZIP files. | Daily (by 6 AM EST) | Apr 2013 → present |
| **GDELT 2.0 GKG** | Richer version of 1.0 GKG — enhanced tone (V2.1Tone), enhanced organisation offsets, 65 translated languages. | Every 15 minutes | Feb 2015 → present |
| GDELT 1.0 Events | Structured political events (Actor → action → Actor, CAMEO codes). No per-article tone. | Daily | 1979 → present |
| GDELT 2.0 Events | Enhanced Events, same structure. | Every 15 minutes | Feb 2015 → present |

**This project uses:** GDELT 1.0 GKG (direct download) and GDELT 2.0 GKG (BigQuery). Events tables are not used — they encode political events, not company-level financial sentiment.

---

## Why 1.0 GKG for Direct Download and 2.0 GKG for BigQuery

This is not an arbitrary split — it is dictated by how GDELT publishes each version.

**GDELT 1.0 GKG** publishes one file per day:
```
http://data.gdeltproject.org/gkg/YYYYMMDD.gkg.csv.zip   ← 1 file/day
```
For 2025–2026 (~527 days) = **527 files to download**. Entirely manageable.

**GDELT 2.0 GKG** publishes one file every 15 minutes:
```
http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.gkg.csv.zip   ← 96 files/day
```
For 2025–2026 (~527 days) = **~50,592 files to download**. Completely impractical as a direct download — it would require 50,000+ HTTP requests and hours of sequential fetching just to retrieve the file list, before any filtering begins.

**BigQuery solves this for 2.0 GKG** — it stores all 50,592+ files internally and exposes the entire dataset as a single queryable table. You write SQL, BigQuery scans all 50K files server-side, and returns only the filtered rows you need. The 15-minute fragmentation is invisible to the user.

| | Direct Download | BigQuery |
|---|---|---|
| GDELT version | **1.0 GKG** (daily files) | **2.0 GKG** (15-min files, queried as one table) |
| Files to handle | 527 | ~50,592 (handled by BigQuery internally) |
| Practical? | Yes | Yes via SQL — impossible via direct download |
| Requires account? | No | Yes (Google Cloud + billing) |

This is why BigQuery is not just an alternative route to the same data — it is the **only practical route to GDELT 2.0 GKG** for any historical date range.

---

## Key Fields Extracted

From the **GKG Organizations** and **Tone** fields:

| Output column | Source GKG field | Meaning |
|---------------|-----------------|---------|
| `v1_organizations` | `ORGANIZATIONS` (col 6) | Semicolon-delimited list of all organisations mentioned in the article |
| `tone` | `TONE` (col 7), field 1 | Overall tone. Negative = bad news, Positive = good news. Range typically −10 to +10 |
| `tone_pos` | `TONE` field 2 | Density of positive language (0–100) |
| `tone_neg` | `TONE` field 3 | Density of negative language (0–100) |
| `polarity` | `TONE` field 4 | Absolute tonal difference (how strongly positive OR negative) |
| `v1_themes` | `THEMES` (col 3) | Semicolon-delimited GDELT theme codes (e.g. `ECON_STOCKMARKET`, `MANMADE_DISASTER`) |
| `source_name` | `SOURCES` (col 9) | News publication domain (e.g. `reuters.com`, `bloomberg.com`) |
| `source_url` | `SOURCEURLS` (col 10) | Full article URL |

**Aggregated daily features** (after step 3):

`gdelt_gkg_daily.parquet` — GKG-specific, richest output:

| Column | Meaning |
|--------|---------|
| `permno` | CRSP PERMNO — links to volatility labels and prices |
| `article_date` | Date of article (YYYY-MM-DD) |
| `n_articles` | Number of articles mentioning this company that day |
| `n_sources` | Distinct news outlets |
| `avg_tone` | Mean overall tone across articles |
| `avg_tone_pos` | Mean positive sentiment score |
| `avg_tone_neg` | Mean negative sentiment score |
| `avg_polarity` | Mean polarity score |
| `n_econ_themes` | Articles with at least one `ECON_` theme code |
| `n_fin_themes` | Articles with finance-specific themes (`ECON_STOCKMARKET`, `ECON_MNA`, etc.) |

`gdelt_daily.parquet` — combined output (GKG + Events union, future use only):

| Column | Meaning |
|--------|---------|
| `permno` | CRSP PERMNO |
| `date` | Date (YYYY-MM-DD) |
| `n_records` | Number of articles/events mentioning this company that day |
| `n_sources` | Distinct news outlets |
| `avg_tone` | Mean overall tone |
| `avg_tone_pos` | Mean positive sentiment score |
| `avg_tone_neg` | Mean negative sentiment score |
| `source` | Data source (`GKG` or `EVENTS`) |

> **Which file to use:** Use `gdelt_gkg_daily.parquet` for all pipeline work. `gdelt_daily.parquet` is a deliberately slimmed-down version — it drops `avg_polarity`, `avg_wordcount`, `n_econ_themes`, and `n_fin_themes` to create a common schema for a future merge with GDELT 1.0 Events data (2006–2014). Since Events data has not been collected yet, `gdelt_daily.parquet` currently contains identical rows to `gdelt_gkg_daily.parquet` but with fewer columns and no additional processing. It can be ignored until historical Events data is added.

---

## Company Name Matching — How PERMNO Assignment Works

GDELT does not use CIK or PERMNO. Companies appear by their name as written in news articles, normalised by GDELT's TABARI system: ALL CAPS, no `INC`/`CORP`, punctuation removed (e.g. `APPLE`, `JOHNSON JOHNSON`, `AT T`).

CRSP `dsenames.csv` provides the bridge:
- For each PERMNO, `comnam` gives the company name valid between `namedt` and `nameendt`
- Each CRSP name is cleaned to GDELT-style: `APPLE INC` → `APPLE`, `JOHNSON & JOHNSON` → `JOHNSON JOHNSON`
- 30+ manual overrides handle edge cases: `AMAZON COM INC` → `AMAZON`, `H P INC` → `HEWLETT PACKARD`, `K K R & CO LP` → `KKR`

**Temporal matching:** An article from 2025-03-15 mentioning `ORACLE` is only assigned to PERMNO 10104 if `namedt ≤ 2025-03-15 ≤ nameendt` for the `ORACLE` entry in dsenames. Companies with `nameendt = 2024-12-31` (end of WRDS data extract, not end of company existence) are extended to a far-future date so 2025–2026 articles match correctly.

Output of matching: **800 unique patterns** across 646 PERMNOs with temporal validity windows → `datasets/gdelt/company_names.csv`.

---

## Approach 1 — Direct Download (GDELT 1.0 GKG)

### What it is

GDELT publishes all GKG daily files at:
`http://data.gdeltproject.org/gkg/YYYYMMDD.gkg.csv.zip`

One file per day, tab-delimited, 11 columns, ~15–35 MB compressed. No account, no API key, no cost.

### Coverage collected

| Year | Days downloaded | Days missing (server gaps) | Rows kept (S&P 500 mentions) |
|------|----------------|---------------------------|------------------------------|
| 2025 | 346 / 365 | 18 (normal GDELT gaps) | 4,056,375 |
| 2026 | 162 / 163 | 1 (normal GDELT gap) | ~800,000 (est.) |

### How it works

1. **Step 1** (`step1_prepare_names.py`): Builds `company_names.csv` — 800 GDELT-style patterns with PERMNO and date bounds.
2. **Step 2** (`step2_download_gkg.py`): Downloads each daily ZIP into RAM, extracts CSV, filters rows where `ORGANIZATIONS` contains any S&P 500 pattern (regex match), saves filtered rows as parquet. ZIP deleted immediately — never written to disk.
3. **Step 3** (`step3_aggregate_daily.py`): Reads all daily parquets, assigns PERMNOs via temporal name matching, aggregates to daily per-PERMNO feature rows.

### Storage

| What | Where | Size |
|------|-------|------|
| Daily filtered parquets (2025–2026) | `datasets/gdelt/raw/gkg_YYYYMMDD.parquet` | ~500 MB total (508 files) |
| Final daily features | `datasets/gdelt/gdelt_gkg_daily.parquet` | ~3 MB |
| Peak RAM during download | *(in-memory ZIP extraction)* | ~400 MB per file |
| C drive usage | None — all files written to D drive | 0 MB |

### Run commands

```powershell
cd "D:\UoE AI\Dissertation\IPP Draft"

# Step 1 (already done — company_names.csv exists)
python scripts/gdelt/step1_prepare_names.py

# Step 2 — download and filter
python scripts/gdelt/step2_download_gkg.py --year 2025
python scripts/gdelt/step2_download_gkg.py --year 2026

# Step 3 — PERMNO matching + daily aggregation
python scripts/gdelt/step3_aggregate_daily.py
```

Resume is automatic — already-downloaded dates are skipped.

### Results (as of 2026-06-13)

Step 3 completed successfully. Final aggregated output:

| Metric | Value |
|--------|-------|
| Rows in `gdelt_gkg_daily.parquet` | 170,153 |
| Rows in `gdelt_daily.parquet` (combined) | 170,153 |
| Unique PERMNOs covered | 543 / 646 (84%) |
| Date range | 2025-01-01 → 2026-06-10 |
| Raw daily parquet files | 508 |
| Total raw article-company rows (pre-aggregation) | ~4.8M |

The 103 unmatched PERMNOs are companies with short, ambiguous, or rarely news-mentioned names that do not reliably appear in GDELT's `ORGANIZATIONS` field. Coverage is sufficient for a pilot analysis.

### Limitations

- GDELT 1.0 GKG has simpler organisation name extraction than 2.0 (no character offsets)
- 18 missing days in 2025 (server-side GDELT gaps — normal, not a script error)
- Tone scores are article-level averages, not financial-event-specific

---

## Approach 2 — Google BigQuery (GDELT 2.0 GKG)

### What it is

GDELT 2.0 GKG is hosted as a public dataset on Google BigQuery at:
`gdelt-bq.gdeltv2.gkg`

Queried using SQL — filter to S&P 500 company names, extract tone and organisation fields, download results. Richer than 1.0 GKG: enhanced tone (V2.1Tone), 65 translated languages, 15-minute update cadence.

### Cost

| Tier | Allowance | Cost |
|------|-----------|------|
| Free tier | 1 TB/month query processing | £0 |
| 2025 + 2026 (~18 months) | ~540 GB total | **£0 — within free tier** |
| Overage | Beyond 1 TB/month | $5/TB (~£4/TB) |

### Hurdle — Credit Card Required

Even though the query cost is £0, **Google requires a billing account (credit card) to be linked** before BigQuery will process any job — including free-tier queries against public datasets. The credit card is for identity verification and to enforce the 1 TB/month limit; you are not charged within the free tier.

> **This is the primary reason direct download (Approach 1) was collected first** — no card required, no account setup, identical data for 2025–2026.

If a billing account is added, the $300 free trial credit Google provides on new accounts far exceeds any realistic GDELT query cost.

### Setup (one-time, ~10 minutes)

```
1. console.cloud.google.com → Create project: "gdelt-dissertation"
   (auto-assigned project ID, e.g. seismic-relic-499300)

2. Search "BigQuery API" → Enable

3. IAM & Admin → Service Accounts → Create
   Name: gdelt-reader
   Roles: BigQuery Job User + BigQuery Data Viewer
   Keys tab → Add Key → JSON → download key file

4. Activate billing (credit card required):
   Top banner → "Start free trial" → enter card details
   (Not charged — $300 free credit + 1 TB/month free BigQuery tier)

5. PowerShell:
   $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\Users\HP\Downloads\<key-file>.json"
```

### Run commands

```powershell
cd "D:\UoE AI\Dissertation\IPP Draft"

# Dry run — confirms GB estimate and cost before running (no charges)
python scripts/gdelt/step2_collect_gkg.py --year 2025 --dry-run

# Actual collection — monthly parquets saved to datasets/gdelt/raw/
python scripts/gdelt/step2_collect_gkg.py --year 2025
python scripts/gdelt/step2_collect_gkg.py --year 2026

# Aggregation (same step3 script — works for both approaches)
python scripts/gdelt/step3_aggregate_daily.py
```

### Output

Monthly parquet files: `datasets/gdelt/raw/gkg_YYYY_MM.parquet`
(vs. daily parquets `gkg_YYYYMMDD.parquet` from the direct download approach)

**Status as of 2026-06-13:** Not yet collected — pending billing account activation (credit card). GCP project `seismic-relic-499300` created, BigQuery API enabled, service account and JSON key generated. Only the billing step remains.

---

## Comparison: Direct Download vs BigQuery

| | Direct Download (Approach 1) | BigQuery (Approach 2) |
|---|---|---|
| GDELT version | 1.0 GKG | 2.0 GKG |
| Account required | None | Google Cloud account |
| Credit card required | No | **Yes** (even for free queries) |
| Cost | Free | Free within 1 TB/month |
| Data granularity | Daily files | Monthly files |
| Tone field | V1.5Tone (7 components) | V2.1Tone (richer, same components + more) |
| Organisation field | Simple semicolon list | Enhanced with character offsets |
| Language coverage | English-dominant | 65 languages |
| Coverage (current) | 2025–2026 | 2015–present |
| Output format | `gkg_YYYYMMDD.parquet` (daily) | `gkg_YYYY_MM.parquet` (monthly) |
| Resume support | Yes | Yes |
| Step 3 compatible | Yes | Yes (same script) |

**Current status:**
- Approach 1 (Direct Download): **DONE** — `gdelt_daily.parquet` produced, 170,153 rows, 543 PERMNOs, 2025-01-01 → 2026-06-10
- Approach 2 (BigQuery): **PENDING** — GCP project ready, blocked on billing account activation (credit card)

**For supervisor decision:** Both approaches produce a daily `(permno, date, avg_tone, n_articles, ...)` feature table via step 3. The BigQuery version is richer (2.0 GKG, 65 languages) but requires credit card setup. For 2025–2026 English-language financial news, both approaches should yield near-identical coverage. The key question for the supervisor is whether to extend GDELT historically (2006–2024) — BigQuery is the only feasible route for that scale.

---

## Directory Structure

```
datasets/
├── gdelt/
│   ├── company_names.csv              ← step 1 output: 800 patterns + PERMNO + date bounds
│   ├── gdelt_gkg_daily.parquet        ← step 3 output: daily per-PERMNO features (GKG)
│   ├── gdelt_events_daily.parquet     ← step 3 output: daily per-PERMNO features (Events, if collected)
│   ├── gdelt_daily.parquet            ← step 3 output: combined GKG + Events
│   └── raw/
│       ├── gkg_20250101.parquet       ← direct download: one file per day
│       ├── gkg_20250102.parquet
│       ├── ...
│       ├── gkg_2025_01.parquet        ← BigQuery: one file per month
│       ├── gkg_2025_02.parquet
│       └── ...
```

---

## Scripts Reference

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `scripts/gdelt/step1_prepare_names.py` | `crsp_dsenames.csv` | `gdelt/company_names.csv` | Cleans CRSP company names to GDELT-style patterns; applies 30+ manual overrides; outputs 800 patterns with PERMNO + temporal bounds |
| `scripts/gdelt/step2_download_gkg.py` | `gdelt/company_names.csv`, GDELT website | `gdelt/raw/gkg_YYYYMMDD.parquet` | **Direct download approach.** Downloads daily GKG ZIPs into RAM, filters for S&P 500 companies, saves filtered rows. Resume-safe. |
| `scripts/gdelt/step2_collect_gkg.py` | `gdelt/company_names.csv`, BigQuery | `gdelt/raw/gkg_YYYY_MM.parquet` | **BigQuery approach.** Queries GDELT 2.0 GKG monthly via SQL; requires GCP project + billing enabled. Dry-run mode estimates cost before executing. |
| `scripts/gdelt/step2b_collect_events.py` | `gdelt/company_names.csv`, BigQuery | `gdelt/raw/events_YYYY_MM.parquet` | **BigQuery — pre-2015 era only.** Queries GDELT 1.0 Events table for 2006–2014 (skip for now — only 2025–2026 collected). |
| `scripts/gdelt/step3_aggregate_daily.py` | `gdelt/raw/*.parquet`, `gdelt/company_names.csv` | `gdelt_gkg_daily.parquet`, `gdelt_daily.parquet` | Matches organisation names to PERMNOs via temporal bounds; aggregates to daily per-PERMNO tone/count features. Works for both direct download and BigQuery parquets. |

---

## Notes on Data Access and Ethics

- **GDELT** is an open, freely available dataset with no redistribution restrictions. Published under an open licence by the GDELT Project (Kalev Leetaru). Available at `gdeltproject.org`.
- **Google BigQuery** access uses a personal Google account — not the University's institutional licence. No University data or credentials are involved.
- **crsp_dsenames.csv** is used internally for name matching only — it is not uploaded to BigQuery or included in any output files. CRSP data remains under the University's institutional licence.
- GDELT output files (`gdelt_daily.parquet`, raw parquets) contain no personally identifiable information and carry no licence restrictions — safe for upload to HuggingFace or sharing with supervisors.
