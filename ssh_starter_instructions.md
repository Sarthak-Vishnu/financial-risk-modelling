# SSH Cluster — Situation Briefing
> Read this fully before doing anything. This is a catch-up document for a new Claude Code session connected via SSH to the UoE GPU cluster.

---

## Who & What

- **Student:** S2880814 (Sarthak Vishnu), University of Edinburgh, School of Informatics
- **Supervisor:** Prof Tiejun Ma (the PhD scholar assiting Sarthak is Sunnie Li)
- **Project title:** Hybrid Topic and Domain-Adaptive Modelling for Financial Risk and Forecasting
- **IPP report:** `S2880814_IPP20_Report.pdf` is in this directory for reference
- **GitHub repo:** `https://github.com/Sarthak-Vishnu/financial-risk-modelling` (private)
- **HuggingFace dataset repo:** `https://huggingface.co/datasets/SarthakVishnu/dissertation-dataset` (private)

---

## What the Project Does

The project builds a hybrid financial risk model over five phases:

| Phase | Description | Status |
|-------|-------------|--------|
| **1 — Data Pipeline** | Corpus + linkages + volatility labels + feature table | ✅ Complete |
| **2 — DAPT** | Domain-adaptive pretraining of `all-mpnet-base-v2` on Item 1A text | 🔲 Next |
| **3 — Contrastive FT** | Three-view contrastive fine-tuning (lexical, chronological, sector views) | 🔲 Pending |
| **4 — BERTopic** | Topic extraction using domain-adapted encoder | 🔲 Pending |
| **5 — Ablation** | Downstream volatility forecasting + ablation study | 🔲 Pending |

**You are currently at the start of Phase 2.**

---

## Cluster Directory Structure

```
~/dissertation/
├── dataset_config/            # Phase 1 scripts (data pipeline)
│   ├── build_filings_index.py
│   ├── build_permno_linkage.py
│   ├── build_feature_table.py
│   ├── compute_volatility_labels.py
│   ├── inspect_Item_1A_dataset.py
│   ├── investigate_unmatched.py
│   └── upload_datasets_to_hf.py
├── datasets/
│   ├── sp500_1A/              # 8,247 pickle files — Item 1A text corpus
│   ├── feature_table.parquet  # Phase 1 master table (8,105 rows)
│   ├── filings_index.csv      # EDGAR filing dates + SIC codes (11,447 rows)
│   ├── volatility_labels.csv  # 30-day lagged + forward vol labels (8,105 rows)
│   └── permno_linkage.csv     # CIK → PERMNO mapping (656 rows)
├── environment.yml
├── requirements.txt
├── README.md
└── S2880814_IPP20_Report.pdf
```

---

## The Dataset

### sp500_1A/ — Primary text corpus
- 8,247 `.pickle` files, each named `{TICKER}_{YEAR}.pickle`
- Each file is a **plain Python string** containing the full Item 1A (Risk Factors) section of a 10-K filing
- Covers 485 unique S&P 500 firms, fiscal years 2006–2025
- This is the **sole text input** for all five phases

### feature_table.parquet — Phase 1 master table
- 8,105 rows, one per filing with an Item 1A pickle file
- Key columns: `ticker, cik, permno, sic, sic_description, fiscal_year, filing_date, report_date, lagged_vol_30d, fwd_vol_30d`
- `embedding` and `topic_vector` columns exist but are NaN — to be filled in Phase 2 and Phase 4
- **Temporal split** (applied via `filing_date`):
  - Train: `filing_date < 2025-01-01` (~7,700 filings)
  - Val: `2025-01-01 ≤ filing_date < 2026-01-01` (~280 filings)
  - Test: `filing_date ≥ 2026-01-01` (~125 filings)

### volatility_labels.csv — Prediction targets
- `fwd_vol_30d` = **30-day forward realised volatility** — the downstream prediction target throughout Phases 2–5
- `lagged_vol_30d` = 30-day lagged vol — used as the AR(1) baseline feature
- Both are annualised: `std(log-returns over 30 trading days) × √252`
- Coverage: 94.9% of filings have both labels

---

## Key Design Decisions (from IPP report)

1. **Base model:** `all-mpnet-base-v2` (110M params) — chosen because it is the SBERT base substitute in the ablation, keeping architecture constant across all conditions
2. **DAPT:** Continued MLM at 15% masking rate; validation perplexity tracked via W&B; best checkpoint selected on val perplexity
3. **Temporal split is strict:** Training capped at 2024 to prevent leakage from base model pre-training data. Val = 2025, Test = 2026
4. **Three-view contrastive signal (Phase 3):** Lexical view (overlapping spans) + Chronological view (date tokens) + Sector view (same 2-digit SIC, same fiscal year = soft positive; cross-sector = hard negative)
5. **Downstream task:** Predicting `fwd_vol_30d` via Ridge regression and MLP heads; primary metric is Spearman ρ
6. **Evaluation baselines (Phase 5):** SBERT (no adaptation), Fin-E5, DAPT-only, DAPT+contrastive, LoRA-adapted, BERTopic-only, Hybrid

---

## Environment

```bash
# Activate the conda environment
conda activate diss

# Key packages already installed via requirements.txt:
# transformers, sentence-transformers, accelerate, peft,
# bertopic, umap-learn, hdbscan, scikit-learn, wandb,
# pandas, numpy, pyarrow, tqdm, requests
```

---

## What Has NOT Been Done Yet

- Phase 2 DAPT training script (to be written)
- Sentence tokenisation / chunking of Item 1A text for MLM (Phase 2 data prep)
- W&B project setup for perplexity tracking
- Teaching cluster job submission scripts (SLURM or SGE)

---

## Important Notes

- Do **not** modify anything in `datasets/` — treat it as read-only
- WRDS/CRSP-derived data (`permno_linkage.csv`, `volatility_labels.csv`, `feature_table.parquet`) must not be redistributed externally per University licence terms
- `sp500_1A/` source is unconfirmed — shared by a senior colleague; redistribution rights unknown
- The `sanity_check_vol(delete_later).py` script on the local machine is a temporary validation file and is not in the repo
- When writing Phase 2 scripts, save them to `~/dissertation/` under a new directory (e.g., `dapt/`)
