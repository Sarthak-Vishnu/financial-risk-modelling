# Hybrid Topic and Domain-Adaptive Modelling for Financial Risk and Forecasting

**Student:** S2880814 | **Supervisor:** Prof Tiejun Ma | **Institution:** University of Edinburgh, School of Informatics

---

## Environment Setup

### 1. Create and activate the conda environment
```bash
conda create -n diss python=3.11 -y
conda activate diss
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Dataset Download

Datasets are stored on a private HuggingFace repo. You need the `hf_...` token to access them.

### 1. Login to HuggingFace
```bash
huggingface-cli login
# paste your hf_... token when prompted
```

### 2. Download all files
```bash
huggingface-cli download SarthakVishnu/dissertation-dataset \
    --repo-type dataset \
    --local-dir ./datasets/

# Unzip the Item 1A corpus
cd datasets/
tar -xzf sp500_1A.tar.gz && rm sp500_1A.tar.gz
cd ..
```

### Expected structure after download
```
datasets/
├── sp500_1A/                  # 8,247 pickle files — Item 1A text corpus
├── feature_table.parquet      # Phase 1 master table (8,105 filings)
├── filings_index.csv          # EDGAR filing dates + SIC codes
├── volatility_labels.csv      # 30-day lagged + forward vol labels
└── permno_linkage.csv         # CIK → PERMNO mapping
```

---

## Running the pipeline

The pipeline runs in stages, each with the entry point below. `MODEL_CARD.md` (Section 8)
gives the full configuration and the reproducibility controls (seeds, thread pinning).

| Stage | Entry point |
|-------|-------------|
| 1. Chunk the Item 1A corpus for MLM | `python dapt/chunk_corpus.py --paragraph_aware --out_dir dapt_data_para` |
| 2. Domain-adaptive pretraining (DAPT) | `sbatch dapt/run_dapt_para.sh` |
| 3. Build contrastive pairs | `contrastive/run_build_pairs.sh` |
| 4. Contrastive fine-tuning | `contrastive/run_contrastive.sh` |
| 5. Vol-aware contrastive encoder | `contrastive/run_volaware.sh` |
| 6. Supervised volatility fine-tune | `contrastive/run_finetune_vol.sh`, `contrastive/run_ftvol2018.sh` |
| 7. Encode the corpus into filing vectors | `topics/run_encode.sh`, `topics/run_encode_three.sh` |
| 8. Fit BERTopic models | `topics/run_topics.sh` |
| 9. Intrinsic evaluation (FinMTEB) | `contrastive/run_finmteb.sh` |
| 10. Downstream grid + expanding-window backtests | `phase5/run_stress.sh` |
| 10b. Downstream grid only (single forward split) | `phase5/run_val2025.sh` |

DAPT-stage detail is in `dapt/README_dapt.md`. To regenerate the result figures, run each
`graphs/fig*.py`, which writes a vector PDF and a PNG.

> **Reproducibility.** Training uses seed 42. The downstream grid reproduces bit-for-bit only
> with every thread pool pinned to 1 on a fixed CPU; `phase5/run_val2025.sh` enforces this.
> See `MODEL_CARD.md` Section 8.
