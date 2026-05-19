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
