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

## TODO Ahead

Status: the multimodal redesign (`new_plan.md`) is largely done — Stage A (structured baseline beats the
old TF-IDF floor: 0.570 > 0.545), Stage D (text adds on top: `struct+TF-IDF` 0.611 / R²_log 0.276), and
Stage B (vol-aware encoder delivered; clean negative — 0.586 vs similarity-trained 0.602, task is lexically
saturated). Full narrative + numbers in `study.md`. What remains, in priority order:

1. **Topic → volatility interpretability analysis (recommended next — high mark value, no new data).**
   Rank the ~49 BERTopic risk topics by how much each drives the cross-section of forward volatility
   (topic-loading coefficients / per-topic IC); produce a "top volatility-driving risk themes" table with
   each topic's top words. This is the thesis's "predict **and explain**" novelty — the contribution TF-IDF
   cannot provide. Runs on artifacts already on disk (`topics/out/filing_topic_vectors_*.parquet`).

2. **Firm up statistical significance of the headline.** `struct+TF-IDF` beats `structured` by +0.050 IC in
   the 2018–2024 backtest but at p=0.090. Extend test years / add bootstrap CIs + paired tests
   (`eval_common.bootstrap_ci`, `paired_year_test`) to push the incremental-text claim below 0.05 and make
   it bulletproof.

3. **Stage C — earnings-call confirmation (optional upside; costs collection time).** The call-anchored
   gate showed tone adds **+0.039 IC** in combination (`phase5/call_combined_gate.py`). Collect the 406
   pre-filing 2025 calls (`dataset_collection_discussion/calls_collection_spec_2025.md`), rebuild features
   (`build_call_features.py`), re-run `run_fusion.py`; backfill 2006–2024 **only if** they add at the filing
   level. Do this only with schedule slack.

4. **Write-up / consolidation.** Lock the final results tables + figures (ablation ladder, backtest, topic
   interpretability) into the dissertation methods/results. Marks come from the written analysis, not more
   experiments — the science is already a strong, defensible positive.

**Recommended order:** 1 → 2 → (4, start writing) → 3 if time allows.
