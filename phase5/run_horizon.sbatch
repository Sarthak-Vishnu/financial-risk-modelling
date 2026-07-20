#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --job-name=horizon
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/run_horizon.log

# #17 Horizon robustness — label build (with H=30 reimplementation check), then the fair text
# pair at each horizon. H=30 runs FIRST off the frozen study labels and must reproduce the known
# anchors (backtest struct+tfidf 0.561 / dIC +0.016 p=0.098; val-2025 s+tfidf 0.610 under this
# protocol, cf. logs/run_horizon_attempt1.log) before any other horizon is read. CPU only.
# Submit from the repo root:  sbatch phase5/run_horizon.sbatch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diss

cd "$SLURM_SUBMIT_DIR"

echo "==================== compute_horizon_labels ===================="
python dataset_config/compute_horizon_labels.py

echo "==================== run_horizon H=30 (anchor reproduction) ===================="
python phase5/run_horizon.py

echo "==================== run_horizon H=20 ===================="
RISK_HORIZON=20 python phase5/run_horizon.py

echo "==================== run_horizon H=60 ===================="
RISK_HORIZON=60 python phase5/run_horizon.py

echo "==================== run_horizon H=90 ===================="
RISK_HORIZON=90 python phase5/run_horizon.py
