#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --job-name=fusion_calls
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/run_fusion_calls.log

# Stage C — full fusion ladder incl. the new earnings-call rows (CPU only).
# Submit from the repo root:  sbatch phase5/run_fusion_calls.sbatch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diss

cd "$SLURM_SUBMIT_DIR"
python phase5/run_fusion.py
