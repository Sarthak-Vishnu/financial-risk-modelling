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

# Pin every thread pool to 1 — see the note in phase5/run_val2025.sh. The tree-head lanes here
# (structured [hgb], struct+calls [hgb]) are the ones that move without it; the ridge and sparse
# lanes are stable either way. Without pinning, a re-run can land ~0.007 IC away from the Stage C
# table in this repository while reaching the same verdict.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python phase5/run_fusion.py
