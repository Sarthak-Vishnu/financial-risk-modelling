#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --job-name=fusion_form4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/run_fusion_form4.log

# Stage E — Form 4 insider features: feature build, fusion ladder (+form4 rows) on both backtest
# windows, then the text-increment conditioning analysis (CPU only).
# Submit from the repo root:  sbatch phase5/run_fusion_form4.sbatch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diss

cd "$SLURM_SUBMIT_DIR"

# Pin every thread pool to 1 — see the note in phase5/run_val2025.sh. The tree-head lanes here
# (structured [hgb], struct+calls [hgb], struct+form4 [hgb]) are the ones that move without it;
# the ridge and sparse lanes are stable either way. Without pinning, a re-run can land ~0.007 IC
# away from the Stage C/E tables in this repository while reaching the same verdict.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "==================== build_form4_features ===================="
python dataset_config/build_form4_features.py

echo "==================== run_fusion (7-year window) ===================="
python phase5/run_fusion.py

echo "==================== run_fusion (12-year window) ===================="
RISK_TEST_START=2013 python phase5/run_fusion.py

echo "==================== form4_text_conditioning (7-year) ===================="
python phase5/form4_text_conditioning.py

echo "==================== form4_text_conditioning (12-year) ===================="
RISK_TEST_START=2013 python phase5/form4_text_conditioning.py
