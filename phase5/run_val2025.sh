#!/bin/bash
#SBATCH --job-name=val2025
#SBATCH --partition=Teaching
#SBATCH --nodelist=damnii07
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/val2025_%j.out
#SBATCH --error=logs/val2025_%j.err

# Re-run ONLY the val-2025 fair grid, to pick up emb_three_fixed.npy / emb_three_lora_fixed.npy
# after topics/run_encode_three.sh. Deliberately does NOT re-run the 2013/2018 backtests, so
# stress_grid_backtest_*_fixed.json (the primary significance lens) are left untouched.
#
# --mode val2025 overwrites stress_grid_val2025_fixed.json in place, so the previous grid is
# copied aside first: without it there is no way to check afterwards whether the pre-existing
# rows reproduced or drifted.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1

# Reproducibility settings. The committed grid (phase5/out/stress_grid_val2025_fixed.json) is the
# one the write-up quotes, and it regenerates exactly only under BOTH of these controls: every
# thread pool pinned to 1, and the job constrained to damnii07 via --nodelist above. Relax either
# and the tree-head and SVD lanes move enough to reorder encoders. Pinning costs wall-time
# (2 cores -> 1 effective); keep both as they are.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Guard: both new caches must exist, or the grid would silently skip the encoders this re-run
# exists to add and produce a result that looks complete but is not. Refuse rather than mislead.
MISSING=0
for C in topics/out/emb_three_fixed.npy topics/out/emb_three_lora_fixed.npy; do
    if [ ! -f "$C" ]; then echo "MISSING: $C"; MISSING=1; fi
done
if [ "$MISSING" = "1" ]; then
    echo "=== ABORTING: run topics/run_encode_three.sh first (check its log for a failed encoder) ==="
    exit 1
fi

PREV=phase5/out/stress_grid_val2025_fixed.PREV.json
if [ -f phase5/out/stress_grid_val2025_fixed.json ] && [ ! -f "$PREV" ]; then
    cp phase5/out/stress_grid_val2025_fixed.json "$PREV"
    echo "=== previous grid preserved -> $PREV ==="
else
    echo "=== $PREV already exists (or no grid to back up) — leaving it alone ==="
fi

echo "=== val-2025 fair grid + everything model ==="
date +"start %F %T"
python phase5/stress_grid.py --mode val2025
date +"end   %F %T"

echo "=== DONE. Diff against $PREV. ==="
