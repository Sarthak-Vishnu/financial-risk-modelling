#!/bin/bash
#SBATCH --job-name=stress
#SBATCH --partition=Teaching
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/stress_%j.out
#SBATCH --error=logs/stress_%j.err

# E1+E2+E3 in one CPU job on the P0-corrected data. Safe to run before the GPU encode
# (encoder/change-semantic rows auto-skip and fill in on a re-run after run_encode.sh).
# Order: change features -> val-2025 grid -> backtests (7yr + 12yr).

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1

echo "=== E2: disclosure-change features ==="
python dataset_config/build_change_features.py

echo "=== E1: val-2025 fair grid + everything model ==="
python phase5/stress_grid.py --mode val2025

echo "=== E3: backtest 2018-2024 ==="
python phase5/stress_grid.py --mode backtest --test-start 2018

echo "=== E3: backtest 2013-2024 (primary significance lens) ==="
python phase5/stress_grid.py --mode backtest --test-start 2013

echo "=== DONE. Results in phase5/out/stress_grid_*.json — fill phase5/STRESS_TEST_RESULTS.md ==="
