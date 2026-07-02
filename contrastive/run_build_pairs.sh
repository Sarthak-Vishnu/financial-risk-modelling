#!/bin/bash
#SBATCH --job-name=buildpairs
#SBATCH --partition=Teaching
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/buildpairs_%j.out
#SBATCH --error=logs/buildpairs_%j.err

# Build contrastive pairs (lexical / chrono / sector / VOL) ONCE on CPU.
# Decoupled from the GPU training so a GPU preemption never re-runs this slow step, and we never
# waste a GPU slice on CPU-only pair construction. Writes contrastive/pairs/pairs_*.jsonl.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling
export PYTHONUNBUFFERED=1

echo "=== build pairs (lexical/chrono/sector/vol) ==="
python contrastive/build_pairs.py
echo "=== DONE. Now: sbatch contrastive/run_volaware.sh (it will skip the build) ==="
