#!/bin/bash
#SBATCH --job-name=volaware
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200:1
#SBATCH --nodelist=saxa
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/volaware_%j.out
#SBATCH --error=logs/volaware_%j.err

# Stage B — volatility-AWARE contrastive encoder.
# Adds the `vol` view: positives are paragraphs from DIFFERENT firms in the SAME within-year
# forward-vol decile, so embeddings become vol-discriminative (the "financial-aware augmentation"
# the brief asks for) rather than only similarity-smooth. lexical/chrono/sector kept as auxiliary
# views; sector down-weighted (lambda 0.5). Base = the in-domain DAPT encoder.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling
set -e                      # stop on first failure (don't cascade train-crash -> fit_topics)
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on the 18GB slice

# 1) pairs (incl. vol view) are built ONCE by run_build_pairs.sh (CPU). Skip here if present, so a GPU
#    preemption never re-runs the slow CPU pair build and never burns GPU time on it.
if [ -f contrastive/pairs/pairs_vol.jsonl ]; then
    echo "=== pairs present — skipping build (run_build_pairs.sh already made them) ==="
else
    echo "=== build pairs (incl. vol view) — none found, building now ==="
    python contrastive/build_pairs.py
fi

# 2) train the vol-aware encoder
echo "=== train vol-aware contrastive encoder ==="
python contrastive/train_contrastive.py \
    --views lexical,chrono,sector,vol \
    --lambda_sector 0.5 --lambda_vol 1.0 \
    --batch_size 64 \
    --lora --fp16 \
    --out_dir contrastive_checkpoints/volaware

# 3) embed all paragraphs with the new encoder + fit topics (writes emb_volaware.npy +
#    filing_topic_vectors_volaware.parquet for the Phase-5 fusion harness)
echo "=== embed + fit topics for volaware ==="
python topics/fit_topics.py --encoders volaware

echo "=== DONE. Then: python phase5/run_fusion.py  (volaware auto-included) ==="
