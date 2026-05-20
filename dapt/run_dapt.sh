#!/bin/bash
#SBATCH --job-name=dapt_mpnet
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/dapt_%j.out
#SBATCH --error=logs/dapt_%j.err

source ~/.bashrc
conda activate diss

cd /home/s2880814/financial-risk-modelling

# Step 1: chunk corpus (CPU-only, run once)
if [ ! -f dapt_data/train.jsonl ]; then
    echo "=== Chunking corpus ==="
    python dapt/chunk_corpus.py
else
    echo "=== dapt_data/train.jsonl already exists, skipping chunk step ==="
fi

# Step 2: DAPT training
echo "=== Starting DAPT training ==="
python dapt/train_dapt.py \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 2e-5 \
    --fp16 \
    --early_stopping_patience 2
