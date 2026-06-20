#!/bin/bash
#SBATCH --job-name=contrastive
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/contrastive_%j.out
#SBATCH --error=logs/contrastive_%j.err

# Phase 3 — Increment 2: three-view contrastive fine-tuning.
# Trains three encoders sequentially from dapt_checkpoints_para/best:
#   1. dual-view  (Chiu replication: lexical + chronological)
#   2. three-view (+ sector, soft-positive down-weighted)
#   3. three-view + LoRA (exploratory, Q+V adapters)
# Requires contrastive/pairs/*.jsonl (built by build_pairs.py + validate_pairs.py).

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1   # stream per-epoch progress to the log instead of buffering

echo "=== 1/3 dual-view (lexical + chrono) ==="
python contrastive/train_contrastive.py \
    --views lexical,chrono \
    --out_dir contrastive_checkpoints/dual \
    --batch_size 64 --lr 2e-5 --epochs 30 --fp16

echo "=== 2/3 three-view (+ sector, lambda_sector 0.5) ==="
python contrastive/train_contrastive.py \
    --views lexical,chrono,sector --lambda_sector 0.5 \
    --out_dir contrastive_checkpoints/three \
    --batch_size 64 --lr 2e-5 --epochs 30 --fp16

echo "=== 3/3 three-view + LoRA (Q+V, r=16) ==="
python contrastive/train_contrastive.py \
    --views lexical,chrono,sector --lambda_sector 0.5 --lora --lora_r 16 \
    --out_dir contrastive_checkpoints/three_lora \
    --batch_size 64 --lr 2e-5 --epochs 30 --fp16

echo "=== DONE. Encoders in contrastive_checkpoints/{dual,three,three_lora}/ ==="
