#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/encode_%j.out
#SBATCH --error=logs/encode_%j.err

# P0-b + E4 — full-text (windowed) re-encode of the corrected paragraph corpus.
# Produces topics/out/emb_<enc>_fixed.npy for the stress grid.
# `bge` requires `hf download BAAI/bge-base-en-v1.5` run on an internet node beforehand;
# if it is missing the script skips it and encodes the rest.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

python topics/encode_paragraphs.py --encoders dual,sbert,volaware,ftvol,bge --batch_size 64

echo "=== DONE. Caches in topics/out/emb_*_fixed.npy ==="
