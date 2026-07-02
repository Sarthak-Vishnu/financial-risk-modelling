#!/bin/bash
#SBATCH --job-name=ftvol
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/ftvol_%j.out
#SBATCH --error=logs/ftvol_%j.err

# Lever 3 — supervised fine-tuning of the encoder ON the volatility task.
# Paragraph-level regression (filing's log fwd-vol as label) so embeddings become vol-discriminative,
# not just similarity-smooth. Train < 2025, monitor on 2025 filing-level cross-sectional IC.
# Default base = the in-domain DAPT encoder; try contrastive_checkpoints/three_lora as an alternative.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling
export PYTHONUNBUFFERED=1

echo "=== Lever 3: supervised volatility fine-tune (base = DAPT) ==="
python contrastive/finetune_vol.py \
    --base_model dapt_checkpoints_para/best \
    --epochs 3 --batch_size 64 --lr 2e-5 --fp16

# Then evaluate the fine-tuned embeddings against the TF-IDF floor (IC 0.545):
#   python phase5/improve_pooling.py     (auto-includes 'ftvol')
echo "=== DONE. Best val filing IC printed above; emb -> topics/out/emb_ftvol.npy ==="
