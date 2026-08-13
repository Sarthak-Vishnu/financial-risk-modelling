#!/bin/bash
#SBATCH --job-name=enc3
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/encode_three_%j.out
#SBATCH --error=logs/encode_three_%j.err

# The two contrastive encoders run_encode.sh never covered: its --encoders default is
# "dual,sbert,volaware,ftvol,bge", so `three` and `three_lora` were never encoded against the
# P0-corrected corpus. Without emb_<enc>_fixed.npy, mean_pooled_filings() returns None and the
# stress grid skips them silently — the project's own three-view contribution has no downstream
# score on corrected data while dual (the replication it extends) does.
#
# Both checkpoints are full merged SentenceTransformer saves (three_lora's LoRA adapters were
# merged at save time — no adapter_config.json, 438 MB model.safetensors, Transformer/Pooling/
# Normalize modules identical to dual), so no special loading path is needed.
#
# Cached emb_<enc>_fixed.npy are skipped automatically, so this is idempotent and cannot touch
# the five caches that already exist. The per-encoder loop also means a cache saved before a
# wall-time kill survives: resubmitting picks up where this left off.
#
# GPU: RTX 2080 Ti rather than the h200_1g.18gb slice the other five caches were built on. saxa
# has idle slices but zero free CPUs, so a slice there cannot be driven. The encode is fp32 either
# way (encode_paragraphs.py calls model.encode() with no autocast and has no --fp16 flag), so the
# only cross-GPU difference is floating-point summation order, order 1e-6 on the embeddings.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

# one python process per encoder: a silent GPU-side kill (seen with volaware, job 3526854, which
# died at 54% with exit 0 and no traceback) must not take the other encoder down with it.
for ENC in three three_lora; do
    echo "--- encoding $ENC ---"
    date +"    start %F %T"
    python topics/encode_paragraphs.py --encoders "$ENC" --batch_size 64 \
        || echo "--- $ENC FAILED (exit $?) — continuing ---"
    date +"    end   %F %T"
done

echo "=== DONE. Caches -> topics/out/emb_three_fixed.npy, emb_three_lora_fixed.npy ==="
echo "=== Next: sbatch phase5/run_val2025.sh ==="
