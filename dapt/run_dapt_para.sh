#!/bin/bash
#SBATCH --job-name=dapt_para
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/dapt_para_%j.out
#SBATCH --error=logs/dapt_para_%j.err

# Paragraph-aware chunking variant of DAPT, for A/B comparison against the
# sentence-aware run (dapt_data/ + dapt_checkpoints/). All hyperparameters are
# identical to run_dapt.sh so the only varying factor is the chunking strategy.

source ~/.bashrc
conda activate diss

cd /home/s2880814/financial-risk-modelling

# Step 1: chunk corpus with paragraph-aware packing (CPU-only, run once)
if [ ! -f dapt_data_para/train.jsonl ]; then
    echo "=== Chunking corpus (paragraph-aware) ==="
    python dapt/chunk_corpus.py --paragraph_aware --out_dir dapt_data_para
else
    echo "=== dapt_data_para/train.jsonl already exists, skipping chunk step ==="
fi

# Step 2: DAPT training on the paragraph-aware chunks
echo "=== Starting DAPT training (paragraph-aware) ==="
python dapt/train_dapt.py \
    --data_dir dapt_data_para \
    --checkpoint_dir dapt_checkpoints_para \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 2 \
    --lr 2e-5 \
    --fp16 \
    --early_stopping_patience 2

# Step 3: fill-rate statistics for both strategies
echo ""
echo "=== Fill-rate stats: sentence-aware (dapt_data) ==="
python dapt/chunk_stats.py dapt_data/train.jsonl
echo ""
echo "=== Fill-rate stats: paragraph-aware (dapt_data_para) ==="
python dapt/chunk_stats.py dapt_data_para/train.jsonl

# Step 4: perplexity comparison
# Fair head-to-head: both checkpoints evaluated on the SAME (original) val set.
echo ""
echo "=== Perplexity on COMMON val set (dapt_data/val.jsonl) — fair comparison ==="
echo "--- sentence-aware checkpoint ---"
python dapt/eval_perplexity.py --model_path dapt_checkpoints/best      --val_file dapt_data/val.jsonl
echo "--- paragraph-aware checkpoint ---"
python dapt/eval_perplexity.py --model_path dapt_checkpoints_para/best --val_file dapt_data/val.jsonl

# Each model on its own val set (convergence check, not directly comparable).
echo ""
echo "=== Perplexity on OWN val set (convergence check) ==="
echo "--- sentence-aware on dapt_data/val.jsonl ---"
python dapt/eval_perplexity.py --model_path dapt_checkpoints/best      --val_file dapt_data/val.jsonl
echo "--- paragraph-aware on dapt_data_para/val.jsonl ---"
python dapt/eval_perplexity.py --model_path dapt_checkpoints_para/best --val_file dapt_data_para/val.jsonl

echo ""
echo "=== DONE. Validation perplexity and chunk statistics are reported above. ==="
