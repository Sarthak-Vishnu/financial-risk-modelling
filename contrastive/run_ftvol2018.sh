#!/bin/bash
#SBATCH --job-name=ftvol2018
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/ftvol2018_%j.out
#SBATCH --error=logs/ftvol2018_%j.err

# Clean-protocol retrain of the supervised vol encoder (the decisive stress-test experiment).
# Protocol: train on filing-years < 2017 (P0-corrected corpus + labels), select the epoch on 2017,
# never touch >= 2018 -> the 2018-2024 expanding-window backtest is leakage-free by construction.
# The original ftvol checkpoint trained on ALL pre-2025 labels and epoch-selected on val-2025,
# so it is inadmissible for any backtest and its val-2025 rows are selection-inflated.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

OUT=contrastive_checkpoints/ftvol2018

echo "=== 1/3 fine-tune (train <2017, select on 2017) ==="
python contrastive/finetune_vol.py \
    --train_end 2018 \
    --out_dir "$OUT" \
    --epochs 20 \
    --batch_size 128 \
    --fp16 \
    --skip_emb \
    || { echo "FINETUNE FAILED (exit $?)"; exit 1; }

# the original ftvol save came out without tokenizer/module files (repaired by hand 2026-07-03);
# if it happens again, transplant them from the same-base volaware checkpoint before encoding.
if [ ! -f "$OUT/tokenizer.json" ]; then
    echo "--- tokenizer files missing from checkpoint; copying from volaware (same mpnet base) ---"
    cp contrastive_checkpoints/volaware/{tokenizer.json,tokenizer_config.json,sentence_bert_config.json} "$OUT/"
fi

echo "=== 2/3 windowed full-text encode ==="
python topics/encode_paragraphs.py --encoders ftvol2018 --batch_size 256 \
    || { echo "ENCODE FAILED (exit $?)"; exit 1; }

echo "=== 3/3 leakage-free 2018-2024 backtest with encoder lane ==="
python phase5/stress_grid.py --mode backtest --test-start 2018 --enc ftvol2018 \
    || { echo "BACKTEST FAILED (exit $?)"; exit 1; }

echo "=== DONE. See phase5/out/stress_grid_backtest_2018_fixed.json ==="
