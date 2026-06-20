#!/bin/bash
#SBATCH --job-name=finmteb_eval
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/finmteb_%j.out
#SBATCH --error=logs/finmteb_%j.err

# Phase 3 — Increment 3: FinMTEB intrinsic evaluation.
# Scores our encoders + baselines on FinMTEB English STS + Retrieval.
# Runs in the `finmteb` env (NOT `diss`). Requires contrastive_checkpoints/{dual,three,three_lora}.

source ~/.bashrc
conda activate finmteb
cd /home/s2880814/financial-risk-modelling
export PYTHONUNBUFFERED=1

echo "=== FinMTEB: our encoders + SBERT + DAPT-only (English STS + Retrieval) ==="
python contrastive/eval_finmteb.py --tasks all --batch_size 64

# --- Optional: Fin-E5 (7B) baseline -----------------------------------------
# Fin-E5 needs last-token pooling + instructions, so it is scored via the repo's
# own runner (heavy: 7B download + GPU). Uncomment to include it; results land in
# ~/FinMTEB/results and are merged into the comparison manually.
# echo "=== Fin-E5 (7B) via FinMTEB runner ==="
# python ~/FinMTEB/eval_FinanceMTEB.py --model_name_or_path FinanceMTEB/FinE5 --pooling_method last

echo "=== DONE. Summary -> eval_results/finmteb/summary.json ==="
