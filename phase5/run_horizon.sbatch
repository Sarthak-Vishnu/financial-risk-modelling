#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --job-name=horizon
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/run_horizon_spectrum.log

# Window spectrum 3-90 trading days (Prof Ma, 2026-08-04) — extends #17 from four horizons to
# eight, and puts the earnings-call and Form 4 extensions on the same window axis so they can be
# compared directly. CPU only. Submit from the repo root:
#   sbatch phase5/run_horizon.sbatch
#
# Pre-registered gates. H=30 runs FIRST off the frozen study labels, and NO other window is read
# until all four of these reproduce:
#   run_horizon.py            struct 0.546 | s+tfidf 0.561 | dIC +0.016 p=0.098
#                             val-2025 struct 0.591 | s+tfidf 0.610
#   run_horizon.py calls      struct+calls -0.004 p=0.395 | struct+tfidf+calls -0.001 p=0.436
#   form4_text_conditioning   uncond +0.016 p=0.098 | abn_intensity -0.012 p=0.037
#                             | disagreement +0.029 p=0.088
#   call_combined_gate.py     structured + tone +0.039 over structured
# The label build self-certifies separately (recomputed H=30 vs the frozen labels, corr > 0.9999).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diss

cd "$SLURM_SUBMIT_DIR"

WINDOWS="3 5 7 10 20 60 90"

echo "==================== compute_horizon_labels (3/5/7/10/20/60/90) ===================="
python dataset_config/compute_horizon_labels.py

echo "==================== ANCHOR GATE: H=30 ===================="
echo "-------- run_horizon H=30 --------"
python phase5/run_horizon.py
echo "-------- form4_text_conditioning H=30 --------"
python phase5/form4_text_conditioning.py

echo "==================== 10-K + calls, filing anchor ===================="
for H in $WINDOWS; do
    echo "-------- run_horizon H=$H --------"
    RISK_HORIZON=$H python phase5/run_horizon.py
done

echo "==================== Form 4 text-increment conditioning ===================="
for H in $WINDOWS; do
    echo "-------- form4_text_conditioning H=$H --------"
    RISK_HORIZON=$H python phase5/form4_text_conditioning.py
done

echo "==================== calls, call anchor (pilot, n~152) ===================="
for W in 30 $WINDOWS; do
    echo "-------- call_combined_gate W=$W --------"
    RISK_CALL_WINDOW=$W python phase5/call_combined_gate.py
done

echo "==================== tabulation ===================="
grep -h "^SUMMARY" logs/run_horizon_spectrum.log || true
