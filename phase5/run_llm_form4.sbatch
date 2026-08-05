#!/bin/bash
#SBATCH --partition=Teaching
#SBATCH --job-name=llmform4
#SBATCH --gres=gpu:h200_1g.18gb:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/run_llm_form4.log

# NOTE ON WALL TIME: 2 hours, deliberately, NOT the 48h used for the CPU jobs. 865 prompts x 5
# samples runs in well under an hour, and the GPU nodes are backfill-scheduled: saxa sits in
# MIXED+PLANNED most of the time (held for a higher-priority job), so a long request never fits
# into the gap ahead of it and the job pends indefinitely with the misleading reason
# "ReqNodeNotAvail, May be reserved for other job". Ask for what the job needs, not for the maximum.
#
# NOTE ON THE SLICE: TWO 1g.18gb slices, not one, and not the 3g.71gb. Reasoning, in order:
#   - The 71GB slices are the contended resource (only 4 exist, with a standing queue). The 18GB
#     slices are 35-strong and typically idle, so they schedule immediately.
#   - ONE 18GB slice is not enough, learned the hard way (job 3584220): a "1g.18gb" slice exposes
#     16.00 GiB usable, and Qwen2.5-7B in bf16 is 15.2 GiB. Worse, transformers 5.x transiently
#     upcasts the largest layer to fp32 while initialising (the 152k x 3584 embedding = 2.03 GiB),
#     so peak load memory is ~17.2 GiB and it OOMs during from_pretrained, before any guard runs.
#   - Two slices give 32 GiB, which clears that peak with room for the KV cache. device_map="auto"
#     places layers across both. MIG instances have no peer-to-peer, so cross-device activations
#     hop through host memory; that is a throughput cost only, and this job is small.
# A MIG slice is a partition of the SAME physical H200, so the arithmetic is identical whichever
# slice size is used — same Hopper tensor cores, same native bf16 path, same kernels.
#
# Do NOT fall back to the 2080Ti nodes just because they are idle: Turing has no native bf16, so
# that run would be fp16, and this pilot's headline output is a measurement of score dispersion —
# changing the numerics would make model sampling noise inseparable from numerical noise.
# The one safe substitution is Qwen/Qwen2.5-3B-Instruct (6.2 GiB, fits a single slice): same
# family and same late-2023 training cutoff, so the post-cutoff argument for the 2024-2025
# evaluation slice survives intact. Use it only if two slices cannot be had.

# Round 8 — bounded generative-LLM pilot (Prof Ma, 2026-08-04). Form 4 windows -> Qwen2.5-7B
# score -> compared against the eight hand-crafted f4_* features under one head and one set of
# folds. Scope is the 2024+2025 filings only, which post-date the model's training cutoff, so
# look-ahead is controlled by construction. See study_extended.md VII.5 for why the scope is not
# negotiable, and phase5/llm_vs_manual.py for the stopping rule.
#
# Submit from the repo root:  sbatch phase5/run_llm_form4.sbatch
# PREREQUISITE: the model must already be in the HF cache. GPU nodes may have no outbound
# internet, so pre-download from the login node first:
#   python -c "from huggingface_hub import snapshot_download; \
#              snapshot_download('Qwen/Qwen2.5-7B-Instruct')"

set -e   # a failed scoring step must not fall through to the evaluation step, which then reports
         # a confusing "scores parquet not found" and buries the real traceback

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diss

cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1                  # fail loudly on an incomplete cache rather than hanging
export HF_HUB_DISABLE_PROGRESS_BARS=1    # tqdm without a TTY writes thousands of log lines
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation on a tight slice

nvidia-smi

echo "==================== build prompts (anonymised) ===================="
python dataset_config/build_form4_llm_prompts.py

echo "==================== score (anonymised, K=5) ===================="
python phase5/llm_score_form4.py

echo "==================== build prompts (identified control) ===================="
RISK_LLM_ANON=0 python dataset_config/build_form4_llm_prompts.py

echo "==================== score (identified control, K=5) ===================="
RISK_LLM_ANON=0 python phase5/llm_score_form4.py

echo "==================== evaluate vs the hand-crafted features ===================="
python phase5/llm_vs_manual.py
