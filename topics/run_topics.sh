#!/bin/bash
#SBATCH --job-name=bertopic
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --nodelist=saxa
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/topics_%j.out
#SBATCH --error=logs/topics_%j.err

# Phase 4 — Increment 2: BERTopic risk-topic extraction.
# For each encoder: embed paragraphs (GPU) -> fit BERTopic on train -> transform val/test ->
# topic table + C_v/diversity + per-filing exposure vectors.
# Requires: topics/data/topic_docs.jsonl (build_topic_docs.py) and `gensim` in the `diss` env
#   (one-time:  conda activate diss && pip install gensim).
# GPU does the encoding; UMAP+HDBSCAN clustering is CPU-heavy (2 cpus is the Teaching cap / 48G).

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling
export PYTHONUNBUFFERED=1   # stream per-encoder progress to the log

# NOTE: sbert already completed in the previous run (cached in topics/out/); skipping it here.
# dapt is DELIBERATELY excluded: its raw MLM embeddings are severely anisotropic
# (mean pairwise cosine 0.60 vs sbert 0.36), which makes UMAP non-convergent — the
# previous run hung ~10h on dapt clustering until the wall-time killed it. DAPT is the
# MLM stage, not a similarity-embedding backend; the contrastive encoders below are.
echo "=== BERTopic over contrastive encoders (dual + three + three_lora) ==="
python topics/fit_topics.py \
    --encoders dual,three,three_lora \
    --min_cluster_size 200 --nr_topics 50

# If topic counts land outside the report's 20-80 band, re-run a single encoder with a tuned
# granularity, e.g.:  python topics/fit_topics.py --encoders three_lora --nr_topics 40
# (cached embeddings make re-runs fast — only clustering repeats.)

echo "=== DONE. Quality -> topics/out/topic_quality.json | vectors -> topics/out/filing_topic_vectors_*.parquet ==="
