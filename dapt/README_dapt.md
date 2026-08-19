# Phase 2 — Domain-Adaptive Pretraining (DAPT)

Continued masked-language-model (MLM) pretraining of `sentence-transformers/all-mpnet-base-v2`
on the Item 1A corpus, adapting its token representations to financial-risk vocabulary before
any supervised signal is introduced. Only the encoder is carried forward to Phase 3; the MLM
head is discarded. Paragraph-aware chunking (which never splits an Item 1A risk factor across
sequences) is the selected variant. Full training configuration, hyperparameters and results
are documented in `../MODEL_CARD.md` (§3.1).

## Scripts

| Script | Purpose |
|--------|---------|
| `chunk_corpus.py` | Chunk Item 1A pickles into MLM sequences → `dapt_data/{train,val}.jsonl` |
| `train_dapt.py` | Continued MLM training (HuggingFace `Trainer`); saves the best checkpoint on val loss |
| `eval_perplexity.py` | MLM perplexity of a checkpoint on the val set |
| `chunk_stats.py` | Fill-rate statistics (tokens per chunk) for a chunked JSONL |
| `run_dapt.sh` | Batch job, sentence-aware chunking |
| `run_dapt_para.sh` | Batch job, paragraph-aware chunking (the selected variant) |

## Temporal split

Chunks are assigned from `filing_date` in `feature_table.parquet`: train `< 2025-01-01`,
validation `2025`, and test filings (`≥ 2026-01-01`) held out for downstream evaluation.

## Running

`dapt_data/` and the checkpoint directories are gitignored; regenerate them in order.

```bash
conda activate diss

# 1. Chunk the corpus (paragraph-aware, the selected variant)
python dapt/chunk_corpus.py --paragraph_aware --out_dir dapt_data_para

# 2. Train (GPU required; submit as a batch job)
mkdir -p logs
sbatch dapt/run_dapt_para.sh

# 3. Evaluate perplexity: general-English baseline, then the DAPT checkpoint
python dapt/eval_perplexity.py --model_path microsoft/mpnet-base
python dapt/eval_perplexity.py --model_path dapt_checkpoints_para/best
```
