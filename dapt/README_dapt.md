# Phase 2 — Domain-Adaptive Pretraining (DAPT)

## What Was Done

Continued MLM pretraining of `sentence-transformers/all-mpnet-base-v2` on the Item 1A corpus to adapt its token representations to financial risk vocabulary before any supervised signal is introduced (following Gururangan et al., 2020).

---

## Scripts

| Script | Purpose |
|--------|---------|
| `chunk_corpus.py` | Chunks Item 1A pickles into MLM-ready sequences; outputs `dapt_data/train.jsonl` and `dapt_data/val.jsonl` |
| `train_dapt.py` | Continued MLM training with HuggingFace Trainer; saves best checkpoint on val loss |
| `eval_perplexity.py` | Evaluates MLM perplexity of any checkpoint on the val set |
| `chunk_stats.py` | Reports fill-rate stats (tokens/chunk) for a chunked JSONL |
| `run_dapt.sh` | SLURM batch job script (sentence-aware chunking) |
| `run_dapt_para.sh` | SLURM batch job script (paragraph-aware chunking variant) |

---

## Chunking Strategy

Both strategies are within-document, non-overlapping greedy packing to ≤510 tokens (RoBERTa
DOC-SENTENCES family, validated by the literature survey — see `../Literature_agent.md`). Item 1A
sections are long regulatory prose (median ~7k words, max ~18k words); fixed-window chunking would
split mid-concept, creating noisy MLM examples, so neither strategy uses it.

- **Sentence-aware** (`sentence_pack`, default): NLTK `sent_tokenize` → greedily pack whole sentences
  to 510 tokens. Single sentences >510 tokens fall back to a fixed token window.
- **Paragraph-aware** (`paragraph_pack`, `--paragraph_aware`): greedily pack whole *paragraphs*
  (`\n\n` split = one Item 1A risk factor) to 510 tokens, so chunk boundaries always land on risk-factor
  edges and no risk factor is ever split across chunks. A paragraph >510 tokens falls back to sentence packing.

**Selected: paragraph-aware.** An A/B comparison (`CHUNKING_COMPARISON.md`) showed paragraph-aware
achieves lower val perplexity on the identical val set (2.2278 vs 2.2671, −1.7%) despite ~18% lower
fill-rate, and its "never split a risk factor" property aligns with the downstream BERTopic stage.
The paragraph-aware checkpoint (`dapt_checkpoints_para/best`) is the encoder base for Phase 3; the
sentence-aware checkpoint is retained for a possible Phase 5 chunking ablation. See
`CHUNKING_COMPARISON.md` for the full numbers and caveats (notably a ~22% training-step confound).

**Result (sentence-aware):** 165,337 train / 11,765 val sequences from 8,017 filings (230 skipped — no feature_table match).
**Result (paragraph-aware):** 201,513 train sequences.

---

## Temporal Split

Chunks are assigned using `filing_date` from `feature_table.parquet`:
- **Train:** `filing_date < 2025-01-01` (7,353 filings)
- **Val:** `2025-01-01 ≤ filing_date < 2026-01-01` (406 filings)
- **Test filings excluded** (`filing_date ≥ 2026-01-01`) — held out for Phase 5 downstream evaluation

---

## Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Base model | `sentence-transformers/all-mpnet-base-v2` | Kept constant across all ablation conditions so only adaptation strategy varies |
| MLM masking rate | 15% | Standard from BERT/Gururangan et al. |
| Learning rate | 2e-5 | Standard DAPT lr from Gururangan et al. |
| Warmup | 6% of total steps | Prevents early instability on a pre-trained model |
| Batch size | 16 × grad_accum 2 = effective 32 | Fits A6000 (49GB) with 512-token sequences; effective batch of 32 standard for MLM |
| Epochs | 5 (early stopping patience 2) | Val perplexity decreased every epoch so all 5 ran |
| Weight decay | 0.01 | Standard regularisation |
| fp16 | True | Mixed precision for training speed on A6000 |
| Max sequence length | 512 | MPNet architecture limit |

---

## Choice of Base Model for DAPT

DAPT was run on `sentence-transformers/all-mpnet-base-v2`, not on `microsoft/mpnet-base`, for a deliberate reason.

`all-mpnet-base-v2` has been fine-tuned for sentence-level similarity (on SNLI, multi-NLI, etc.), giving its encoder far better sentence-level geometry than the base `mpnet-base` which only has token-level MLM pre-training. Since **the MLM head is discarded after DAPT** — only the encoder weights carry forward into Phase 3 (contrastive FT) and all downstream phases — the quality of the encoder's sentence representations matters more than the MLM head's starting state. Starting from `all-mpnet-base-v2` preserves that sentence similarity structure and then reshapes its token vocabulary toward financial risk language.

`microsoft/mpnet-base` is used only as the perplexity comparison baseline because it has a pre-trained MLM head on general English, making it a fair benchmark for measuring how much domain adaptation improved financial-text prediction.

---

## Why 5 Epochs

Val perplexity decreased every epoch (2.83 → 2.49 → 2.36 → 2.29 → 2.27) with no sign of plateau, so early stopping (patience = 2) never triggered and all 5 scheduled epochs ran to completion. Training was not extended beyond 5 epochs because:

1. The improvement from epoch 4 → 5 was marginal (2.29 → 2.27, ~0.9%)
2. The primary goal of DAPT is encoder adaptation, not MLM head convergence — the encoder representation shift is largely complete within the first few epochs and further epochs yield diminishing returns

Hence, the checkpoint at epoch 5 is sufficient. If the downstream Phase 5 ablation shows the DAPT encoder underperforms, an extended run (10 epochs) is a straightforward fallback.

---

## Results

Perplexity on the common val set (`dapt_data/val.jsonl`):

| Model | Val Perplexity | Notes |
|-------|----------------|-------|
| `sentence-transformers/all-mpnet-base-v2` (randomly initialised MLM head) | 946,018.79 | Actual training start point — MLM head absent in sentence-transformer variant, initialised from random weights |
| `microsoft/mpnet-base` (general English MLM head) | 2.6669 | Comparison baseline — pre-trained MLM head on general English |
| `dapt_checkpoints/best` — sentence-aware DAPT, epoch 5 | 2.2671 | Continued MLM on sentence-packed Item 1A corpus |
| `dapt_checkpoints_para/best` — paragraph-aware DAPT, epoch 5 | **2.2278** | **Selected encoder base for Phase 3** (see `CHUNKING_COMPARISON.md`) |

**~16.5% perplexity reduction** over `mpnet-base` (paragraph-aware). Val perplexity decreased monotonically across all 5 epochs in both runs, satisfying the IPP Phase 2 success criterion.

Best checkpoint: **`dapt_checkpoints_para/best/`** — used as the encoder base for Phase 3 (Contrastive FT). The sentence-aware checkpoint `dapt_checkpoints/best/` is retained for a possible Phase 5 chunking ablation.

> **⚠️ Flag for supervisor discussion (Phase 2):**
> 1. **Chunking choice.** Paragraph-aware packing (never splits an Item 1A risk factor) was selected over sentence-aware on a −1.7% val-perplexity edge (2.2278 vs 2.2671). The margin is small and confounded by ~22% more training steps (paragraph-aware yields 201k vs 165k chunks at fixed 5 epochs). Confirm whether this is an acceptable basis to commit to paragraph-aware for Phase 3, or whether a step-matched re-run / deferral to the Phase 5 downstream comparison is preferred. Full analysis + caveats in `CHUNKING_COMPARISON.md`.
> 2. **Perplexity baseline.** `microsoft/mpnet-base` (2.67) is the intended comparison baseline (general English vs. financial domain, same architecture). `all-mpnet-base-v2` (946k) is the literal training start point but not a meaningful MLM baseline since its head was random. Confirm which to report in the dissertation.

---

## Reproducing

`dapt_data/` and `dapt_checkpoints/` are gitignored. Run the following steps in order to regenerate them.

```bash
# 1. Chunk the corpus → generates dapt_data/train.jsonl and dapt_data/val.jsonl
python dapt/chunk_corpus.py

# 2. Train DAPT via SLURM batch job (Teaching partition, A6000 GPU, 48h limit)
mkdir -p logs
sbatch dapt/run_dapt.sh
# Monitor: squeue -u $USER
# Logs:    tail -f logs/dapt_<JOBID>.out

# 3. Evaluate perplexity of any checkpoint
python dapt/eval_perplexity.py                                        # pre-DAPT (mpnet-base baseline)
python dapt/eval_perplexity.py --model_path dapt_checkpoints/best     # post-DAPT
```

> For an interactive GPU session: `srun --partition=Teaching --gres=gpu:nvidia_rtx_a6000:1 --pty bash`  
> If A6000 unavailable, substitute `nvidia_rtx_a40` — same 48GB class.  
> Activate env first: `conda activate diss`

### Baseline perplexity (for reporting)

```bash
# General English baseline (apples-to-apples comparison — same architecture, no domain adaptation)
python dapt/eval_perplexity.py --model_path microsoft/mpnet-base

# Actual training start point (random MLM head — not a meaningful MLM baseline, recorded for completeness)
python dapt/eval_perplexity.py --model_path sentence-transformers/all-mpnet-base-v2

# Post-DAPT result
python dapt/eval_perplexity.py --model_path dapt_checkpoints/best
```
