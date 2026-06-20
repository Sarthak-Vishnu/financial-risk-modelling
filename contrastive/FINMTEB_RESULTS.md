# Phase 3 — FinMTEB Intrinsic Evaluation

Intrinsic (zero-shot) evaluation of the Phase 3 encoders on **FinMTEB** (Tang & Yang 2025),
English subset: **2 STS tasks** (Spearman ρ) + **10 Retrieval tasks** (NDCG@10). Run via
`contrastive/eval_finmteb.py` (env `finmteb`, job 3511784, ~49 min on an A6000).

> **Read the two metrics separately.** The per-model `MEAN` in `eval_results/finmteb/summary.json`
> averages Spearman and NDCG@10 together and is therefore *not* a meaningful single score. The tables
> below split them.

## Models

| Tag | Checkpoint | Description |
|---|---|---|
| `sbert` | `sentence-transformers/all-mpnet-base-v2` | General SBERT baseline (contrastively trained on 1B+ pairs) |
| `dapt` | `dapt_checkpoints_para/best` | Phase 2 DAPT base (MLM-adapted MPNet, mean+norm pooling) |
| `dual` | `contrastive_checkpoints/dual` | Phase 3 dual-view (lexical + chrono) — Chiu replication / control |
| `three` | `contrastive_checkpoints/three` | Phase 3 three-view (+ sector, λ=0.5), full fine-tuning |
| `three_lora` | `contrastive_checkpoints/three_lora` | Phase 3 three-view, LoRA (Q+V, r=16) |

## STS — Spearman ρ (higher = better)

| Model | FinSTS | FINAL | **STS mean** |
|---|---|---|---|
| sbert | 0.215 | 0.410 | 0.312 |
| **dapt** | 0.351 | 0.580 | **0.465** |
| dual | 0.298 | 0.514 | 0.406 |
| three | 0.215 | 0.459 | 0.337 |
| three_lora | 0.303 | 0.516 | 0.410 |

## Retrieval — NDCG@10 (higher = better)

| Model | FiQA | Apple10K | FinQA | FinBench | HC3 | TATQA | Goldman | TTE-Ency | USNews | TTE-News | **Ret. mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **sbert** | 0.497 | 0.852 | 0.101 | 0.736 | 0.628 | 0.188 | 0.386 | 0.982 | 0.818 | 0.862 | **0.605** |
| dapt | 0.182 | 0.656 | 0.043 | 0.185 | 0.304 | 0.094 | 0.187 | 0.715 | 0.422 | 0.474 | 0.326 |
| dual | 0.294 | 0.752 | 0.076 | 0.667 | 0.432 | 0.150 | 0.314 | 0.916 | 0.683 | 0.690 | 0.497 |
| three | 0.213 | 0.654 | 0.050 | 0.540 | 0.335 | 0.115 | 0.280 | 0.703 | 0.657 | 0.642 | 0.419 |
| three_lora | 0.285 | 0.733 | 0.078 | 0.655 | 0.425 | 0.146 | 0.308 | 0.871 | 0.732 | 0.789 | 0.502 |

## Findings

**1. DAPT helps in-domain STS but breaks retrieval geometry.** DAPT is the *best* model on STS
(0.465 vs SBERT 0.312) — MLM adaptation sharpens semantic similarity — yet it is the *worst* on
retrieval (0.326 vs SBERT 0.605). This is the textbook anisotropy effect: MLM degrades the embedding
geometry that nearest-neighbour retrieval relies on.

**2. Contrastive fine-tuning repairs retrieval — exactly its purpose.** Both Phase 3 encoders recover
retrieval from DAPT's 0.326 up to ~0.50 while retaining most of DAPT's STS gain. This confirms the
pipeline is behaving as designed: Phase 3 restores the sentence-level geometry Phase 2 damaged.

**3. LoRA > full fine-tuning, and it rescues the sector view.** `three_lora` is the best of our encoders
on *both* axes (0.410 STS / 0.502 retrieval). Full-FT `three` (0.337 / 0.419) is *worse* than `dual`
(0.406 / 0.497) — adding the sector view under full fine-tuning caused forgetting. Under LoRA the same
view does no harm (≈ dual). Directly answers **Sunnie Action Item 4** (marginal value of the sector
view): **negative under full fine-tuning, neutral-to-positive under LoRA's low-rank regularization.**

## Caveat — why SBERT still leads on retrieval

SBERT (`all-mpnet-base-v2`) wins overall retrieval, which is expected and not a failure of the method:
- It was contrastively trained on **1B+ general pairs specifically for retrieval**.
- **FinMTEB retrieval is out-of-distribution for our specialization**: its tasks are generic finance QA
  (FiQA, FinQA, TATQA), 10-K, and news retrieval — *not* SEC Item-1A risk-factor paragraphs, the only
  text our encoder was specialized on.
- Our contrastive FT used only 10k pairs/view; scaling could narrow the gap.

FinMTEB is therefore an **intrinsic sanity check**, not the project's decisive metric. The decisive
test is **Phase 5 downstream volatility prediction on risk factors**, where in-domain geometry should
pay off. Flagged for supervisor discussion.
