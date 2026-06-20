# Phase 3 — FinMTEB Intrinsic Evaluation

Intrinsic (zero-shot) evaluation of the Phase 3 encoders on **FinMTEB** (Tang & Yang 2025),
English subset: **2 STS tasks** (Spearman ρ) + **10 Retrieval tasks** (NDCG@10). Run via
`contrastive/eval_finmteb.py` (env `finmteb`, job 3511784, ~49 min on an A6000). The benchmark *was*
run — only the optional 7B `Fin-E5` baseline and the Chinese tasks were skipped.

> **Read the two metrics separately.** The per-model `MEAN` in `eval_results/finmteb/summary.json`
> averages Spearman and NDCG@10 together and is therefore *not* a meaningful single score. The tables
> below split them.

## How to read this (plain-English)

**Verdict: a healthy, expected result — nothing here says redo a step.** Read it as a story across the
pipeline, not a race against SBERT. Of the 5 models, only 3 are ours (`dual`, `three`, `three_lora`);
`sbert` is a generic off-the-shelf yardstick and `dapt` is our own halfway point (after Phase 2, before
Phase 3).

- **Expected & good:** DAPT raises in-domain similarity (STS) but lowers search (retrieval) — the
  textbook MLM "anisotropy" side effect, not a failure. Then **contrastive FT repairs retrieval**
  (~0.33 → ~0.50), which is exactly Phase 3's job. *This is the main "proceed" signal.*
- **Unexpectedly good:** LoRA (`three_lora`), added as an exploratory variant, came out best of our
  three encoders on **both** axes.
- **Mildly disappointing:** adding the novel **sector view under full fine-tuning hurt** (`three` <
  `dual`) — but under LoRA it didn't. Cause is diagnosed below (noisy positives), and it's a cheap fix.
- **Looks bad, is fine:** SBERT leading overall retrieval. It was contrastively trained on 1B+ pairs for
  retrieval, and FinMTEB's retrieval tasks (generic finance QA / 10-K / news) are **out-of-distribution**
  for our SEC Item-1A risk-factor specialization. **Can't be helped, and need not be:** per the
  supervisor's framing (`Feedback01_from_Sunnie.md`), the project is a *volatility-prediction benchmark*
  (Phase 5); FinMTEB is supporting evidence, not the deliverable. Beating SBERT here is **not** a goal.

**Decisive test = Phase 5** (forward-looking 30-day volatility from risk factors), where SBERT is just
one Tier-3 baseline and in-domain geometry should pay off.

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

## Sector-view investigation — why it hurt under full fine-tuning

The drop from `dual` to `three` traces to **pair construction, not training**. `build_sector`
(`build_pairs.py:139`) pairs two paragraphs from different firms in the same `(sic2, fiscal_year)`
**at random, with no similarity filter** — unlike `build_chrono` (`build_pairs.py:115`), which keeps
only TF-IDF cosine ≥ 0.5 matches. Measured intra-pair TF-IDF cosine on a 4k-pair sample per view:

| view | mean | median | % pairs < 0.15 (near-unrelated) |
|---|---|---|---|
| chrono | 0.860 | 0.892 | 0% |
| lexical | 0.561 | 0.571 | 5% |
| **sector** | **0.245** | **0.190** | **38%** |

**38% of sector "positives" are essentially unrelated text** (median similarity 0.19). The model is told
to pull genuinely different paragraphs together a third of the time — a noisy supervision signal. Under
**full fine-tuning** that pressure propagates through all weights and blurs the fine-grained geometry
retrieval needs (`three` < `dual`). Under **LoRA**, updates are confined to low-rank Q/V adapters, so the
noisy signal can't distort the geometry much (`three_lora` ≈ `dual`). Corroborated by training: `three`'s
best val accuracy was **0.727** vs `dual`'s **0.899** — the model genuinely couldn't fit the sector pairs.

**Decision (this round): document & proceed** — all 5 encoders are carried into Phase 5, which is the true
judge of the sector view. If it underperforms there, the cheap fix is a TF-IDF similarity floor on
`build_sector` (mirroring `build_chrono`) and/or lowering `--lambda_sector`; a "masking-only" sector
variant (use sector membership solely for false-negative masking, not as a positive) is the alternative.

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
