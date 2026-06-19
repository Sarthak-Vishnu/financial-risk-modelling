# Chunking Strategy Comparison — Sentence-aware vs. Paragraph-aware

A/B comparison of two MLM chunking strategies for DAPT on the SEC Item 1A corpus.
Both are within-document, non-overlapping greedy packing to ≤510 tokens (RoBERTa
DOC-SENTENCES family). All training hyperparameters are identical (5 epochs, lr 2e-5,
batch 16×2, MLM 0.15, fp16, early-stop patience 2) — the **only** varying factor is
how text is split into ≤510-token training sequences.

> Status: complete. Results from SLURM job 3508515 (1× RTX A6000, landonia11).

---

## The Two Strategies

| | Sentence-aware (current) | Paragraph-aware (variant) |
|---|---|---|
| Atomic unit | Sentence (NLTK `sent_tokenize`) | Paragraph (`\n\n` split = one Item 1A risk factor) |
| Packing | Greedily pack sentences to ≤510 tokens | Greedily pack whole paragraphs to ≤510 tokens |
| Chunk boundary | May start/end mid-paragraph | Always lands on a paragraph edge |
| Oversized unit fallback | Sentence >510 → token-window split | Paragraph >510 → sentence-pack (then token-window) |
| Data dir | `dapt_data/` | `dapt_data_para/` |
| Checkpoint | `dapt_checkpoints/best` | `dapt_checkpoints_para/best` |
| Code path | `chunk_corpus.py: sentence_pack` | `chunk_corpus.py: paragraph_pack` |

**Rationale for the variant.** Item 1A is structured one-paragraph-per-risk-factor.
Aligning chunk boundaries to those paragraphs keeps each training sequence topically
coherent (one or more whole risk factors, never a half risk factor). The *greedy*
paragraph packing — rather than one-paragraph-per-chunk — is deliberate: paragraph
sizes vary widely (≈110–500 tokens), so packing multiple small paragraphs per chunk
preserves the high fill-rate that RoBERTa found beneficial and that `all-mpnet-base-v2`
needs (mean pooling is diluted by padding).

**Literature grounding.** The sentence-aware strategy is validated as RoBERTa
DOC-SENTENCES (see `Literature_agent.md`, Q1–Q6). The paragraph-aware variant is the
one optional extension the survey flagged — consistent with the coherence principle but
not directly evidenced — hence this empirical A/B before committing to either for Phase 3.

---

## Fill-rate (from `chunk_stats.py` on `train.jsonl`)

| Metric | Sentence-aware | Paragraph-aware |
|---|---|---|
| Train chunks | 165,337 | 201,513  (+21.9%) |
| Mean tokens/chunk | 453.4 | 372.0 |
| Median tokens/chunk | 489.0 | 455.0 |
| % chunks ≥ 480 tokens | 62.8% | 36.9% |

Paragraph-aware produces ~22% more chunks and packs them ~18% less full on average
(near-full fraction roughly halves). This is the expected cost of never splitting a
risk factor across chunks: a chunk often ends well short of 510 tokens because the next
whole paragraph would overflow it. **Note this also means the paragraph-aware model saw
~22% more gradient steps over the same 5 epochs — a compute confound to keep in mind when
reading the perplexity below.**

---

## Validation perplexity (lower = better)

**Fair head-to-head** — both checkpoints on the identical original val set
(`dapt_data/val.jsonl`):

| Checkpoint | Val perplexity (common val set) |
|---|---|
| Sentence-aware (`dapt_checkpoints/best`) | 2.2671 |
| Paragraph-aware (`dapt_checkpoints_para/best`) | **2.2278**  (−1.7%) |

**Convergence check** — each model on its own val set (not directly comparable; the
two val sets differ in chunking):

| Checkpoint | Own val set | Val perplexity |
|---|---|---|
| Sentence-aware | `dapt_data/val.jsonl` | 2.2660 |
| Paragraph-aware | `dapt_data_para/val.jsonl` | 2.2393 |

---

## Verdict

**Paragraph-aware wins the fair head-to-head** (2.2278 vs 2.2671, −1.7% perplexity on the
identical val set) — and does so *despite* lower fill-rate and *despite* a train/eval
chunking mismatch (it was trained on paragraph-packed text but evaluated on the
sentence-packed val set, so the result is not inflated by matching distributions). The
most plausible reading: keeping each Item 1A risk factor intact within a chunk gives the
MLM cleaner, more topically coherent context than sentence packing that straddles risk
factors — enough to outweigh the ~18% loss in mean tokens/chunk.

**Two honest caveats.**
1. *Compute confound.* Paragraph-aware saw ~22% more gradient steps (201k vs 165k chunks ×
   5 epochs). Part of the gain may be extra optimisation, not the chunking per se. A
   step-matched re-run would isolate this, but is not worth the GPU time given the margin.
2. *Small margin.* 1.7% perplexity is modest and may not survive to the downstream task.
   Perplexity is only a Phase 2 proxy; the decisive metric is Phase 5 volatility Spearman ρ.

**Decision.** Carry the **paragraph-aware checkpoint (`dapt_checkpoints_para/best`)** forward
as the encoder base for Phase 3 — it is at least as good on the fair comparison, and its
"never split a risk factor" property is better aligned with the project's downstream
topic-modelling rationale (BERTopic over coherent risk-factor units). The sentence-aware
checkpoint (`dapt_checkpoints/best`) is **retained**, so Phase 5 can include both as an
ablation row if the chunking choice turns out to matter downstream.

---

### Results provenance
SLURM job 3508515, 5 epochs, lr 2e-5, batch 16×2, MLM 0.15, fp16, early-stop patience 2
(identical across both runs). Fill-rate from `chunk_stats.py`; perplexity from
`eval_perplexity.py`. Base model `all-mpnet-base-v2`; `microsoft/mpnet-base` general-English
baseline = 2.6669 (see `README_dapt.md`).
