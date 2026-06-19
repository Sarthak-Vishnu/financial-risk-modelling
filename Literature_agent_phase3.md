# Literature Survey: Phase 3 — Contrastive Fine-Tuning Design

**Date:** 2026-06-19
**Companion to:** `Literature_agent.md` (Phase 2 — MLM/DAPT input segmentation).
**Context:** SimCSE-style contrastive fine-tuning of a domain-adapted `all-mpnet-base-v2` encoder (post-DAPT) on SEC Item 1A risk disclosures. Extends Chiu et al. (2025, EMNLP)'s dual-view framework (lexical + chronological) with a third **sector view**. Contrastive unit = risk-factor paragraph. Loss = `MultipleNegativesRankingLoss` (MNRL, in-batch negatives) in `sentence-transformers`. Three encoders to be trained: dual-view (Chiu replication), three-view, and a LoRA variant.

**Current view definitions under evaluation:**

| View | Anchor → Positive | Strength | Hard negative |
|---|---|---|---|
| Lexical | Two overlapping spans of the *same* paragraph | Strong | (in-batch) |
| Chronological | Two paragraphs, *same firm*, different fiscal years | Medium | (in-batch) |
| Sector | Two paragraphs, *different firms*, same 2-digit SIC + same fiscal year | Soft | Cross-sector pair |

> **Two upstream corrections before the per-question analysis (both from reading the full Chiu et al. 2025 paper, §3.2.2 & §4.1):**
>
> **(a) Chiu's chronological view ≠ "different fiscal years."** Chiu pairs two paragraphs *from the same firm that contain identical date-format tokens* (e.g. both reference "July 8, 2024"), on the premise that firms experience ~one significant event per day; the date tokens are then **removed before encoding** so the model cannot match on them. Your "same firm, different fiscal years" is a *different and arguably weaker* construction — it has no event-anchoring and risks pairing unrelated risk factors that merely co-occur in the same firm's filings across years. This is a deviation from the cited precedent, not a replication. Flagged under Q5.
>
> **(b) "Cross-sector pair = hard negative" is mislabelled.** In contrastive learning a *hard* negative is one that is close in embedding/surface space but semantically wrong. A cross-sector paragraph is the *natural easy negative* (different SIC ⇒ different risk vocabulary). The genuinely hard negative for a sector positive is a **same-sector, different-risk-type** paragraph (e.g. two semiconductor firms, one paragraph on export-control risk vs one on goodwill-impairment risk). Flagged under Q5.

---

## Q1. Multi-view loss combination with positives of differing strength

**Question:** With >2 positive-pair types of differing strength (strong lexical vs soft sector), should one use a single InfoNCE/MNRL over pooled positives, separate per-view weighted losses, or a temperature/margin scheme? How does Chiu combine its two views? Is there evidence for down-weighting weak positives?

### How Chiu et al. (2025) actually combine their two views

From the full paper (§3.2.1, §4.1): Chiu constructs **8,500 positive paragraph pairs + 1,000 validation pairs for *each* of the two views**, pools them into one training set, and trains a **single InfoNCE objective** (van den Oord et al., 2018) with **in-batch negatives** (Contriever-style; for anchor *pᵢ*, the negatives are all other in-batch positives, giving *B*−1 negatives per anchor). Batch size 64.

Critically: **Chiu applies no per-view weighting and no per-view temperature.** Both views are treated as equal-strength positives. Their ablation (Table 2) trains *separate* single-view encoders and finds lexical ≥ chronological individually, with the pooled "both views" encoder best overall. So even with a strength asymmetry between their two views, equal pooling worked.

> **Caveat for your case:** Chiu's *weakest* view (chronological) is still **within-firm and event-anchored**. Your sector view is **cross-firm** and only loosely topically aligned (same SIC + year). It is weaker than anything Chiu pooled equally, so "Chiu pooled equally, therefore I can" does not fully transfer.

### What the broader literature recommends

- **Multi-positive InfoNCE has two canonical forms — Khosla et al. (2020), *Supervised Contrastive Learning*, NeurIPS 2020.** They define **L_out** (sum over positives *outside* the log) and **L_in** (sum *inside* the log) and show empirically that **L_out is superior** (the gradient structure is better-behaved for many positives). This is the reference formulation for "one anchor, several positives."
- **MIL-NCE — Miech et al. (2020), *End-to-End Learning of Visual Representations from Uncurated Instructional Videos*, CVPR 2020.** Puts the positive set inside a log-sum-exp, requiring the anchor to be close to *at least one* positive rather than *all*. This is **explicitly designed for noisy/weak positives** — exactly the situation a soft sector positive creates. Strong evidence that weak positives should not be forced to full alignment.
- **Soft / similarity-weighted positives — Denize et al. (2023), *Similarity Contrastive Estimation for Self-Supervised Soft Contrastive Learning*, WACV 2023.** Replaces binary positive/negative targets with **continuous similarity weights**, so a weak positive contributes proportionally less. Direct evidence for *down-weighting* soft positives rather than treating them as hard positives.
- **Supervised SimCSE — Gao et al. (2021), *SimCSE*, EMNLP 2021.** Precedent for **weighting by pair type**: the supervised objective adds NLI contradiction pairs as hard negatives with a tunable weight term α, and uses τ = 0.05. Establishes that differential weighting of pair *types* (not just a single pooled loss) is standard and beneficial.

### Verdict

⚠️ **Partially supported — refine.** A single MNRL over pooled positives is defensible (it is what Chiu did), **but** equal-weighting a cross-firm soft sector positive against a same-paragraph lexical positive is riskier than any case Chiu validated. Better-evidenced alternatives, in order of evidence strength for your setting:

1. **Per-view weighted sum of losses:** `L = L_lexical + λ_chrono·L_chrono + λ_sector·L_sector` with `λ_sector < 1`. Cleanest, matches Gao et al.'s pair-type weighting and Denize et al.'s down-weighting principle. **Recommended.**
2. **MIL-NCE-style** log-sum-exp over the positive set per anchor (Miech et al. 2020) — robust to the sector view being occasionally a false/weak positive.
3. **Soft similarity weighting** (Denize et al. 2023) — weight the sector positive by TF-IDF/embedding similarity of the pair, so only genuinely-aligned cross-firm pairs pull strongly.

Avoid pooling all three at equal weight in one MNRL as the *only* configuration — at minimum ablate it against a down-weighted sector variant.

---

## Q2. Shortcut prevention — token masking / normalization

**Question:** Beyond Chiu's date removal, what masking/normalization (dates, firm names, tickers, boilerplate) does the literature recommend so pairs match on risk semantics, not surface identifiers?

The sector view is the highest shortcut risk in your design: two same-SIC paragraphs can be matched on **shared sector vocabulary, firm names, ticker symbols, and recurring boilerplate** rather than on shared *risk semantics*. Evidence-backed mitigations:

- **Entity blanking — Soares et al. (2019), *Matching the Blanks: Distributional Similarity for Relation Learning*, ACL 2019.** (Cited in Chiu's own reference list.) Replace entity mentions with a `[BLANK]` token with fixed probability during contrastive training, forcing the encoder to learn *relational/semantic* structure rather than memorise entity identity. This is the canonical precedent for your problem: **blank firm names and tickers** so the sector view cannot shortcut on "both mention NVDA." Recommended.
- **Number normalization — Loukas et al. (2022), *FiNER: Financial Numeric Entity Recognition for XBRL Tagging*, ACL 2022 (arXiv:2203.06482).** This is the paper that released the **SEC-BERT** family, including **SEC-BERT-NUM** (every numeric token → `[NUM]` pseudo-token) and **SEC-BERT-SHAPE** (numbers → shape tokens like `[XX.X]`). Direct financial-domain evidence that **replacing magnitudes prevents the model keying on specific dollar/percentage figures** — relevant because boilerplate risk factors differ mainly in their numbers. (Note: this also resolves the open verification note in `Literature_agent.md` — arXiv:2203.06482 is FiNER, which *introduces* SEC-BERT.)
- **Date removal — Chiu et al. (2025).** Already in your plan for the chronological view; extend the same `[DATE]`/removal normalization to the sector and lexical views for consistency.
- **General entity-masking taxonomy** (NER/relation-extraction literature): `UNK` (single mask token), `NE` (replace with entity *type*), and grammatical-role substitution. The `NE`-type strategy (replace "Pfizer" → `[COMPANY]`, "Q3 2024" → `[DATE]`, "$4.2 billion" → `[NUM]`) generalises best while preserving syntactic structure.
- **Boilerplate** — no clean masking precedent, but **Tang & Yang (2025, FinMTEB)** explicitly flag SEC boilerplate ("The company's performance is subject to various risks…") as frequent, low-information text that confounds financial embedding models. Mitigate by (a) the normalizations above, which strip the identifiers boilerplate clusters around, and/or (b) optional boilerplate down-weighting via Loughran-McDonald-style frequency filtering — but treat this as exploratory, not evidenced.

### Verdict

✅ **Supported but currently incomplete.** Date removal (inherited from Chiu) is necessary but not sufficient for a cross-firm sector view. The literature provides two well-cited, directly applicable additions you are not yet using: **(1) entity/firm/ticker blanking (Soares et al. 2019)** and **(2) number normalization (Loukas et al. 2022, SEC-BERT-NUM)**. Add both, especially for the sector view.

---

## Q3. False negatives under in-batch negatives

**Question:** With sector positives, two same-sector paragraphs can land in one batch and be wrongly pushed apart as negatives. How do SimCSE/Chiu/related works handle false-negative contamination?

This is the **single most important issue in your three-view design.** MNRL / in-batch negatives assume every non-designated in-batch item is a true negative. Once "same sector" is a *positive* axis, any batch containing two same-sector paragraphs that are **not** the designated pair will treat them as negatives — directly contradicting the sector objective. Chiu does **not** face this (their views are within-firm/within-paragraph, so accidental cross-firm collisions are rare); your sector view introduces it.

Evidence on handling false negatives:

- **SimCSE — Gao et al. (2021), EMNLP 2021.** Baseline: ignores false negatives, assumes random in-batch items are true negatives. Tolerable for unsupervised STS, **not** when a semantic-equivalence axis is used as a positive — your case.
- **DCLR — Zhou et al. (2022), *Debiased Contrastive Learning of Unsupervised Sentence Representations*, ACL 2022 (arXiv:2205.00656).** Uses a complementary model to **down-weight (punish) likely false negatives** via per-instance weights. Evidence that instance-level reweighting of suspected false negatives improves sentence embeddings.
- **False Negative Cancellation — Huynh et al. (2022), *Boosting Contrastive Self-Supervised Learning with False Negative Cancellation*, WACV 2022 (arXiv:2011.11765).** Identifies likely false negatives and either **removes** them from the negative set or **attracts** them (promotes to positives); reports **+2.1% over SimCSE on STS**. Two concrete masking strategies.
- **Clustering-aware negative sampling** (Findings of ACL 2023) — uses cluster structure to avoid sampling same-cluster items as negatives. Conceptually identical to "don't sample same-sector items as negatives."
- **Supervised Contrastive Learning — Khosla et al. (2020), NeurIPS 2020.** The clean solution **when labels are known**: same-label items are *positives, not negatives*, by construction. **You know the SIC label** — so this is directly available to you, unlike the unsupervised debiasing works above which must *estimate* false negatives.

### Verdict

❌ **Current design (plain MNRL) is contradicted by the literature for the sector view** — it will manufacture false negatives. Because your sector label is **known**, the best-evidenced fix is the *supervised* one (Khosla et al. 2020), not estimation-based debiasing:

1. **Label-aware negative masking (recommended):** for each anchor, mask out of the negative set all in-batch paragraphs sharing its (SIC, fiscal-year) — they are not valid negatives. (In `sentence-transformers`, this requires a custom loss / batch sampler; default MNRL does *not* do it.)
2. **Sector-aware batch construction:** build batches so same-sector paragraphs appear **as designated positive pairs**, not as unrelated fillers — turning potential collisions into intended positives.
3. If 1–2 are impractical, fall back to **DCLR-style (Zhou et al. 2022)** or **False-Negative-Cancellation (Huynh et al. 2022)** soft handling — but these are second-best given you have ground-truth sector labels.

This is the change most likely to materially affect sector-view quality. Do not ship plain MNRL for the three-view encoder without one of the above.

---

## Q4. LoRA for sentence encoders under a contrastive objective

**Question:** FinGPT cites r∈{8,16} but that's generative LLaMA. For LoRA on a bidirectional MPNet/BERT encoder under contrastive FT, what target modules and ranks does the literature support? Any work applying LoRA to SBERT-style contrastive FT?

The FinGPT precedent is the **wrong reference class**: FinGPT (Yang et al., 2023) applies LoRA to a *generative, decoder-only* LLaMA for instruction tuning. Ranks and target modules do not transfer cleanly to a bidirectional encoder under a contrastive (not autoregressive) loss. Better-matched evidence:

- **LoRA (original) — Hu et al. (2021/2022), *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.** Their ablation on a BERT-family setting finds adapting **W_q and W_v only is sufficient** (adapting all of Q,K,V,O gives little extra), with **r as low as 8** competitive and **α = 2r** a common heuristic. This is the foundational target-module result and applies to encoders.
- **LoRACode — *LoRA Adapters for Code Embeddings* (2025, arXiv:2503.05315).** The closest precedent: LoRA on **bidirectional BERT-family encoders (CodeBERT, GraphCodeBERT, UniXcoder) under a contrastive embedding objective with mean-pooling**. Configuration: **target = Query + Value**, **rank ∈ {16, 32, 64}**, **α = 2·rank**, **dropout 0.1**; they found Q+V sufficient and middle-layer adaptation strongest. Almost exactly your setting (bi-encoder + contrastive + mean pooling).
- **General embedding-LoRA practice** (community/applied): **r = 32, α = 32, dropout 0.05**, applied to **Q, K, V, and output** projections — a slightly heavier config that also works.
- **Architecture note for `all-mpnet-base-v2`:** MPNet's attention exposes `q`, `k`, `v`, `o` linear modules (HF `MPNetSelfAttention`). The minimal well-supported target set is therefore `["q", "v"]`; a reasonable heavier set is `["q", "k", "v", "o"]`.

### Verdict

✅ **Supported, but replace the FinGPT justification.** Cite **Hu et al. (2021)** and **LoRACode (2025)**, not FinGPT, as the relevant precedents.

- **Target modules:** start with **query + value** (`["q","v"]`) — Hu et al. + LoRACode both support this as sufficient. Optionally add `o`/`k` and ablate.
- **Rank:** **r = 16–32** is the better-matched range for contrastive encoder FT (LoRACode), with **α = 2r**. FinGPT's r∈{8,16} is a defensible *lower* bound but is justified by the wrong reference class; r=8 risks under-capacity for reshaping a *post-DAPT* embedding space across three views.
- **Dropout:** 0.05–0.1 (both precedents).
- Keep mean-pooling + L2-norm (consistent with LoRACode and your base model).

---

## Q5. Unit granularity + Chiu's exact recipe

**Question:** Confirm Chiu's exact construction (unit, token length, lexical overlap fraction, temperature, batch size). Is paragraph-level better-supported than sentence-level for financial contrastive FT?

### Chiu et al. (2025) exact recipe — verified from the paper

| Component | Value (Chiu et al. 2025) |
|---|---|
| Base model | BERT-base-uncased |
| **Unit** | **Paragraph** |
| **Token length** | Truncated/padded to **256 tokens** |
| Training corpus | 10-K filings **2018–2020**, all firms, all sections (eval retrieval restricted to Item 1A + 7A) |
| Positive pairs | **8,500 per view** (+1,000 validation per view) |
| Loss | **InfoNCE** (van den Oord et al. 2018), in-batch negatives, *B*−1 per anchor |
| **Batch size** | **64** |
| Learning rate | **2×10⁻⁵**, Adam, linear warmup |
| Regularization | L2, **≤50 epochs**, early stopping |
| Pooling | **Mean pooling** over final-layer tokens + **L2 normalization** |
| **Lexical overlap** | **Random** indices *i < j* over the token sequence → spans `[w₁…w_j]` and `[w_i…w_n]`; the overlap region `[w_i…w_j]` is **variable, not a fixed fraction** |
| Chronological | Same firm + **identical date-format tokens**; dates **removed** before encoding |
| **Temperature τ** | Appears as a hyperparameter in their InfoNCE equation; **a numeric value is not reported in the paper** |

> **Two design notes from this recipe:**
> - **Lexical overlap is not a tunable fraction in Chiu** — it is two random nested spans of the same paragraph, so overlap varies per example (often large). If your spec fixes a specific overlap fraction, that is a reasonable engineering choice but is *not* "the Chiu setting"; consider randomizing *i<j* to replicate faithfully.
> - **Temperature:** Chiu does not publish τ. SimCSE's canonical **τ = 0.05** (Gao et al. 2021) is the standard default; note that `sentence-transformers` **MNRL uses `scale=20.0`, which is exactly τ = 1/20 = 0.05** — so your MNRL default already matches the SimCSE convention. No action needed, but state it explicitly rather than leaving τ implicit.

### Paragraph vs sentence granularity

- **Paragraph is better-supported for *this* task.** A SEC Item 1A risk factor is a **paragraph-level semantic unit** (one risk per paragraph). Chiu (2025) — the closest precedent — operates at paragraph/256-token granularity for exactly this reason. Sentence-level fragmentation would split a single risk factor across multiple training units, diluting the risk semantics the views are meant to capture.
- **Sentence-level precedents exist but for different downstream goals:** Jehnen et al. (2026, FinTextSim) tokenise to sentences for *topic modelling*; SimCSE (Gao et al. 2021) is sentence-level for *general STS*. Neither targets paragraph-scale *risk* semantics.
- **FinMTEB (Tang & Yang, 2025)** evaluates both granularities and finds domain-adapted models win regardless — neutral on the unit, but reinforces that the *domain adaptation* (your DAPT + contrastive stack) matters more than the unit choice.

### Verdict

✅ **Paragraph-level is the better-supported choice** for financial risk contrastive FT, matching the closest precedent (Chiu 2025). Two design deviations to flag (carried from the top of this document):

1. **Chronological view:** your "same firm, different fiscal years" is **not** Chiu's construction ("same firm, same date-token"). It is weaker and unanchored. Either adopt Chiu's date-anchored pairing faithfully, or justify the fiscal-year variant explicitly as a deliberate extension (and ablate it).
2. **"Cross-sector hard negatives" are easy negatives, not hard ones.** For a genuine hard-negative signal on the sector view, mine **same-sector, different-risk-type** paragraphs (high lexical overlap, different risk semantics), e.g. ANCE-style embedding-space mining. Relabel and, ideally, ablate hard-negative mining as a separate factor.

---

## Summary Verdict Table

| Q | Topic | Current design | Verdict | Best-evidenced action |
|---|---|---|---|---|
| 1 | Multi-view loss combination | Single MNRL over pooled positives | ⚠️ Partial | Down-weight the soft sector view: per-view `λ_sector<1` (Gao 2021; Denize 2023) or MIL-NCE (Miech 2020). Ablate vs equal pooling. |
| 2 | Shortcut prevention | Date removal only (from Chiu) | ✅ but incomplete | Add **entity/firm/ticker blanking** (Soares 2019) + **number normalization** (Loukas 2022, SEC-BERT-NUM). |
| 3 | False negatives | Plain in-batch MNRL | ❌ Contradicted for sector view | **Label-aware negative masking / sector-aware batching** (Khosla 2020) — you have the SIC labels. Most important fix. |
| 4 | LoRA on encoder | FinGPT r∈{8,16} (generative) | ✅ wrong citation | Cite **Hu 2021** + **LoRACode 2025**: target **Q+V**, **r=16–32**, **α=2r**, dropout 0.05–0.1. |
| 5 | Unit + Chiu recipe | Paragraph; "diff fiscal year" chrono; "cross-sector hard neg" | ✅ unit; ⚠️ two deviations | Keep paragraph/256-tok. Fix chrono to date-anchored (or justify). Relabel hard negatives → same-sector-different-risk. Set τ=0.05 (=MNRL scale 20) explicitly. |

**Bottom line:** The paragraph unit, MNRL/in-batch backbone, and LoRA direction are all on solid literature footing. The three changes most strongly supported by evidence — in priority order — are: **(1) label-aware false-negative handling for the sector view (Q3)**, **(2) down-weighting the soft sector positive (Q1)**, and **(3) entity + number normalization to block shortcuts (Q2)**. Q4 is a citation/scope fix, Q5 is two construction corrections.

---

## Key Papers Referenced

### Contrastive learning — foundations and losses

| Paper | Year | Venue | Relevant finding |
|---|---|---|---|
| van den Oord et al. — CPC / InfoNCE | 2018 | arXiv:1807.03748 | InfoNCE objective; basis of MNRL |
| Reimers & Gurevych — Sentence-BERT | 2019 | EMNLP-IJCNLP | Bi-encoder + contrastive FT framework; in `Literature/` |
| Khosla et al. — Supervised Contrastive Learning | 2020 | NeurIPS 2020 | **L_out vs L_in multi-positive forms (L_out better); same-label = positive, not negative** |
| Miech et al. — MIL-NCE | 2020 | CVPR 2020 | Log-sum-exp over positives; **robust to weak/noisy positives** |
| Gao et al. — SimCSE | 2021 | EMNLP 2021 | τ=0.05; supervised variant weights hard negatives (α); pair-type weighting precedent |
| Denize et al. — Similarity Contrastive Estimation | 2023 | WACV 2023 | **Soft/continuous weighting of positives by similarity** |

### False-negative handling

| Paper | Year | Venue | Relevant finding |
|---|---|---|---|
| Huynh et al. — False Negative Cancellation | 2022 | WACV 2022 (arXiv:2011.11765) | Identify + remove/attract false negatives; +2.1% over SimCSE on STS |
| Zhou et al. — DCLR | 2022 | ACL 2022 (arXiv:2205.00656) | Instance-weighting to punish false negatives in sentence embeddings |
| (Clustering-aware negative sampling) | 2023 | Findings of ACL 2023 | Avoid sampling same-cluster items as negatives |

### Shortcut prevention / normalization

| Paper | Year | Venue | Relevant finding |
|---|---|---|---|
| Soares et al. — Matching the Blanks | 2019 | ACL 2019 | **`[BLANK]` entity masking to learn relations, not identities** (cited in Chiu) |
| Loukas et al. — FiNER / SEC-BERT | 2022 | ACL 2022 (arXiv:2203.06482) | **SEC-BERT-NUM / -SHAPE: number → `[NUM]`/shape tokens**; resolves the SEC-BERT verification note in `Literature_agent.md` |

### LoRA

| Paper | Year | Venue | Relevant finding |
|---|---|---|---|
| Hu et al. — LoRA | 2021 | ICLR 2022 | **Adapting W_q, W_v sufficient; r≈8 competitive; α=2r** |
| LoRACode — LoRA Adapters for Code Embeddings | 2025 | arXiv:2503.05315 | **Closest precedent: bi-encoder + contrastive + mean-pool; Q+V, r∈{16,32,64}, α=2r, dropout 0.1** (authors to verify) |
| FinGPT — Yang et al. | 2023 | arXiv:2306.06031 | r∈{8,16} but **generative LLaMA** — wrong reference class for encoder contrastive FT; in `Literature/` |

### Financial-domain context

| Paper | Year | Venue | Relevant finding |
|---|---|---|---|
| Chiu et al. — Dual-view Adaptation | 2025 | EMNLP 2025 | **Direct parent**; paragraph/256-tok, batch 64, lr 2e-5, InfoNCE in-batch, mean-pool; lexical = random nested spans; chrono = same-firm same-date-token (dates removed); DOI: https://doi.org/10.18653/v1/2025.emnlp-main.1336 |
| Tang & Yang — FinMTEB | 2025 | EMNLP 2025 | Boilerplate flagged as low-information confound; domain adaptation > unit choice; DOI: https://doi.org/10.18653/v1/2025.emnlp-main.179 |
| Jehnen et al. — FinTextSim | 2026 | Frontiers in AI | Sentence-level granularity (topic modelling, not risk); DOI: https://doi.org/10.3389/frai.2026.1752103 |

> **To verify before citing in the dissertation:** LoRACode authors and exact venue (arXiv:2503.05315); the numeric τ in `sentence-transformers` MNRL (`scale=20.0` ⇒ τ=0.05) against your installed version; Khosla et al. L_out/L_in result page reference.

---

*Last updated: 2026-06-19. Covers Q1–Q5 on contrastive fine-tuning design (multi-view loss, shortcut prevention, false negatives, LoRA, unit granularity) for SimCSE-style FT of post-DAPT `all-mpnet-base-v2` on SEC Item 1A. Companion to `Literature_agent.md` (Phase 2 DAPT segmentation). Papers not in `Literature/` or `Literature (new)/` should be verified before citing.*
