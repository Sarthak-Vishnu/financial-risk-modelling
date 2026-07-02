# Original Pipeline — Design Details, Parameters, and Literature Grounding

**Project:** Hybrid Topic and Domain-Adaptive Modelling for Financial Risk and Forecasting
**Student:** S2880814 (Sarthak Vishnu) · **Supervisor:** Prof Tiejun Ma · University of Edinburgh, School of Informatics
**Scope of this document:** the *original* pipeline as committed on the `main` branch (Phases 1–5), i.e. the
work demonstrated to the supervisor. The later multimodal redesign (`exp/supervised-finetune` branch) is
**not** covered here.

This is a reference write-up of *what was built, with which parameters, and why* — every non-obvious design
choice is traced to the literature surveys (`Literature_agent.md`, `Literature_agent_phase3.md`) or the
in-repo design memos (`dapt/README_dapt.md`, `dapt/CHUNKING_COMPARISON.md`, `contrastive/FINMTEB_RESULTS.md`).

---

## 0. Research framing and target

- **Task.** Predict a firm's **30-day forward realised volatility** (`fwd_vol_30d`) from the text of its
  SEC 10-K **Item 1A "Risk Factors"** disclosure, optionally combined with a lagged-volatility anchor.
- **Modelling target.** `log(fwd_vol_30d)` — volatility is approximately log-normal and strictly positive;
  the log transform is applied once and used by *every* tier so R²/RMSE are directly comparable
  (`baselines/run_baselines.py:86`, floor `EPS = 1e-6` before the log).
- **Primary metrics.** Within-split **Spearman rank correlation** (information coefficient, IC) and
  **R²_log**; RMSE in both log and original vol scale reported as secondary
  (`baselines/run_baselines.py:91` `metrics()`).
- **Volatility definition** (`dataset_config/compute_volatility_labels.py`):
  `vol = std(daily log-returns over 30 trading days) × √252` (annualised).
  - `lagged_vol_30d` — 30 trading days strictly **before** the filing date.
  - `fwd_vol_30d` — 30 trading days strictly **after** the filing date (the label).
- **The central research question of Phase 5:** *do learned text representations (domain-adapted encoder +
  BERTopic topic vectors) beat a TF-IDF + lagged-volatility floor?*

---

## 1. Phase 1 — Dataset construction

### 1.1 Corpus and universe
- **Universe:** S&P 500 constituents (incl. delisted, to avoid survivorship bias).
- **Text:** SEC 10-K **Item 1A Risk Factors**, one pickle per filing (`datasets/sp500_1A/`, ~8,247 pickles,
  one string per filing).
- **Master table:** `datasets/feature_table.parquet`, **one row per filing (~8,105 filings)**, keyed by
  `(ticker, fiscal_year)` / `cik`. Date range spans roughly 2006–2026.

### 1.2 Construction scripts (`dataset_config/`)
| Script | Role |
|---|---|
| `build_filings_index.py` | EDGAR filing dates, SIC codes, fiscal year, `has_pickle` flag |
| `build_permno_linkage.py` | CIK → PERMNO/PERMCO/GVKEY (CRSP/Compustat link); dedup dual-class shares (GOOGL/GOOG, BRK.A/B) by keeping the valid-PERMNO row |
| `compute_volatility_labels.py` | `lagged_vol_30d` / `fwd_vol_30d` from daily prices |
| `build_feature_table.py` | joins the three above into `feature_table.parquet` (spine = pickle filings) |

### 1.3 Price sources (priority order, `compute_volatility_labels.py`)
1. **FINSABER** `all_sp500_prices_2000_2024_delisted_include.csv` (2000–2024), log-returns from
   `adjusted_close`.
2. **WRDS CRSP** `crsp_2025_daily.csv` (2025), `log(1 + DlyRet)`.
3. **yfinance** fallback for tickers with no PERMNO in CCM (`ETN`, `PARA`).

### 1.4 Temporal split (used identically by every later phase)
- **Train:** `filing_date < 2025-01-01` (2006–2024), ~7,353 filings.
- **Val:** `2025-01-01 ≤ filing_date < 2026-01-01` (2025), 406 filings.
- **Test:** `filing_date ≥ 2026-01-01` (2026) — held out; forward-vol window not yet closed at the time of
  the runs, so Phase 5 scores **val (2025)**.
- After requiring text + lagged + forward labels all present, the scorable set is **train ≈ 7,268 / val ≈ 397**.

---

## 2. Phase 2 — Domain-Adaptive Pretraining (DAPT)

**Goal:** adapt the encoder's token representations to financial-risk vocabulary by continued MLM, *before*
any supervised signal (following **Gururangan et al. 2020, "Don't Stop Pretraining", ACL**).

### 2.1 Base model — and why this one
- **`sentence-transformers/all-mpnet-base-v2`** (not `microsoft/mpnet-base`).
- Rationale (`dapt/README_dapt.md`): `all-mpnet-base-v2` already has good *sentence-level* geometry (trained
  on SNLI/multi-NLI etc.). The **MLM head is discarded after DAPT** — only the encoder carries forward — so
  preserving sentence geometry and reshaping the token vocabulary toward financial risk is the right
  trade-off. `microsoft/mpnet-base` is used **only** as the perplexity comparison baseline (it has a real
  general-English MLM head).

### 2.2 Chunking — the key Phase-2 design decision
Long Item 1A prose (median ~7k words, up to ~18k) must be split into ≤510-token MLM sequences. Two
strategies were built, both **within-document, non-overlapping, greedy** packing (the **RoBERTa
DOC-SENTENCES** family, Liu et al. 2019):

| Strategy | Unit | Code |
|---|---|---|
| Sentence-aware | NLTK `sent_tokenize`, greedy-pack sentences to 510 tok (oversized sentence → token window) | `chunk_corpus.py: sentence_pack` |
| **Paragraph-aware (selected)** | `\n\n` split = one risk factor, greedy-pack whole paragraphs to 510 tok (oversized paragraph → sentence-pack) | `chunk_corpus.py: paragraph_pack` |

- **Literature grounding (`Literature_agent.md`, Q1–Q6):** RoBERTa's controlled ablation found
  DOC-SENTENCES ≥ FULL-SENTENCES > SEGMENT-PAIR > SENTENCE-PAIR (+0.5–1 GLUE point for coherent packing);
  fixed-length windows were not even tested as a serious competitor. NSP dropped. Overlap/stride rejected
  (masking-leakage + diversity-loss; no pre-training work uses it). Corroborated by Cramming (Geiping &
  Goldstein 2022) and the domain precedents Legal-BERT (Chalkidis 2020), SEC-BERT/FiNER (Loukas 2022).
- **A/B result (`CHUNKING_COMPARISON.md`):** paragraph-aware reaches **lower val perplexity on the identical
  val set (2.2278 vs 2.2671, −1.7%)** despite ~18% lower fill-rate — its "never split a risk factor"
  property gives cleaner MLM context and aligns with the downstream BERTopic stage.
- **Honest caveats recorded:** the margin is small and confounded by ~22% more gradient steps (201,513 vs
  165,337 chunks at fixed 5 epochs). Both checkpoints retained for a possible Phase-5 ablation. Flagged for
  supervisor discussion (chunking choice; which perplexity baseline to report).
- **Chunk counts:** sentence-aware 165,337 train / 11,765 val (from 8,017 filings, 230 skipped — no
  feature-table match); paragraph-aware 201,513 train.

### 2.3 DAPT hyperparameters (`dapt/train_dapt.py`, `dapt/run_dapt.sh`)
| Parameter | Value | Reasoning |
|---|---|---|
| Base model | `all-mpnet-base-v2` | held constant across conditions |
| MLM masking rate | **0.15** | standard BERT/Gururangan |
| Learning rate | **2e-5** | standard DAPT lr |
| Warmup | **6%** of total steps (linear) | stability on a pre-trained model |
| Batch size | **16 × grad_accum 2 = effective 32** | fits A6000 48 GB at 512 tok |
| Epochs | **5**, early-stop patience 2 | val ppl fell every epoch → all 5 ran |
| Weight decay | 0.01 | standard |
| Precision | fp16 | speed on A6000 |
| Max seq length | 512 | MPNet limit |
| Best-checkpoint selection | min `eval_loss` (perplexity), `load_best_model_at_end` | HF `Trainer` + `EarlyStoppingCallback` |

### 2.4 Phase-2 results (val perplexity, `dapt_data/val.jsonl`)
| Model | Val PPL | Note |
|---|---|---|
| `microsoft/mpnet-base` (general-English MLM head) | 2.6669 | comparison baseline |
| sentence-aware DAPT, epoch 5 | 2.2671 | retained for ablation |
| **paragraph-aware DAPT, epoch 5** | **2.2278** | **selected encoder base for Phase 3** |

≈ **16.5% perplexity reduction** over `mpnet-base`; monotone decrease across all 5 epochs (2.83 → 2.49 →
2.36 → 2.29 → 2.27). Selected checkpoint: **`dapt_checkpoints_para/best/`**.

---

## 3. Phase 3 — Contrastive fine-tuning (SimCSE-style)

**Goal:** reshape the post-DAPT embedding space with a SimCSE-style in-batch-negatives objective, extending
**Chiu et al. (2025, EMNLP)**'s dual-view (lexical + chronological) framework with a third **sector** view.
Encoder architecture: `models.Transformer(max_seq_length=256)` → **mean Pooling** → **L2 Normalize**.

### 3.1 The three positive-pair views (`contrastive/build_pairs.py`)
| View | Anchor → Positive | Strength | Construction |
|---|---|---|---|
| **Lexical** | two random nested spans `[0:b]`, `[a:n]` (a<b) of the **same** paragraph | strong | Chiu's randomised-overlap scheme (overlap = `words[a:b]`, variable) |
| **Chronological** | **same firm, adjacent years (t, t+1)**, risk factors matched by **TF-IDF cosine ≥ 0.5** | medium | captures temporal persistence of a specific risk |
| **Sector** | two paragraphs from **different firms**, same **2-digit SIC + fiscal year** | soft | random within group, ≤50 pairs/group; same-(sic2,fy) doubles as the false-negative-masking label |

- **Unit = paragraph** (one risk factor), truncated/padded to **256 tokens** — matches Chiu, the closest
  precedent; sentence-level fragmentation would dilute risk semantics (`Literature_agent_phase3.md` Q5).
- **Shortcut normalisation** (`normalize_text`): numbers → `[NUM]` (SEC-BERT-NUM, Loukas 2022), dates →
  `[DATE]` (Chiu). Probabilistic firm/ticker blanking (Soares 2019 "Matching the Blanks") left **off** by
  default — naive ticker string-replace mangles common-word tickers (`A`, `ALL`, `IT`).
- **Pairs built from the TRAIN split only**; 2025 emitted as val units for monitoring but not paired; 2026
  excluded.

### 3.2 The loss — weighted InfoNCE with label-aware false-negative masking
Implemented as a **custom PyTorch loop** (not the stock SentenceTransformer trainer) because the loss needs
per-example `(sic2, fiscal_year, view)` metadata (`contrastive/train_contrastive.py`):

- **Objective:** InfoNCE / MNRL, cosine similarity scaled by **20.0 ⇒ τ = 0.05** (SimCSE default; the
  survey confirmed `sentence-transformers` MNRL `scale=20.0` already equals τ=0.05).
- **Label-aware false-negative masking (Khosla et al. 2020, Supervised Contrastive Learning):** for each
  anchor, any in-batch item sharing its `(sic2, fiscal_year)` is masked out of the negative set
  (`sim.masked_fill(false_neg, -inf)`). This is the survey's **most important fix** (Q3) — plain MNRL would
  manufacture false negatives once "same sector" is a *positive* axis. Available because the SIC label is
  known (the supervised solution, not estimation-based debiasing).
- **Per-view down-weighting (Q1):** `lambda_map = {lexical:1.0, chrono:1.0, sector:λ}` with
  **`λ_sector = 0.5`** — the soft cross-firm sector positive contributes proportionally less (Gao 2021
  pair-type weighting; Denize 2023 soft positives).

### 3.3 Three encoders trained (`contrastive/run_contrastive.sh`)
1. **`dual`** — lexical + chrono (Chiu replication / control).
2. **`three`** — + sector, `λ_sector 0.5`, full fine-tuning.
3. **`three_lora`** — + sector, **LoRA** (Q+V adapters, `r=16`, `α=2r=32`, dropout 0.1) — citation basis
   Hu et al. 2021 + LoRACode 2025 (the FinGPT r∈{8,16} justification was the wrong reference class).

### 3.4 Training hyperparameters (`train_contrastive.py`)
| Parameter | Value |
|---|---|
| Base | `dapt_checkpoints_para/best` |
| Max seq length | 256 |
| Batch size | 64 |
| LR | 2e-5 (AdamW) |
| Epochs | ≤30, early-stop patience 3, best on **val in-batch accuracy** |
| Pairs per view | 10,000 train / 500 val (balanced per view) |
| Scale (τ) | 20.0 (τ=0.05) |
| Pooling | mean + L2 norm |
| Precision | fp16 |
| Seed | 42 |

### 3.5 Phase-3 intrinsic results (FinMTEB, `contrastive/FINMTEB_RESULTS.md`)
Zero-shot on FinMTEB English (2 STS / Spearman, 10 Retrieval / NDCG@10):

| Model | STS mean | Retrieval mean |
|---|---|---|
| sbert (`all-mpnet-base-v2`) | 0.312 | **0.605** |
| **dapt** | **0.465** | 0.326 |
| dual | 0.406 | 0.497 |
| three | 0.337 | 0.419 |
| three_lora | 0.410 | 0.502 |

**Reading (as recorded):** DAPT lifts in-domain **STS** but breaks **retrieval geometry** (textbook MLM
anisotropy); contrastive FT then **repairs retrieval** (0.326 → ~0.50) — exactly its job. **LoRA is best of
our three** on both axes. **Sector view hurt under full fine-tuning** (`three` < `dual`) but was neutral
under LoRA. SBERT leads overall retrieval because it was trained on 1B+ generic retrieval pairs and
FinMTEB's tasks are out-of-distribution for Item-1A risk text — **beating SBERT here was never the goal**;
Phase 5 is the decisive test.

- **Sector-view diagnosis:** `build_sector` pairs at random with *no* similarity filter; measured intra-pair
  TF-IDF cosine — chrono 0.86 / lexical 0.56 / **sector 0.25 (38% of pairs < 0.15, near-unrelated)**. Under
  full FT this noisy signal blurs the geometry (`three` val acc 0.727 vs `dual` 0.899); LoRA's low rank
  contains it. Decision: document and carry all encoders into Phase 5.

---

## 4. Phase 4 — BERTopic topic modelling

**Goal:** the *explanation* contribution — which risk themes drive a firm's volatility — and a per-filing
topic-exposure feature for Phase 5.

### 4.1 Document set (`topics/build_topic_docs.py`)
- Same paragraph extraction + `[NUM]`/`[DATE]` normalisation as `build_pairs.py` (no divergence), but
  **keeps the 2026 test filings** so the forward-looking benchmark can score test later.
- One JSONL doc per paragraph with `(uid, text, ticker, cik, fiscal_year, sic2, filing_date, split)`.

### 4.2 BERTopic configuration (`topics/fit_topics.py`)
- **Fit on TRAIN docs only**, then `transform()` val + test — **leakage-safe**: no held-out text shapes the
  topic space.
- Encoder embeddings precomputed and cached (`emb_<enc>.npy`), passed to BERTopic with
  `embedding_model=None` so the *same* vectors drive both clustering and the Phase-5 dense ablation.
- **UMAP:** `n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine, random_state=42`.
- **HDBSCAN:** `min_cluster_size=200, metric=euclidean, cluster_selection_method=eom`.
- **c-TF-IDF vectoriser:** `CountVectorizer(stop_words=english, ngram_range=(1,2), min_df=10)`.
- **Topic reduction:** `reduce_topics(nr_topics=50)` → **49 topics** for every encoder (targets the report's
  20–80 band).
- **Per-filing topic vector** = fraction of the filing's non-outlier paragraphs in each topic (`t0..t48`),
  plus `n_paras`, `outlier_frac`.

### 4.3 Topic quality (`topics/out/topic_quality.json`)
C_v coherence (gensim, single-process + subsampled — the multiprocessing path deadlocked a 48h job for ~5h;
deliverables written *before* coherence so a failure can't cost the Phase-5 inputs) and diversity (Dieng
2020), all at 49 topics:

| Encoder | C_v | Diversity |
|---|---|---|
| sbert | 0.716 | 0.769 |
| dual | 0.673 | 0.722 |
| three | 0.696 | 0.639 |
| three_lora | 0.682 | 0.708 |

---

## 5. Phase 5 — Baselines + the decisive volatility benchmark

All tiers share the **same filings, split, log-target, and `metrics()`** so the numbers slot into one
ablation table. Predictors fit on **train only**, scored on **val 2025**.

### 5.1 Baseline tiers (`baselines/run_baselines.py`)
| Tier | Model |
|---|---|
| 1a naive persistence | `y_hat = log(lagged_vol_30d)` (random walk, 0 params) |
| 1b AR(1) | RidgeCV `log(fwd) ~ log(lagged)` |
| 2 TF-IDF | RidgeCV `log(fwd) ~ tfidf(text)` |
| 2+ TF-IDF + lagged | RidgeCV `log(fwd) ~ tfidf + log(lagged)` — **does text add over persistence?** |

- **TF-IDF vectoriser:** `stop_words=english, ngram_range=(1,2), min_df=5, max_features=20000,
  sublinear_tf=True`. **RidgeCV** over `alphas = logspace(-2, 3, 20)`.

### 5.2 Learned tiers (`phase5/run_phase5.py`), per encoder × condition × head
- **Tier-3a encoder** — mean-pooled 768-d filing vector.
- **Tier-3b topic** — the BERTopic `t0..t48` exposures.
- **Tier-3c hybrid** — `[mean-pooled encoder | topic vector | log_lagged]`.
- Each through **RidgeCV** (`alphas` as above) **and** a small MLP (`hidden=(64,), alpha=1.0,
  early_stopping`), all on `StandardScaler` fit on train only.

### 5.3 Headline results on val 2025 (n≈397)
**Baselines:**
| Model | R²_log | Spearman |
|---|---|---|
| tier1a naive persistence | −0.030 | 0.466 |
| tier1b AR(1) | 0.031 | 0.466 |
| tier2 TF-IDF | −0.032 | **0.6025** |
| **tier2+ TF-IDF + lagged** | **0.1757** | 0.5604 |

**Learned (Ridge head, Spearman / R²_log):**
| Encoder | tier3a encoder | tier3b topic | tier3c hybrid |
|---|---|---|---|
| sbert | 0.441 / −0.12 | 0.240 / −0.27 | 0.525 / 0.10 |
| dapt | 0.463 / −0.10 | — | — |
| dual | 0.430 / −0.18 | 0.294 / −0.20 | **0.545 / 0.13** |
| three | 0.350 / −0.20 | 0.303 / −0.18 | 0.481 / 0.10 |
| three_lora | 0.421 / −0.18 | 0.264 / −0.22 | 0.537 / 0.13 |

### 5.4 The conclusion the original pipeline reached
- **TF-IDF is the strongest text representation.** Tier-2 TF-IDF ranks best on Spearman (**0.6025**), and
  no learned encoder/topic condition beats the **TF-IDF + lagged floor** (R²_log **0.176**, Spearman
  **0.56**). The best learned condition (`dual` hybrid) reaches Spearman 0.545 / R²_log 0.13 — below the
  floor on both.
- **Lagged vol carries the R².** Text alone has ≈0 R²_log (the level of volatility is set by persistence);
  text's value is in **rank** (Spearman), where TF-IDF leads.
- This is the result that motivated the later redesign: a bag-of-words baseline beat every domain-adapted
  neural encoder on this task, across five independent encoder conditions.

---

## 6. Compute / infrastructure

- **SLURM Teaching partition**, 48 h wall-time, `cpus-per-task=2`, `mem=32G`.
- **GPU:** 1× RTX A6000 (48 GB) for DAPT, contrastive, and Phase-4 encode/topic jobs.
- Environments: `diss` (main), `finmteb` (intrinsic eval). W&B logging optional (gated on `WANDB_API_KEY`).
- Datasets on a private HF repo (`SarthakVishnu/dissertation-dataset`); the compute node has no internet, so
  `hf download` is run from a login node.

---

## 7. Key citations adopted (and what was taken from each)

| Paper | Used for |
|---|---|
| Gururangan et al. 2020 (DAPT, ACL) | continued-MLM domain adaptation; RoBERTa FULL/DOC-SENTENCES packing |
| Liu et al. 2019 (RoBERTa) | the chunking ablation — DOC-SENTENCES greedy packing, no NSP, no overlap |
| Devlin et al. 2019 (BERT) | 15% MLM masking, 512-token sequences |
| Loukas et al. 2022 (FiNER / SEC-BERT) | number → `[NUM]` normalisation; closest SEC-filing precedent |
| Chalkidis et al. 2020 (Legal-BERT) | 512-token non-overlapping chunks for long legal/regulatory prose |
| Geiping & Goldstein 2022 (Cramming) | non-overlapping sentence packing + max token utilisation |
| Chiu et al. 2025 (Dual-view, EMNLP) | **direct parent** — paragraph/256-tok, lexical + chronological views, InfoNCE in-batch, batch 64, lr 2e-5, mean-pool |
| Khosla et al. 2020 (SupCon) | label-aware false-negative masking on `(sic2, fiscal_year)` |
| Gao et al. 2021 (SimCSE) | τ=0.05; pair-type weighting precedent |
| Denize et al. 2023 (SCE) | soft/down-weighted positives → `λ_sector < 1` |
| Hu et al. 2021 (LoRA) + LoRACode 2025 | LoRA target Q+V, r=16, α=2r, dropout 0.1 (correct reference class) |
| Soares et al. 2019 (Matching the Blanks) | entity blanking rationale (left off by default; flagged) |
| Tang & Yang 2025 (FinMTEB) | intrinsic benchmark; domain models > generic on financial tasks |
| Jehnen et al. 2026 (FinTextSim) | analogous 10-K → sentence-transformer → BERTopic precedent |

Full per-question grounding (with verdicts and open verification notes) is in `Literature_agent.md` (Phase 2
segmentation, Q1–Q6) and `Literature_agent_phase3.md` (Phase 3 contrastive design, Q1–Q5).

---

## 8. Open items the original pipeline flagged for supervisor discussion
1. **Chunking choice** — paragraph-aware selected on a −1.7% perplexity edge confounded by ~22% more steps;
   confirm vs a step-matched re-run or deferral to a downstream ablation.
2. **Perplexity baseline** — report `microsoft/mpnet-base` (2.67) vs the literal random-head start point.
3. **Sector view** — negative under full fine-tuning, neutral under LoRA; cheap fixes are a TF-IDF
   similarity floor on `build_sector` or a masking-only sector variant.
4. **The headline negative** — TF-IDF beats every neural encoder on the volatility task; this is the finding
   that prompted the subsequent redesign.
