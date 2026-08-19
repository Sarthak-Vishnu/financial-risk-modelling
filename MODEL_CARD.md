# Model Card — SEC Item 1A risk-factor encoders

Encoders trained in this project for predicting forward realised volatility from the "Risk Factors"
(Item 1A) section of SEC 10-K filings. Covers the domain-adaptively pretrained model, the three
contrastive variants built on it, and the two task-aligned (volatility-supervised) variants.

**Environment.** The `diss` conda environment (`environment.yml`, `requirements.txt`); FinMTEB runs in a
separate `finmteb` environment. Seed 42 is set for contrastive training and for the TF-IDF→SVD
projection. The downstream grid is reproducible, but only when both the thread count and the CPU are
held fixed. `phase5/run_val2025.sh` therefore pins every thread pool to 1 and constrains the job to
`damnii07`; run on that configuration, the grid reproduces bit-for-bit on IC and on both bootstrap
CI bounds, and R²_log and the DM p-value agree to within 7.4e-06 and 5.0e-04 respectively. Relax
either control and the tree-head lanes move by up to 0.018 IC, with the SVD-projected ridge lane
moving 0.011 — enough to reorder encoders and to carry a DM p-value across a conventional
threshold. Seeds are not the issue; everything is seeded. Both controls are requirements, not
preferences.

---

## 1. Model details

### 1.1 Base model

| | |
|---|---|
| Base | `sentence-transformers/all-mpnet-base-v2` (`dapt/train_dapt.py:33`) |
| Architecture | `MPNetModel`, 12 layers, hidden size 768 (`config.json` of each checkpoint) |
| Sentence-transformer modules | Transformer → Pooling → Normalize (`modules.json`; the `ftvol2018` checkpoint has Transformer → Pooling only, no Normalize module) |
| Embedding dimension | 768 |
| Base model licence | Not recorded in this repository |

### 1.2 Checkpoints in this repository

| Tag | Path | Base | Stage |
|---|---|---|---|
| `dapt` (sentence-aware) | `dapt_checkpoints/best` | all-mpnet-base-v2 | Phase 2 DAPT, MLM |
| `dapt` (paragraph-aware) | `dapt_checkpoints_para/best` | all-mpnet-base-v2 | Phase 2 DAPT, MLM — **the selected base for Phase 3** |
| `dual` | `contrastive_checkpoints/dual` | `dapt_checkpoints_para/best` | Phase 3 contrastive, 2 views |
| `three` | `contrastive_checkpoints/three` | `dapt_checkpoints_para/best` | Phase 3 contrastive, 3 views |
| `three_lora` | `contrastive_checkpoints/three_lora` | `dapt_checkpoints_para/best` | Phase 3 contrastive, 3 views, LoRA |
| `volaware` | `contrastive_checkpoints/volaware` | `dapt_checkpoints_para/best` | Stage B contrastive, 4 views incl. volatility, LoRA |
| `ftvol` | `contrastive_checkpoints/ftvol` | `dapt_checkpoints_para/best` | Supervised volatility regression |
| `ftvol2018` | `contrastive_checkpoints/ftvol2018` | `dapt_checkpoints_para/best` | Supervised volatility regression, leakage-clean protocol |

All contrastive and task-aligned checkpoints are saved as complete SentenceTransformer directories
with full merged weights (`model.safetensors`, 437,967,672 bytes for `dual` / `three` /
`three_lora`). The LoRA-trained variants (`three_lora`, `volaware`) contain no `adapter_config.json`
or `adapter_model.safetensors` — the adapters were merged at save time, so they load with a plain
`SentenceTransformer(path)` identically to the fully fine-tuned checkpoints.

### 1.3 Versioning

No version tags, release numbers or model hashes are recorded in this repository. Checkpoints are
identified by directory path and filesystem timestamp only. There is no `.gitattributes` LFS
tracking or model registry entry; the checkpoint directories are not in git.

---

## 2. Training data

### 2.1 Source corpus

| | |
|---|---|
| Universe | S&P 500 constituents, **including delisted firms** (survivorship-bias-free) |
| Text | SEC 10-K Item 1A "Risk Factors", one pickle per filing |
| Corpus size | ~8,247 filing pickles |
| Master feature table | `datasets/feature_table_fixed.parquet`, 8,105 rows, 16 columns, keyed by ticker and fiscal year |
| Unique tickers | 485 |
| Filing-date coverage | 2006-02-22 to 2026-03-02 |
| Fiscal-year coverage | 2005 to 2025 |
| Modelling panel | 7,367 filings, filing years 2006 to 2025 |

Filings per year in the modelling panel: 2006: 83, 2007: 295, 2008: 320, 2009: 326, 2010: 321,
2011: 322, 2012: 330, 2013: 345, 2014: 369, 2015: 387, 2016: 401, 2017: 408, 2018: 424, 2019: 429,
2020: 428, 2021: 444, 2022: 452, 2023: 447, 2024: 443, 2025: 393.

### 2.2 Label and market data

The prediction target is the standard deviation of daily log returns over the 30 trading days
strictly after the filing date, annualised by √252, modelled in natural logs. Daily prices come
from three sources in priority order:

1. FINSABER S&P 500 price file (2000-2024, including delisted names) — 4,736,109 rows
2. WRDS CRSP daily returns for 2025 — 159,092 rows

### 2.3 Temporal split

Used identically by every phase:

- **Train**: filings dated before 2025-01-01 (~7,353 filings)
- **Validation**: 2025 filings (406 filings, ~397 with complete labels)
- **Test**: 2026, held out — forward windows had not closed at the time of the experiments

All model selection and fitting uses training years only. Contrastive pairs are built from the
train split only; 2025 filings are emitted as validation units for monitoring but never paired.

### 2.4 Chunking for MLM

Within-document, non-overlapping greedy packing to ≤510 tokens (RoBERTa DOC-SENTENCES family). Two
strategies were built and compared:

| | Sentence-aware | Paragraph-aware |
|---|---|---|
| Atomic unit | Sentence (NLTK `sent_tokenize`) | Paragraph (`\n\n` = one Item 1A risk factor) |
| Train sequences | 165,337 | 201,513 (+21.9%) |
| Validation sequences | 11,765 | Not recorded |
| Source filings | 8,017 (230 skipped, no feature_table match) | Not recorded separately |
| Mean tokens/chunk | 453.4 | 372.0 |
| Median tokens/chunk | 489.0 | 455.0 |
| Chunks ≥480 tokens | 62.8% | 36.9% |

### 2.5 Preprocessing for contrastive pairs

Numbers are replaced with `[NUM]` and dates with `[DATE]` (SEC-BERT-NUM convention) to suppress
shortcut learning. Firm/ticker blanking is deliberately **off**: naive ticker string-replacement
mangles common-word tickers such as "A", "ALL" and "IT". Probabilistic entity blanking is recorded
as deferred and was not implemented.

---

## 3. Training procedure

### 3.1 Phase 2 — domain-adaptive pretraining (MLM)

Continued masked-language-model pretraining following Gururangan et al. (2020).

| Hyperparameter | Value |
|---|---|
| Objective | MLM, masking probability 0.15 |
| Epochs | 5 |
| Batch size | 16, gradient accumulation 2 (effective 32) |
| Learning rate | 2e-5 |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Precision | fp16 |
| Early stopping | patience 2, on validation loss |
| Hardware | 1× NVIDIA RTX A6000 |

Both chunking variants used identical hyperparameters, so chunking was the only varying factor.

### 3.2 Phase 3 — contrastive fine-tuning

Custom PyTorch loop (not the stock SentenceTransformer trainer), because the loss needs per-example
weighting and label-aware false-negative masking.

| Hyperparameter | Value |
|---|---|
| Loss | Weighted InfoNCE, cosine similarity scaled by 20.0 (τ = 0.05, SimCSE/MNRL default) |
| False-negative handling | Label-aware masking on (2-digit SIC, fiscal year) keys, after Khosla et al. (2020) |
| Epochs | 30 |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Early stopping | patience 3 |
| Pairs per view | 10,000 train, 500 validation |
| Sector-view weight (λ) | 0.5 |
| Volatility-view weight (λ) | 1.0 (`volaware` only) |
| LoRA | rank 16, targets q and v (`three_lora`, `volaware`) |
| Precision | fp16 |
| Seed | 42 |
| Max sequence length | Set by `MAX_LEN` in `train_contrastive.py`; the checkpoints' `sentence_bert_config.json` records `max_seq_length: None` |
| Hardware | 1× NVIDIA RTX A6000 |

**Views (positive-pair definitions):**

- **lexical** — two random nested spans of the *same* paragraph (after Chiu et al. 2025)
- **chronological** — same firm, adjacent years (t, t+1), risk factors matched by TF-IDF cosine;
  captures temporal persistence of a specific risk
- **sector** — two paragraphs from *different* firms sharing 2-digit SIC and fiscal year; a soft
  positive, down-weighted to λ=0.5. **This third view is the project's own contribution over the
  two-view prior work it replicates.**
- **vol** (`volaware` only) — paragraphs from different firms in the same within-year forward-volatility decile

**Per-checkpoint training outcomes** (from each `train_summary.json`):

| Checkpoint | Views | LoRA | Best val pair-accuracy | Best epoch |
|---|---|---|---|---|
| `dual` | lexical, chrono | no | 0.8990 | 3 |
| `three` | lexical, chrono, sector | no | 0.7273 | 13 |
| `three_lora` | lexical, chrono, sector | yes | 0.6473 | 15 |
| `volaware` | lexical, chrono, sector, vol | yes | 0.5005 | 17 |

> **Caveat, and this one is mine rather than the repository's:** these accuracies are *not*
> comparable across rows. Each configuration is scored on its own view mixture, and adding views
> makes the retrieval task harder, so the descending column reflects changing task difficulty at
> least as much as changing model quality. The repository does not record a common-task
> re-evaluation of the four checkpoints, so there is no recorded number that makes them comparable.

### 3.3 Task-aligned supervised fine-tuning

Paragraph-level regression onto the filing's log forward volatility, so embeddings become
volatility-discriminative rather than only similarity-smooth.

**`ftvol`** — base `dapt_checkpoints_para/best`, 3 epochs, batch 64, lr 2e-5, fp16, no LoRA.
Trained on 184,177 paragraphs, validated on 4,953 paragraphs from 397 filings. Per-epoch validation
filing-level IC: epoch 1 = 0.4327, epoch 2 = 0.4556, epoch 3 = 0.4439. No `train_summary.json`
exists for this checkpoint. The checkpoint was originally saved without tokenizer/module files and
was repaired by hand on 2026-07-03.

**`ftvol2018`** — the leakage-clean retrain. Trained on filing years before 2017, epoch selected on
2017, never exposed to 2018 or later, so the 2018-2024 expanding-window backtest is leakage-free by
construction.

| Field | Value |
|---|---|
| Base | `dapt_checkpoints_para/best` |
| Epochs | 20 |
| Batch size | 128 |
| Train end | 2018 |
| Selection year | 2017 |
| Best validation filing IC | 0.634639 |
| Target standardisation | mean −1.363124, sd 0.554488 |
| LoRA | no |
| Corrected data | yes (`legacy_data: false`) |

---

## 4. Intended use

**In scope.** Producing filing-level or paragraph-level dense representations of SEC 10-K Item 1A
risk-factor prose, for cross-sectional ranking research on forward realised volatility across
large-cap US equities. Reproducing or contesting this study's finding that count-based TF-IDF
representations outperform these learned encoders on that task.

**Out of scope.**

- **Trading, portfolio construction or risk management.** These are research artifacts evaluated on
  a single held-out year plus a walk-forward backtest, on one universe, with no transaction-cost,
  capacity or execution modelling.
- **`ftvol` in any backtest.** It was trained on all pre-2025 labels and epoch-selected on validation
  2025, so it is inadmissible for any pre-2025 test window, and its validation-2025 rows are
  selection-inflated. `ftvol2018` exists specifically to be the admissible version.
- **Non-S&P-500, non-US, or non-10-K text.** The universe is S&P 500 constituents and the register
  is US regulatory risk-factor prose; nothing here evidences transfer beyond that.
- **General-purpose financial retrieval or semantic search.** FinMTEB retrieval scores are below the
  off-the-shelf SBERT baseline for every model trained here (section 5.1).
- **Any use treating a volatility forecast as a forecast of returns, direction or firm quality.**

---

## 5. Evaluation

### 5.1 Intrinsic — FinMTEB

FinMTEB (Tang & Yang 2025), English subset: 2 STS tasks scored by Spearman ρ and 10 retrieval tasks
scored by NDCG@10. Run by `contrastive/eval_finmteb.py`, ~49 minutes on an A6000.
The optional 7B `Fin-E5` baseline and the Chinese tasks were skipped.

| Model | FinSTS | FINAL | FiQA2018 | Apple10K | FinQA | FinanceBench | HC3 | TATQA | TheGoldmanEn | TradeTheEventEncy | USNews | TradeTheEventNews | MEAN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sbert` | 0.2148 | 0.4099 | 0.4975 | 0.8524 | 0.1014 | 0.7358 | 0.6279 | 0.1876 | 0.3864 | 0.9825 | 0.8179 | 0.8622 | 0.5563 |
| `dapt` | 0.3505 | 0.5802 | 0.1820 | 0.6563 | 0.0434 | 0.1851 | 0.3037 | 0.0941 | 0.1871 | 0.7153 | 0.4220 | 0.4735 | 0.3495 |
| `dual` | 0.2980 | 0.5136 | 0.2941 | 0.7519 | 0.0760 | 0.6675 | 0.4322 | 0.1497 | 0.3141 | 0.9160 | 0.6827 | 0.6896 | 0.4821 |
| `three` | 0.2152 | 0.4591 | 0.2133 | 0.6536 | 0.0499 | 0.5396 | 0.3355 | 0.1151 | 0.2801 | 0.7034 | 0.6570 | 0.6415 | 0.4053 |
| `three_lora` | 0.3032 | 0.5163 | 0.2849 | 0.7330 | 0.0776 | 0.6547 | 0.4252 | 0.1464 | 0.3076 | 0.8713 | 0.7316 | 0.7892 | 0.4868 |

> The `MEAN` column is reproduced as recorded in `eval_results/finmteb/summary.json`, but the
> repository explicitly warns that it averages Spearman and NDCG@10 together and is therefore not a
> meaningful single score. The first two columns are STS; the remaining ten are retrieval.

Recorded reading of this table: DAPT raises in-domain similarity (STS) while lowering retrieval — the
expected MLM anisotropy side effect — and contrastive fine-tuning then repairs retrieval. `three_lora`
is the best of the three project encoders on both axes. Adding the sector view under full fine-tuning
*hurt* (`three` < `dual`), but under LoRA it did not.

### 5.2 Intrinsic — BERTopic topic quality

Topic coherence (C_v) and topic diversity (Dieng et al. 2020) over topic models fitted on train
documents only, 49 topics each, `min_cluster_size` 200, `nr_topics` 50.

| Encoder | Topics | C_v | Diversity |
|---|---|---|---|
| `sbert` | 49 | 0.7161 | 0.7694 |
| `three` | 49 | 0.6962 | 0.6388 |
| `three_lora` | 49 | 0.6816 | 0.7082 |
| `dual` | 49 | 0.6726 | 0.7224 |
| `volaware` | 49 | 0.6704 | 0.6694 |

No topic model was fitted over the `ftvol`, `ftvol2018` or `bge` embedding spaces. `dapt` was
deliberately excluded: its raw MLM embeddings are severely anisotropic (mean pairwise cosine 0.60
against sbert's 0.36), which made UMAP non-convergent — a previous run hung roughly 10 hours on
dapt clustering until wall-time killed it.

### 5.3 Intrinsic — MLM perplexity

Both DAPT checkpoints on the identical original validation set (lower is better):

| Checkpoint | Perplexity |
|---|---|
| Sentence-aware (`dapt_checkpoints/best`) | 2.2671 |
| Paragraph-aware (`dapt_checkpoints_para/best`) | 2.2278 |

Each on its own validation set (convergence check, not directly comparable): sentence-aware 2.2660,
paragraph-aware 2.2393.

Paragraph-aware was selected, on a 1.7% perplexity improvement and on the property that it never
splits a risk factor across chunks. **Recorded confound:** paragraph-aware produced ~22% more chunks
and therefore saw ~22% more gradient steps over the same 5 epochs.

### 5.4 Downstream — forward volatility, validation 2025

Each encoder's mean-pooled filing embedding added on top of the structured financial baseline, under
a ridge and a histogram gradient-boosted tree head. Metric is within-year cross-sectional Spearman
IC on validation 2025 (n = 393), with a bootstrap 95% CI, R² on the log level, and a Diebold-Mariano
p-value against the `struct+tfidf [sparse]` reference.

| Condition | IC | 95% CI | R²_log | DM p vs ref |
|---|---|---|---|---|
| `struct+tfidf [sparse]` (reference) | 0.6032 | [0.5344, 0.6699] | 0.2260 | — |
| `structured [ridge]` (no text) | 0.5912 | [0.5221, 0.6592] | 0.1747 | 1.25e-07 |
| `structured [hgb]` | 0.5584 | [0.4873, 0.6277] | 0.0939 | 0.000645 |
| `lagged [hgb]` | 0.4573 | [0.3744, 0.5409] | −0.1299 | 1.06e-11 |
| `tfidf+lag [sparse]` | 0.5075 | [0.4277, 0.5852] | 0.0686 | 7.83e-05 |
| `struct+tfidf_svd [ridge]` | 0.5936 | [0.5263, 0.6609] | 0.1668 | 9.22e-05 |
| `struct+tfidf_svd [hgb]` | 0.5693 | [0.4990, 0.6386] | 0.1675 | 0.123 |
| `struct+enc[dual] [ridge]` | 0.5819 | [0.5089, 0.6515] | 0.1618 | 0.0132 |
| `struct+enc[dual] [hgb]` | 0.5823 | [0.5111, 0.6508] | 0.1599 | 0.0507 |
| `struct+enc[sbert] [ridge]` | 0.5618 | [0.4819, 0.6359] | 0.0170 | 8.62e-14 |
| `struct+enc[sbert] [hgb]` | 0.5914 | [0.5214, 0.6577] | 0.1494 | 0.0434 |
| `struct+enc[volaware] [ridge]` | 0.5866 | [0.5144, 0.6557] | 0.1475 | 0.00287 |
| `struct+enc[volaware] [hgb]` | 0.5972 | [0.5303, 0.6627] | 0.2072 | 0.593 |
| `struct+enc[three] [ridge]` | 0.5638 | [0.4889, 0.6337] | 0.0998 | 2.27e-06 |
| `struct+enc[three] [hgb]` | 0.5999 | [0.5325, 0.6618] | 0.2446 | 0.587 |
| `struct+enc[three_lora] [ridge]` | 0.5884 | [0.5174, 0.6593] | 0.1453 | 0.000862 |
| `struct+enc[three_lora] [hgb]` | 0.5958 | [0.5269, 0.6608] | 0.1985 | 0.430 |
| `struct+enc[ftvol] [ridge]` | 0.6058 | [0.5404, 0.6725] | 0.1849 | 0.207 |
| `struct+enc[ftvol] [hgb]` | 0.5979 | [0.5293, 0.6658] | 0.1670 | 0.155 |
| `struct+enc[bge] [ridge]` | 0.5800 | [0.5062, 0.6486] | 0.0932 | 1.39e-07 |
| `struct+enc[bge] [hgb]` | 0.5835 | [0.5121, 0.6509] | 0.1449 | 0.0232 |
| `struct+change [hgb]` | 0.5565 | [0.4839, 0.6295] | 0.0836 | 0.000668 |
| `struct+tfidf+change [sparse]` | 0.6090 | [0.5418, 0.6723] | 0.2195 | 0.355 |

`three` and `three_lora` were encoded against the corrected corpus on 2026-08-11
(`topics/run_encode_three.sh`) and entered the grid the next day; before that they had no downstream
score at all, because `run_encode.sh` never covered them and `mean_pooled_filings` therefore returned
`None`. Their rows above are the first corrected-data downstream numbers for the project's own
three-view contribution.

Neither reaches parity with the count-based reference under the ridge head — `three` at 2.27e-06 is
among the worst in the table, `three_lora` at 0.000862 little better. Under the tree head both are
statistically indistinguishable from it, `three` at DM p 0.587 and `three_lora` at 0.430, but so are
`dual`, `volaware` and `ftvol`. The two conditions that *fail* parity under the tree head are
`sbert` (0.0434) and `bge` (0.0232), the two general-purpose off-the-shelf encoders. On this
evidence the separation runs along domain adaptation rather than along task alignment, and no
encoder beats TF-IDF.

`struct+enc[three] [hgb]` has the highest R²_log in the table at 0.2446, above the reference's
0.2260, while its IC of 0.5999 sits mid-pack. That is the one respect in which the sector view leads,
and it is a level-accuracy gain rather than a ranking gain.

The `ftvol` rows above are the selection-inflated ones described in section 4 and must not be read
as clean estimates.

**Encoder-selection rows.** The grid's `EVERYTHING` conditions and its attention-pooling rows pick
their encoder by maximising HGB IC *on the same validation rows they are then scored on*. `three`
takes that selection at 0.5999 against `ftvol`'s 0.5979, a margin of 0.0021, and the pooling rows go
to those same top two. The selected-encoder identity in these row names is a within-sample artefact
and carries no evidential weight.

| Condition | IC | 95% CI | R²_log | DM p vs ref |
|---|---|---|---|---|
| `struct+enc[three,risk_weighted] [hgb]` | 0.5997 | [0.5307, 0.6642] | 0.2394 | 0.700 |
| `struct+enc[three,topk_risk] [hgb]` | 0.5866 | [0.5165, 0.6556] | 0.2180 | 0.810 |
| `struct+enc[ftvol,risk_weighted] [hgb]` | 0.6040 | [0.5377, 0.6724] | 0.1848 | 0.324 |
| `struct+enc[ftvol,topk_risk] [hgb]` | 0.6074 | [0.5385, 0.6773] | 0.1944 | 0.454 |
| `EVERYTHING svd+enc[three]+chg [hgb]` | 0.6084 | [0.5421, 0.6732] | 0.2150 | 0.777 |
| `EVERYTHING tfidf+enc[three]+chg [sparse]` | 0.5535 | [0.4764, 0.6231] | −0.0038 | 1.08e-07 |

Neither `EVERYTHING` row carries topic exposures. No corrected-data condition anywhere in this
repository fuses topic exposures with another feature block: no
`filing_topic_vectors_*_fixed.parquet` exists for any encoder, so `topic_filings()` returns `None`
and the topic block is never appended. This gap is left open deliberately and is not closed by the
2026-08-12 run.

### 5.5 Downstream — expanding-window backtest

The only encoder admissible for the 2018-2024 walk-forward backtest is `ftvol2018`. Paired
across-years ΔIC against `struct+tfidf [sparse]`:

| Condition | ΔIC | p |
|---|---|---|
| `struct+enc[ftvol2018] [ridge]` | −0.057 | 0.0874 |
| `struct+enc[ftvol2018] [hgb]` | −0.064 | 0.0965 |

That is, the leakage-clean supervised encoder underperforms the count-based reference. No other
encoder has a recorded backtest result.

---

## 6. Ethical and licensing considerations

**Source filings.** SEC 10-K filings are public regulatory disclosures retrieved from EDGAR. They
describe corporate entities rather than natural persons, so the corpus carries no personal data of
the kind that would raise subject-rights concerns — with the caveat that risk-factor text
occasionally names individual executives. No licence file, terms-of-use record or attribution
requirement for EDGAR retrieval is recorded in this repository. EDGAR's own access policies
(including rate limits and user-agent requirements) are not documented here.

**Market data.** The FINSABER S&P 500 price file, WRDS CRSP daily returns and Compustat
fundamentals are third-party datasets with their own licences. **No licence terms, redistribution
conditions or entitlement records for any of them are recorded in this repository.** CRSP and
Compustat are accessed through WRDS, which is normally an institutional subscription with
redistribution restrictions — meaning the derived label and feature tables may not be freely
redistributable.

**Base model and dependencies.** `sentence-transformers/all-mpnet-base-v2` and
`BAAI/bge-base-en-v1.5` are third-party models with their own licences.

**Fairness.** The universe is large-cap US firms. Nothing here evidences behaviour on smaller
issuers, non-US filers, or firms in sectors thinly represented in the S&P 500. No subgroup
performance breakdown is recorded.

---

## 7. Reproduction

| Stage | Entry point |
|---|---|
| Chunk corpus for MLM | `dapt/chunk_corpus.py --paragraph_aware --out_dir dapt_data_para` |
| DAPT | `dapt/run_dapt_para.sh` |
| Build contrastive pairs | `contrastive/run_build_pairs.sh` |
| Contrastive fine-tune | `contrastive/run_contrastive.sh` |
| Vol-aware contrastive | `contrastive/run_volaware.sh` |
| Supervised vol fine-tune | `contrastive/run_finetune_vol.sh`, `contrastive/run_ftvol2018.sh` |
| Encode corrected corpus | `topics/run_encode.sh`, `topics/run_encode_three.sh` |
| Fit topic models | `topics/run_topics.sh` |
| FinMTEB | `contrastive/run_finmteb.sh` |
| Downstream grid + backtests | `phase5/run_stress.sh` |
| Downstream grid only | `phase5/run_val2025.sh` |
