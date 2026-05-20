# Sunnie Feedback Analysis — Feedback 01 (Prof Tiejun Ma)

> **Context:** Feedback received after submitting the IPP draft (S2880814_IPP20_Report.pdf, April 2026).
> Analysis covers each paragraph of the Sunnie's reply, with a verdict on correctness and a comparison against the current report.

---

## Sunnie's Full Reply

> Hi, I've just got a detailed look into the proposal. I think the project will be strongest if it is framed as a forward-looking volatility prediction benchmark from SEC Item 1A risk disclosures, rather than as a broad financial NLP system.
>
> The key question can be: does Item 1A text add incremental predictive value for future realised volatility beyond lagged volatility and standard financial controls?
>
> To make the benchmark convincing, I would include three groups of baselines: first, non-text volatility baselines such as AR(1) / lagged realised volatility and historical volatility; second, simple text baselines such as TF-IDF and Loughran-McDonald dictionary features; third, representation baselines such as SBERT, E5/BGE, FinBERT, BERTopic, a Chiu-style dual-view contrastive encoder, and your proposed three-view encoder.
>
> I would also separate the risk representation module from the volatility prediction head. The thesis can evaluate 30-day forward volatility prediction, but the reusable output should be a firm-filing feature table containing CIK, PERMNO, filing date, SIC/industry, dense risk embeddings, topic exposure vectors, lagged volatility, and future realised volatility labels.
>
> This keeps the project manageable while making the contribution stronger: a reusable Item 1A risk representation module validated by a strict forward-looking volatility benchmark.

---

## Paragraph 1 — Framing: "Volatility Prediction Benchmark" vs. "Broad Financial NLP System"

> *"The project will be strongest if it is framed as a forward-looking volatility prediction benchmark from SEC Item 1A risk disclosures, rather than as a broad financial NLP system."*

### Is it correct?
**Yes — and it is the sharpest criticism.** The supervisor is identifying a framing diffusion problem. Right now the project has two centres of gravity simultaneously: (a) building a better encoder (NLP contribution), and (b) predicting volatility (finance contribution). When a project tries to be both, neither claim is as defensible as it could be. A benchmark framing is stronger because it gives the work a single falsifiable question and a clear comparator hierarchy.

### How it differs from the current report
In the current report, the primary language positions the encoder as the product:
- Abstract: *"developing a hybrid, domain-adapted sentence encoder"*
- Phase 3 output: *"fine-tuned encoder + FinMTEB intrinsic scores"* (Deliverable D₂)
- Phase 5 is presented as downstream validation of the encoder, not as the central question

Volatility prediction is real in the design (Phase 5, Ridge/MLP, 30-day forward vol), but it is framed as evidence *for the encoder's quality*, not as the primary research question. The supervisor wants that relationship inverted: the research question drives everything, and the encoder is the means to answer it.

---

## Paragraph 2 — The Central Research Question

> *"The key question can be: does Item 1A text add incremental predictive value for future realised volatility beyond lagged volatility and standard financial controls?"*

### Is it correct?
**Yes — and this is a standard, rigorous bar in empirical finance.** Volatility is strongly autocorrelated (volatility clustering), meaning yesterday's volatility is already an excellent predictor of tomorrow's. In finance, the test for whether *any* new variable — text included — is informative is always: does it improve prediction **after** controlling for what we already knew? If the model cannot beat AR(1) on lagged volatility, a finance audience will dismiss the entire exercise regardless of how sophisticated the encoder is.

### How it differs from the current report
The current ablation (Phase 5, 7 conditions) **never includes a non-text baseline**. All 7 conditions are text-based representations:

1. SBERT
2. Fin-E5
3. DAPT-only
4. DAPT + Contrastive
5. LoRA-adapted
6. BERTopic-only
7. Hybrid

So the current success criterion — *"Hybrid Spearman ρ > any single-representation condition"* (Table 2) — only proves the encoder is better than other text encoders. It does **not** prove that text adds anything over a simple lagged-volatility model. This is the most concrete gap in the current design.

---

## Paragraph 3 — Three-Tier Baseline Structure

> *"Three groups of baselines: (1) non-text: AR(1)/lagged realised vol, historical vol; (2) simple text: TF-IDF and Loughran-McDonald dictionary; (3) representation: SBERT, E5/BGE, FinBERT, BERTopic, Chiu dual-view, your three-view."*

### Is it correct?
**Yes — this is how a rigorous NLP-meets-finance paper structures its evaluation.** Each tier answers a different question:

| Tier | Baseline | Question Answered |
|------|----------|-------------------|
| 1 — Non-text | AR(1), lagged vol | Does any text help at all? |
| 2 — Simple text | TF-IDF, LM dictionary | Do neural representations help over bag-of-words? |
| 3 — Representations | SBERT → your encoder | Does domain adaptation and your architecture help over generic encoders? |

All three tiers are needed or the bottom of the argument is missing. Tier 1 is the most critical; without it there is no claim that text is informative. Tier 2 is important because Loughran & McDonald (2011) is already reference [1] in the report — it would be inconsistent to cite it as foundational work and then not include it as a baseline.

### How it differs from the current report

| Tier | Status in Current Report |
|------|--------------------------|
| Tier 1 (non-text) | **Completely absent.** No AR(1), no GARCH, no lagged-vol-only baseline anywhere |
| Tier 2 (simple text) | **Absent as a baseline.** LM [1] is cited in the lit survey but not included in the ablation. TF-IDF not mentioned |
| Tier 3 (representations) | **Mostly covered:** SBERT ✓, Fin-E5 ✓, BERTopic-only ✓, three-view ✓. Missing: FinBERT, E5/BGE, explicit Chiu dual-view condition |

> **Note:** Including FinBERT is reasonable — it is the most widely used financial BERT model and serves as a natural lower-bound for domain-specific encoders.

---

## Paragraph 4 — Separating the Representation Module from the Prediction Head + Feature Table

> *"Separate the risk representation module from the volatility prediction head. The reusable output should be a firm-filing feature table: CIK, PERMNO, filing date, SIC/industry, dense risk embeddings, topic exposure vectors, lagged volatility, and future realised volatility labels."*

### Is it correct?
**Yes — both technically and strategically.** Technically, modular separation is already implicit in the current design (encoder trained in Phase 3, regression head attached in Phase 5). But the supervisor is making a stronger point: make the **feature table itself a named, standalone deliverable**. This matters for three reasons:

1. It is a concrete, reusable research artefact — other researchers can use it without re-running the pipeline
2. It forces the pipeline to produce clean, reproducible outputs at every stage rather than just a final metric
3. Including lagged volatility **in the same table** as the embeddings makes the incremental-value test trivially reproducible: anyone can run AR(1) on the lagged-vol column and compare to regression on the embeddings column

### How it differs from the current report
Current deliverables (Table 5 in the report):

| Deliverable | Description |
|-------------|-------------|
| D₃ | BERTopic risk topic analysis + *per-filing topic-feature matrix* |
| D₄ | Full ablation + downstream volatility forecasting results |

D₃ is the closest match — it mentions a "per-filing topic-feature matrix." But:
- It does not include dense embeddings alongside topic vectors
- It does not include lagged volatility or future realised volatility labels
- It is described as a BERTopic output, not as a unified ML-ready dataset

The supervisor's proposed feature table would merge what is currently split across D₁–D₄ into one structured artefact. This does not require additional work — it is a matter of saving intermediate outputs into a joined table during Phase 5.

---

## Paragraph 5 — Summary: "Manageable and Stronger"

> *"This keeps the project manageable while making the contribution stronger: a reusable Item 1A risk representation module validated by a strict forward-looking volatility benchmark."*

### Is it correct?
**Yes — and it is a reassurance, not a criticism.** The supervisor is explicitly saying the changes are not scope-expanding. They are scope-focusing: the same work produces two cleaner outputs — a reusable module and a rigorous benchmark — instead of a loosely defined "financial NLP system."

### How it differs from the current report
The current Section 3.2 (Expected Outcomes) lists five deliverables but frames them as encoder milestones:
- *"domain-adapted encoder outperforms SBERT on FinMTEB"* → encoder-quality claim
- *"contrastive encoder matches/exceeds Fin-E5"* → encoder-quality claim
- *"BERTopic Cᵥ > SBERT baseline"* → topic-quality claim

None of the five outcomes is framed as "does text add incremental predictive value?" The supervisor's phrasing reorders these: the forecasting benchmark is the primary claim, and the encoder quality is what supports it.

---

## Overall Summary

| Supervisor Point | Valid? | Gap in Current Report |
|------------------|--------|------------------------|
| Reframe as volatility benchmark, not NLP system | ✓ Strong | Framing is encoder-first; volatility is downstream validation |
| Research Q: does text add beyond lagged vol? | ✓ Strong | No non-text baseline exists anywhere |
| Three-tier baselines | ✓ Strong | Tiers 1 & 2 are entirely absent |
| Separate module from head; output feature table | ✓ Constructive | Feature table not a named deliverable; lagged vol not in output |
| Project stays manageable | ✓ Reassurance | No change in scope needed |

---

---

## Concepts Q&A

**Q: What is volatility?**
How wildly a stock price moves — measured as the standard deviation of daily returns over a window (e.g. 30 days). High volatility = large unpredictable swings. Low volatility = calm and stable. Used by investors and risk managers to price options and manage exposure.

**Q: What is lagged volatility?**
The volatility already observed *before* a filing date — historical fact, not a prediction. A key empirical finding in finance is that volatility clusters: if a stock was bouncy last month, it tends to stay bouncy next month. This makes lagged volatility a surprisingly strong predictor of future volatility on its own.

**Q: What is AR(1)?**
AutoRegressive model with one lag. In plain English: "next volatility ≈ constant × last volatility + noise." One formula, no text, no NLP. Despite its simplicity it is hard to beat. If a complex NLP model cannot outperform AR(1), the premise that text adds information collapses entirely — which is why it must be included as a baseline.

**Q: How does a downstream task help?**
A downstream task (here: 30-day forward volatility prediction) is the real-world stress test that proves embeddings carry economically meaningful signal, not just linguistic structure. Without it you only know the encoder scores well on intrinsic benchmarks like FinMTEB — but intrinsic quality ≠ practically useful. The downstream task is also why the supervisor wants it reframed as the headline: "does Item 1A text predict future volatility beyond lagged vol?" is a crisp, falsifiable claim that a finance audience can evaluate.

---

## Feature Table — Column-by-Column Reference

> The supervisor proposed a firm-filing feature table as a standalone deliverable. Each row = one company's one 10-K filing. Below is what each column means, how it looks, and where it comes from.

---

### 1. CIK — Central Index Key
**What it means:** The unique ID the SEC assigns to every company that files with them. Like a passport number for a firm on EDGAR.

**How it looks:**
```
CIK
0000320193   ← Apple Inc.
0000789019   ← Microsoft Corp.
0001318605   ← Tesla Inc.
```
**Source:** EDGAR directly. Every filing URL contains it. Collected automatically by the Phase 1 scraper when downloading Item 1A text.

---

### 2. PERMNO — Permanent Number
**What it means:** The unique ID that CRSP assigns to every *stock* (traded security). While CIK identifies the legal company, PERMNO identifies the share. One company can have multiple share classes → multiple PERMNOs.

**How it looks:**
```
PERMNO
14593    ← Apple
10107    ← Microsoft
93436    ← Tesla
```
**Source:** CRSP/Compustat Merged (CCM) linkage table on WRDS — the bridge between the SEC world (CIK) and the finance world (PERMNO). Without it, stock prices cannot be attached to filings.

---

### 3. Filing Date
**What it means:** The date the company submitted the 10-K to EDGAR. Everything after this date is "the future" for that filing — critical for the temporal split.

**How it looks:**
```
filing_date
2021-10-29
2022-01-26
2023-02-02
```
**Source:** EDGAR submissions API (`data.sec.gov/submissions/CIK{cik}.json`) — in the metadata of every filing alongside the document text.

---

### 4. SIC / Industry
**What it means:** Standard Industrial Classification — a government code categorising what industry a firm operates in. The 2-digit version (broader grouping) is used for the sector view in contrastive fine-tuning (same SIC = soft positives; cross-sector = hard negatives).

**How it looks:**
```
SIC2   industry_label
73     Business Services
36     Electronic Equipment
28     Chemicals & Pharma
```
**Source:** EDGAR submissions API — same JSON that gives the filing date also gives the SIC code. No WRDS dependency for this column.

---

### 5. Dense Risk Embeddings
**What it means:** The numerical representation of the Item 1A text produced by the trained encoder. The entire risk disclosure is compressed into a fixed-length vector that captures its semantic meaning. Two filings with similar risk language will have vectors close together in this space.

**How it looks:**
```
embedding
[-0.0312,  0.1847, -0.2103,  0.0091, ...,  0.1542]   ← 768 numbers for one filing
[-0.1204,  0.0563,  0.3021, -0.1187, ..., -0.0872]   ← next filing
```
Each row is a list of 768 floating-point numbers (all-mpnet-base-v2 hidden size).

**Source:** Produced by the pipeline — Phase 3 (contrastive fine-tuned encoder). Item 1A text is passed through the encoder and mean-pooled. This column does not exist externally; the model creates it.

---

### 6. Topic Exposure Vectors
**What it means:** For each filing, BERTopic assigns a probability distribution over discovered topics (e.g. "regulatory litigation risk", "supply-chain disruption", "interest rate exposure"). Each filing gets a vector showing how much of each topic it contains.

**How it looks:**
```
topic_vector
[0.42, 0.03, 0.31, 0.00, 0.18, ...]   ← filing heavily about topics 0, 2, 4
[0.01, 0.67, 0.05, 0.22, 0.00, ...]   ← filing heavily about topic 1
```
Each number is between 0 and 1. Length = number of topics BERTopic discovers (typically 20–80 for a corpus this size).

**Source:** Produced by the pipeline — Phase 4 (BERTopic extraction). This is the interpretable half of the hybrid representation.

---

### 7. Lagged Volatility
**What it means:** The realised volatility of the firm's stock in the 30 days *before* the filing date — computed as the standard deviation of daily log-returns over that window. This is already known at filing time; it is not a prediction. It enables the AR(1) baseline: predict future vol ≈ α × this number.

**How it looks:**
```
lagged_vol_30d
0.0243    ← Apple (calm)
0.0581    ← Tesla (more volatile)
0.0187    ← Microsoft (calmer)
```
**Source:** CRSP daily stock returns via WRDS. Computed from daily returns for the 30 trading days before `filing_date` using the PERMNO.

---

### 8. Future Realised Volatility (Label)
**What it means:** The realised volatility of the firm's stock in the 30 days *after* the filing date. This is the target variable — ground truth the regression head is trained to forecast. For the test set (2025–2026), collected from yfinance as a CRSP fallback.

**How it looks:**
```
fwd_vol_30d
0.0271    ← Apple's actual volatility in 30 days after filing
0.0634    ← Tesla's
0.0201    ← Microsoft's
```
**Source:** CRSP daily stock returns via WRDS (yfinance fallback for 2025–2026 test filings).

---

### Summary Table

| Column | Type | Size per row | Source |
|--------|------|--------------|--------|
| CIK | ID string | 10 chars | EDGAR (scraped) |
| PERMNO | ID integer | 5 digits | WRDS CCM linkage |
| filing_date | Date | YYYY-MM-DD | EDGAR API |
| SIC2 | Integer | 2 digits | EDGAR API |
| embedding | Float vector | 768 numbers | Your encoder (Phase 3) |
| topic_vector | Float vector | ~20–80 numbers | BERTopic (Phase 4) |
| lagged_vol_30d | Float | 1 number | CRSP via WRDS |
| fwd_vol_30d | Float | 1 number | CRSP / yfinance |

> **Key insight:** The first four columns are identifiers (who + when). The next two are produced by the NLP pipeline. The last two are financial labels from market data. Together, one row is a complete, self-contained record that any researcher can use to reproduce experiments or run new ones without re-running the full pipeline.

---

## Action Items (Derived from Feedback)

The methodology pipeline itself is **not being criticised** — DAPT → Contrastive FT → BERTopic → Ablation is accepted. The required changes are:

1. **Reframe Introduction and Abstract** — lead with the research question ("does Item 1A text add incremental value for volatility prediction?"), not with the encoder construction
2. **Add Tier 1 baselines to Phase 5** — AR(1) on lagged realised volatility, historical volatility (e.g. 30-day rolling std)
3. **Add Tier 2 baselines to Phase 5** — TF-IDF + Ridge, Loughran-McDonald sentiment scores + Ridge
4. **Add FinBERT and Chiu dual-view to Tier 3** — explicit dual-view condition to demonstrate the marginal value of the sector (third) view
5. **Define the feature table as a deliverable** — structured CSV/parquet: `CIK | PERMNO | filing_date | SIC | embedding_vector | topic_vector | lagged_vol | fwd_vol_label`
6. **Update success criteria in §3.3** — primary criterion should be "hybrid text model Spearman ρ > lagged-vol-only AR(1) baseline on 2025–2026 test set"
