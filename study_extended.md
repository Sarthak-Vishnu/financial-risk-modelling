# Predicting Forward Volatility from 10-K Risk Disclosures: The Extended Study

**Author:** Sarthak Vishnu (s2880814) · **Supervisor:** Prof Tiejun Ma · University of Edinburgh
**Date:** 2026-07-06
**Status:** Report-draft narrative. This document supersedes `study.md` as the working account of the
study. All citations are grounded in `Literature_agent_study_extended.md`, `Literature_agent.md`,
and `Literature_agent_phase3.md`. All numbers trace to `phase5/STRESS_TEST_RESULTS.md` and the
Phase 1 to 5 result files on this branch.

**How this document is organised.** The reader is assumed to know the original pipeline from the
IPP report and nothing after it. Part 0 therefore restates the original pipeline in full, so this
document stands alone. Part I describes three corrections that were made to the original pipeline
after an internal audit, and explains why each one is disclosed rather than silently applied.
Part II describes the reframing of the research question around a structured financial baseline.
Part III presents the primary results under the corrected, fair protocol. Part IV describes the
robustness extension, which is a deliberate methodological adaptation beyond the original design,
and the final verdict it produced. Parts V and VI cover the disclosure-change branch and the
ongoing earnings-call work. Part VII states the conclusions and contributions.

---

## Part 0 — The original pipeline

### 0.1 Research framing and target

The task is to predict a firm's 30-day forward realised volatility from the text of its SEC 10-K
Item 1A "Risk Factors" disclosure. Volatility is defined as the standard deviation of daily log
returns over the 30 trading days strictly after the filing date, annualised by multiplying by the
square root of 252. The corresponding lagged measure uses the 30 trading days strictly before the
filing date. The modelling target is the natural logarithm of forward volatility, because
volatility is approximately log-normal and strictly positive, and using the log once and
everywhere keeps error metrics comparable across all models.

Two metrics are reported throughout. The first is the within-year cross-sectional Spearman rank
correlation between predicted and realised volatility, referred to as the information coefficient
(IC). It answers the question "did the model rank firms correctly from least to most volatile
within a year?". The second is R² on the log target (R²_log), which measures how much of the
variance in the volatility level the model explains. Ranking is the primary lens because the
economic use of such a model is cross-sectional (which firms are riskier than which), while the
level is dominated by market-wide conditions.

For context on magnitudes, Grinold and Kahn (2000) establish that an IC of 0.05 to 0.10 is
considered good for cross-sectional return prediction. Volatility is a far more forecastable
quantity than returns: Corsi (2009) shows that the HAR-RV model reaches R² of 40 to 70 percent for
realised-volatility forecasting using only past volatility, whereas return-prediction R² is
typically below 2 percent. ICs in the range 0.5 to 0.6 for volatility ranking are therefore
plausible and should be judged against a persistence baseline, not against zero.

### 0.2 Dataset

The universe is the S&P 500 constituents, including delisted firms to avoid survivorship bias. The
text corpus is one pickle file per 10-K filing containing the Item 1A section, around 8,247
filings spanning roughly 2006 to 2026. The master feature table has one row per filing (around
8,105 rows), keyed by ticker and fiscal year, and is built by a chain of scripts that index EDGAR
filing dates, link CIK identifiers to CRSP/Compustat identifiers, and compute the volatility
labels. Daily prices come from three sources in priority order: the FINSABER S&P 500 price file
(2000 to 2024, including delisted names), WRDS CRSP daily returns for 2025, and a yfinance
fallback for two tickers with no CRSP link.

The temporal split, used identically by every phase, is: training on filings dated before
2025-01-01 (around 7,353 filings), validation on 2025 filings (406 filings, of which around 397
have complete labels), and a held-out 2026 test year whose forward windows had not closed at the
time of the experiments. All model selection and fitting uses the training years only.

### 0.3 Phase 2 — Domain-adaptive pretraining (DAPT)

Following Gururangan et al. (2020), the encoder was first adapted to financial-risk language by
continued masked-language-model training on the 10-K corpus. The base model is
`all-mpnet-base-v2` rather than the raw `microsoft/mpnet-base`, because the sentence-transformer
variant already has useful sentence-level geometry and only the encoder (not the MLM head) is
carried forward, so the right trade-off is to reshape its token vocabulary toward financial risk
while preserving that geometry.

The main design decision in this phase was how to chunk long Item 1A prose (median around 7,000
words) into 512-token MLM sequences. Two strategies were built, both within-document,
non-overlapping, greedy packing in the RoBERTa DOC-SENTENCES family (Liu et al. 2019): a
sentence-aware packer and a paragraph-aware packer that never splits an individual risk factor.
The literature survey supported coherent within-document packing without overlap (RoBERTa's
ablation, corroborated by Geiping and Goldstein 2022 and the domain precedents Legal-BERT and
SEC-BERT). An A/B comparison on an identical validation set selected the paragraph-aware variant
(validation perplexity 2.2278 against 2.2671), with the recorded caveat that the margin is small
and partly confounded by the paragraph variant taking about 22 percent more gradient steps at a
fixed five epochs.

Training used the standard DAPT recipe: 15 percent masking, learning rate 2e-5, effective batch 32,
five epochs with early stopping on validation perplexity, fp16, 512-token sequences. The selected
checkpoint reduces validation perplexity by about 16.5 percent relative to `microsoft/mpnet-base`
(2.23 against 2.67).

### 0.4 Phase 3 — Contrastive fine-tuning

The post-DAPT encoder was then fine-tuned with a SimCSE-style InfoNCE objective with in-batch
negatives, extending the dual-view framework of Chiu et al. (2025), the closest published
precedent, which pairs 256-token risk paragraphs. Three positive-pair views were constructed:
a lexical view (two overlapping spans of the same paragraph), a chronological view (the same
firm's matched risk factor in adjacent years, requiring TF-IDF cosine at least 0.5), and a new
sector view (paragraphs from different firms in the same two-digit SIC industry and fiscal year).
Numbers and dates are normalised to `[NUM]` and `[DATE]` tokens following SEC-BERT (Loukas et al.
2022). The loss is InfoNCE at temperature 0.05 (Gao et al. 2021) with label-aware false-negative
masking on the sector label (Khosla et al. 2020), and the soft sector view is down-weighted by a
factor of 0.5. Pairs are built from the training split only.

Three encoders were trained: `dual` (lexical plus chronological, a replication of Chiu et al.),
`three` (adding the sector view, full fine-tuning), and `three_lora` (the sector variant trained
with LoRA adapters, r=16, following Hu et al. 2021). Intrinsic evaluation on FinMTEB showed the
expected pattern: DAPT improves in-domain semantic similarity but damages retrieval geometry
(the known anisotropy of raw MLM embeddings), and contrastive fine-tuning repairs it. A diagnosis
of the sector view found its random within-industry pairing produced many near-unrelated pairs,
which blurred the geometry under full fine-tuning but was contained under LoRA.

### 0.5 Phase 4 — BERTopic topic modelling

To provide the explanatory axis (which risk themes a filing loads on), BERTopic was fitted on the
training paragraphs of each encoder's embedding space, with UMAP (15 neighbours, 5 components,
cosine metric) followed by HDBSCAN (minimum cluster size 200) and a class-based TF-IDF topic
representation, reduced to 49 topics. The topic model is fitted on training documents only and
then applied to validation and test documents, so no held-out text shapes the topic space. Each
filing receives a 49-dimensional topic-exposure vector (the fraction of its paragraphs assigned to
each topic). Topic coherence (C_v) sits between 0.67 and 0.72 across encoders. The DAPT-only
backend was excluded because its anisotropic embeddings prevent UMAP from converging.

### 0.6 Phase 5 — The original evaluation and its conclusion

The original evaluation compared tiers of models on the 2025 validation year, all sharing the same
filings, split, log target and metric code. The baseline tiers were naive persistence (predict the
lagged volatility), an AR(1) ridge regression on lagged volatility, TF-IDF text alone, and TF-IDF
plus lagged volatility. The TF-IDF vectoriser is not a naive one: English stop words removed,
unigrams and bigrams, minimum document frequency 5, 20,000 features, sublinear term frequency,
with the ridge penalty selected by cross-validation. This follows the standard strong lexical
baseline in the 10-K literature: Kogan et al. (2009) established TF-IDF bag-of-words features with
a linear model as the reference method for predicting volatility from 10-K text, and Loughran and
McDonald (2011) established that finance-calibrated lexical features are strong predictors in this
domain. The learned tiers scored each contrastive encoder as a mean-pooled 768-dimensional filing
vector, the topic-exposure vector, and a hybrid of encoder, topics and lagged volatility, each
under a ridge head and a small MLP.

The headline result of the original pipeline was that the count-based representation won. TF-IDF
alone reached Spearman 0.60 on the 2025 validation year, the TF-IDF plus lagged floor reached
R²_log 0.176 with Spearman 0.56, and the best learned condition (the `dual` hybrid) reached only
0.545 with R²_log 0.13. Every one of five encoder conditions came in at or below the count-based
floor. The recorded interpretation was that volatility prediction from risk-factor text is a
lexical task: the frequency of specific risk language carries the signal, and a bag-of-words
representation captures it directly, while semantic embeddings blur it.

### 0.7 The two structural limitations

Two features of this design limited what it could conclude. First, the only non-text predictor in
the entire pipeline was lagged volatility, one number, even though the asset-pricing literature
establishes a set of structured firm characteristics as first-order determinants of volatility.
Any comparison between text representations was therefore a comparison between models that all
lacked the dominant predictors. Second, the question itself ("which text representation ranks
best?") is weaker than the question the literature actually cares about, which is whether
disclosure text adds predictive power over the strong quantitative baseline. These two limitations
motivated everything that follows.

---

## Part I — Three corrections to the original pipeline

Before extending the pipeline, an internal audit re-examined its foundational choices. The audit
found two data defects and one evaluation defect. All three are disclosed here rather than
silently fixed, for three reasons. The numbers in the IPP report were computed on the flawed
panel, so any discrepancy between that report and this document must be explainable. Each fix is
load-bearing for a claim made later in this document. And each fix produced a finding of its own.
The corrected panel is the basis for every number from Part III onward, and the pre-fix numbers
are quoted only for comparison.

### I.1 The label-join defect

**The defect.** The corpus filenames carry the year of the filing, but the feature table is keyed
by fiscal year, and for most firms these differ. In addition, 68 ticker and fiscal-year groups
contained two different filings (an artifact of 52/53-week fiscal calendars, where two fiscal
year-ends fall in the same labelled year), and every merge in the pipeline keyed on ticker and
fiscal year. The consequence was that 136 labelled rows attached text to a wrong-year label, 122
filings were silently dropped because of ticker renames, and, once the filename-year convention
was traced through, about 74 percent of all text-to-label pairs turned out to be paired with the
adjacent year's label rather than their own.

**The fix.** A repair script (`dataset_config/fix_filing_join.py`) re-keys every filing by its
true filing year, enforces unique firm-year keys, and produces a corrected feature table and
corpus (`datasets/feature_table_fixed.parquet`, `topics/data/topic_docs_fixed.jsonl`). The
mismatch was conservative rather than leaky: the mispaired text was older than its label, not
newer, so no future information contaminated any result. The 122 orphaned filings are recoverable
in principle through a ticker-rename map but require label recomputation, and remain excluded.

**What the fix revealed.** Re-running the headline comparison on the corrected panel barely moved
it (structured plus TF-IDF went from 0.611 to 0.610 on the validation year). Correcting 74 percent
of the pairs and changing the answer by one thousandth is itself informative: it means year-stale
risk text predicts volatility almost exactly as well as current risk text, which is direct
evidence that Item 1A language is highly persistent from year to year. This connects to the
disclosure-change literature. Cohen, Malloy and Nguyen (2020) built their "Lazy Prices" result on
precisely the observation that most filings change very little year over year. Campbell et al.
(2014) showed that risk-factor disclosures are informative rather than empty boilerplate. Our
finding is consistent with both: the text carries real signal (Part III shows it survives every
control), and that signal sits in the persistent level of the language rather than in its
year-over-year freshness.

### I.2 The truncation defect

**The defect.** The encoders processed each paragraph truncated to 256 tokens, following the
convention of the parent paper (Chiu et al. 2025). In this corpus, 36.6 percent of paragraphs are
longer than 256 tokens, and in total about 65 percent of the corpus words never reached any
encoder. The TF-IDF representation, by contrast, ingests the full text. Every encoder-versus-
TF-IDF conclusion in the original pipeline therefore compared a full-text count model against
encoders that saw roughly one third of the words.

**The fix.** A windowed encoding scheme (`topics/encode_paragraphs.py`) splits each long paragraph
into overlapping 256-token windows with a stride of 192 tokens, encodes every window, and
mean-pools the window vectors back to a paragraph vector before the usual paragraph-to-filing
pooling. This is the standard approach for documents that exceed a transformer's context window:
segment, encode, and pool. Splitting a long document into fixed-length, possibly overlapping
chunks, encoding each chunk independently, and combining the chunk representations is documented
as established prior practice in the long-document literature (Beltagy et al. 2020, Section 2).
The financial-NLP precedent is the hierarchical encoding of long transcripts in Yang et al.
(2020, HTML), which encodes each fixed-length segment with BERT and aggregates the segment
representations at document level; flat windowing with mean-pooling is the simplified variant of
the same principle, and the 256-token paragraph unit follows Chiu et al. (2025). All five encoder
caches were rebuilt this way: 462,590 windows over 176,616 paragraphs.

**What the fix revealed.** Full-text encoding did not rescue the encoders (the full grid is in
Part III). The value of the fix is that the comparison is now fair, so the conclusion "the count
model wins" is defended rather than an artifact of unequal inputs.

### I.3 The head-standardisation defect

**The defect.** The redesign's ablation ladder (Part II) compared a structured-only baseline
scored with a gradient-boosted-tree head (HGB) against a structured-plus-TF-IDF model scored with
a sparse ridge head. Two model classes differ in capacity and regularisation, so the measured
increment of "adding text" was confounded with the effect of switching model class. This is
exactly the comparison hygiene problem described by Lipton and Steinhardt (2019): when two things
change at once, the credit assignment is arbitrary.

**The fix and what it revealed.** Scoring every rung of the ladder under the same head shows the
confound was large. Structured features under ridge reach IC 0.591 on the validation year and
0.603 on the twelve-year backtest, against 0.558 and 0.584 under HGB. The apparent text increment
of about +0.05 IC in the pre-audit redesign shrinks to +0.011 IC (paired p = 0.057 over twelve
years) once both arms share the ridge head. The text increment is therefore real, consistent in
sign across years, but roughly five times smaller than first reported. On the level-accuracy
metric the text gain remains substantial (R²_log 0.175 to 0.226 on the validation year, and the
difference in squared error is significant under a Diebold-Mariano test at p < 0.001). A related
audit finding is that tuning the ridge penalty on the last training year hurt the following year,
which confirms that model-selection variance swamps deltas of this size and that only paired
multi-year tests should be trusted for them. All headline comparisons from here on hold the head
fixed.

---

## Part II — The reframing: incremental value over a structured baseline

### II.1 Why the question changed

The original pipeline asked which text representation ranks firms best. The literature's actual
question is different: given that a set of structured firm characteristics is known to drive
volatility, does disclosure text add anything on top? Answering the second question requires
building the structured baseline that the original pipeline never had.

### II.2 The structured feature block

The structured block contains the standard cross-sectional volatility predictors, each grounded in
the asset-pricing literature. Realised volatility at horizons of 21, 63, 126 and 252 trading days
follows the multi-horizon persistence structure of the HAR-RV model (Corsi 2009). Market beta and
idiosyncratic volatility follow Ang, Hodrick, Xing and Zhang (2006), the canonical treatment of
volatility in the cross-section, which also establishes firm size, leverage and book-to-market as
the standard covariates of this analysis (book-to-market and size originating with Fama and French
1992). Illiquidity uses the Amihud (2002) ratio, the definitional measure. Return skewness and
kurtosis follow the higher-moment pricing evidence of Conrad, Dittmar and Ghysels (2013), with
kurtosis included as an exploratory control (no single canonical paper establishes it as a
standalone volatility predictor). Momentum is included as a standard control in the same spirit as
the control sets of Ang et al. (2006). Vol-of-vol, drawdown and dollar volume complete the block.
Fundamentals (log market capitalisation, leverage, book-to-market, return on assets) come from
Compustat, prices from the sources in Part 0.2. Coverage is approximately 99 percent across 7,666
filings. Building this block involved no new modelling, only feature engineering, which is
precisely the point: it is the baseline any volatility study should start from.

### II.3 The ablation ladder

The evaluation is an ablation ladder over feature blocks with the prediction head held fixed
(Part I.3): structured features alone, then structured plus a text representation, where the text
representation is TF-IDF, an encoder embedding, topic exposures, change features, or combinations
of these. Each condition is scored on the validation year and, in Part IV, on expanding-window
backtests. Statistical comparisons are paired: within each evaluation year the two models predict
the same firms, so their per-year ICs are compared as matched pairs across years, and their
per-firm squared errors are compared by Diebold-Mariano tests.

---

## Part III — Tier 1: primary results under the corrected, fair protocol

These are the primary results of the study. The protocol is the original single-split design
(train before 2025, evaluate on 2025), which is the design most comparable to the literature, run
on the corrected panel (Part I.1), with full-text encoder inputs (Part I.2) and a fixed head per
comparison (Part I.3).

### III.1 How to read the tables

Every results table in this document uses the following columns. **IC** is the within-year
cross-sectional Spearman rank correlation between predicted and realised 30-day forward
volatility, the primary metric (Part 0.1 explains why, and why values near 0.5 to 0.6 are the
relevant range for volatility rather than the 0.05 to 0.10 range familiar from return prediction).
**95% CI** is the bootstrap confidence interval on that single-year IC; with n = 393 firms the
interval spans roughly plus or minus 0.06, which is why single-year differences of 0.01 to 0.02
can never be conclusive on their own. **R²_log** is the coefficient of determination on the log
volatility level. **DM p** is the p-value of a Diebold-Mariano (1995) test comparing the squared
prediction errors of the row's model against the reference model on the same firms; it tests
accuracy on the level, complementing the rank-based IC. Model names state the feature blocks and
the head in brackets, so "struct+tfidf [sparse]" is structured features plus the TF-IDF block
under the sparse ridge head, and "structured [ridge]" is the structured block alone under a ridge
head.

### III.2 The anchor results

| condition | IC | 95% CI | R²_log | DM p vs struct+tfidf |
|---|---|---|---|---|
| lagged [hgb] | 0.457 | — | −0.130 | — |
| structured [hgb] | 0.558 | [0.487, 0.628] | 0.094 | 0.001 |
| structured [ridge] | 0.591 | [0.522, 0.659] | 0.175 | 0.000 |
| tfidf+lag [sparse] | 0.541 | — | 0.116 | — |
| **struct+tfidf [sparse]** | **0.603** | [0.534, 0.670] | **0.226** | reference |

Three statements summarise this table. First, the structured baseline alone beats the entire
original text-only pipeline: 0.591 against the old TF-IDF-plus-lagged floor of 0.541. The standard
quantitative characteristics rank firm volatility better than any text-only method, which
retroactively explains why the original pipeline's comparisons were all fought below this level.
Second, text on top of the structured baseline adds: struct+tfidf reaches 0.603 IC and raises
R²_log from 0.175 to 0.226, and the Diebold-Mariano test says this accuracy gain is highly
significant. Third, the IC increment (+0.012 on this single year) is inside the single-year
confidence interval, which is exactly why Part IV moves to multi-year paired tests before claiming
anything about it.

### III.3 The encoder grid

The full grid scores every encoder under both heads on the corrected panel with full-text windowed
inputs. The encoders are: `dual` and `sbert` (the original similarity-trained contrastive encoder
and the off-the-shelf sentence transformer), `bge` (BAAI bge-base-en-v1.5, a modern strong
general-purpose embedder, included to address the objection that the original encoders were
simply not good enough), `volaware` (the Stage B contrastive encoder whose positive pairs are
paragraphs from firms in the same within-year forward-volatility decile), and `ftvol` (an encoder
fine-tuned end-to-end on the volatility regression target).

| condition | IC | 95% CI | R²_log | DM p vs struct+tfidf |
|---|---|---|---|---|
| struct+enc[dual] [ridge] | 0.582 | [0.509, 0.652] | 0.162 | 0.013 |
| struct+enc[dual] [hgb] | 0.582 | [0.511, 0.651] | 0.160 | 0.051 |
| struct+enc[sbert] [ridge] | 0.562 | [0.482, 0.636] | 0.017 | 0.000 |
| struct+enc[sbert] [hgb] | 0.591 | [0.521, 0.658] | 0.149 | 0.043 |
| struct+enc[bge] [ridge] | 0.580 | [0.506, 0.649] | 0.093 | 0.000 |
| struct+enc[bge] [hgb] | 0.584 | [0.512, 0.651] | 0.145 | 0.023 |
| struct+enc[volaware] [ridge] | 0.587 | [0.514, 0.656] | 0.147 | 0.003 |
| struct+enc[volaware] [hgb] | 0.597 | [0.530, 0.663] | 0.207 | 0.593 |
| struct+enc[ftvol] [ridge] | 0.606 | [0.540, 0.672] | 0.185 | 0.207 |
| struct+enc[ftvol] [hgb] | 0.598 | [0.529, 0.666] | 0.167 | 0.155 |
| struct+enc[ftvol, topk_risk] [hgb] | 0.607 | [0.538, 0.677] | 0.194 | 0.454 |
| struct+tfidf_svd [ridge] | 0.594 | [0.526, 0.661] | 0.167 | 0.000 |
| EVERYTHING svd+enc[ftvol]+chg [hgb] | 0.608 | [0.542, 0.675] | 0.170 | 0.204 |
| EVERYTHING tfidf+enc[ftvol]+chg [sparse] | 0.590 | [0.525, 0.662] | 0.165 | 0.135 |

The pattern is clean. The generic semantic encoders (dual, sbert, bge) sit at or below the no-text
structured baseline and are significantly less accurate than struct+tfidf under a Diebold-Mariano
test in at least one head. The inclusion of bge settles the "you never tried a strong modern
embedder" objection: it loses too, and giving the encoders the full text (Part I.2) did not change
this. The direction of this result has independent support: the FinMTEB benchmark reports that
bag-of-words representations outperform dense embedding models on financial semantic-similarity
tasks (Su et al. 2025), so count-based representations beating dense ones on financial text is a
documented phenomenon, not an idiosyncrasy of this pipeline. The only encoders that reach statistical parity with the count model are the two that were
trained on the volatility task itself, ftvol and volaware. Task alignment, not model modernity, is
what closes the gap. None of the parity rows beats the reference, and the kitchen-sink fusion of
every representation at once also fails to beat it. This grid also supersedes the original
pipeline's encoder comparison: task-supervised fine-tuning, which the original post-mortem
speculated would have worked better, was built (following the precedent of end-to-end
volatility-supervised text encoders in Qin and Yang 2019 and Yang et al. 2020), and it reaches
parity but not superiority.

One caveat belongs to this table and it matters for everything in Part IV. The parity rows are
in-period results for encoders whose training saw pre-2025 volatility labels, and the ftvol
checkpoint was additionally epoch-selected on the validation year itself. These rows are honest as
exploratory evidence and inflated as claims. Establishing what a task-aligned encoder can really
do requires the clean protocol of Part IV.3.

---

## Part IV — Tier 2: the robustness extension

Everything in this part goes beyond the original evaluation design. It is presented as a
deliberate methodological adaptation, adopted for a specific reason: the single-year comparison in
Part III cannot statistically separate models 0.01 apart, and, more seriously, it cannot detect a
class of leakage that turned out to be present. Both problems require evaluation across many
years.

### IV.1 The expanding-window backtest

The protocol is the standard walk-forward design for financial prediction (the reference treatment
of why k-fold cross-validation is inappropriate for financial time series, and of walk-forward
evaluation and lookahead bias generally, is Lopez de Prado 2018, chapters 4 and 7). For each test
year Y from the start year to 2024, every model is trained on filings from years strictly before
Y and scored on year Y, exactly as a deployed model would have been. Two windows are reported: a
twelve-year window (2013 to 2024) and a seven-year window (2018 to 2024). Comparisons between
models are paired by year (the per-year IC differences are tested against zero) and by firm
(Diebold-Mariano on squared errors). The t-statistic in the tables below is the mean yearly IC
divided by its standard error across years, so it measures the consistency of the ranking skill,
and the paired p-values test whether one model is reliably better than another on identical data.

| condition | 2018–2024 IC (t) | 2013–2024 IC (t) |
|---|---|---|
| lagged [hgb] | 0.456 (4.6) | 0.526 (8.3) |
| structured [hgb] | 0.511 (5.0) | 0.584 (9.0) |
| structured [ridge] | 0.546 (5.6) | 0.603 (9.9) |
| tfidf+lag [sparse] | 0.499 (5.2) | 0.558 (9.3) |
| **struct+tfidf [sparse]** | **0.561 (5.9)** | **0.614 (10.4)** |

Paired across years against the fair structured [ridge] baseline, struct+tfidf adds +0.016 IC
(p = 0.098) on the seven-year window and +0.011 IC (p = 0.057) on the twelve-year window. The text
increment is small, positive in sign in both windows and in most individual years, and just short
of the conventional significance threshold on twelve years of data. The structured baseline's own
superiority over pure persistence is unambiguous (+0.059 IC over lagged, p = 0.007). This is the
honest shape of the headline result: a strong structured baseline, plus a small, consistent,
statistically suggestive lexical text increment, with a much clearer text advantage on level
accuracy than on ranking.

### IV.2 The admissibility audit

Before running encoder backtests, the training provenance of each encoder was audited against the
backtest years. The audit found that the two task-aligned encoders could not legitimately be
backtested with their existing checkpoints. The ftvol encoder was trained on the volatility labels
of all pre-2025 filings and its best epoch was selected on validation-year IC. The volaware
encoder's contrastive pairing used within-year forward-volatility deciles across all pre-2025
filings. A backtest of, say, year 2018 with these checkpoints would evaluate an encoder that had
already seen the 2018 labels during its own training, which is lookahead leakage even though the
downstream head is trained cleanly. This is precisely the temporal-leakage failure mode the
financial-ML methodology literature warns about (Lopez de Prado 2018), one level removed: the
leak is inside the representation, not the predictor. For the same reason, the change features
derived from those encoders' embeddings are excluded from all backtest lanes
(`phase5/eval_common.py` enforces this).

The practical consequence: the in-period parity of ftvol and volaware in Part III.3 could not be
confirmed or refuted by any backtest using the existing checkpoints. It required a retrain under a
clean protocol.

### IV.3 The clean-protocol retrain and the final verdict

The decisive experiment retrains the supervised encoder with a strict temporal cutoff: training
only on filings from years before 2017, epoch selection on 2017 (the last pre-cutoff year), the
checkpoint then frozen, all on the corrected corpus. Years 2018 to 2024 are never touched by the
encoder in any way, which makes the 2018 to 2024 expanding-window backtest admissible by
construction. The retrained encoder (ftvol2018) is strong in-period: filing-level IC of 0.635 on
its 2017 selection year, from text alone, confirming that the supervised objective learns a real
signal.

| lane | mean IC 2018–2024 | paired vs struct+tfidf | paired vs structured [hgb] |
|---|---|---|---|
| struct+enc[ftvol2018] [ridge] | 0.504 | −0.057 (p = 0.087) | −0.007 (p = 0.83) |
| struct+enc[ftvol2018] [hgb] | 0.497 | −0.064 (p = 0.097) | −0.014 (p = 0.64) |
| struct+tfidf [sparse] (reference) | 0.561 | — | +0.050 (p = 0.11) |

Out of period, the encoder collapses. Its embeddings add nothing over the structured features
alone (the difference is statistically zero) and it loses to the count model by about 0.06 IC.
Combined with Part III.3, the verdict of the whole study is: **no encoder configuration beats the
count-based model under any admissible protocol.** The apparent parity of the original ftvol on
the validation year is now explained: it came from training on all pre-2025 labels plus epoch
selection on the evaluation year, both of which the clean protocol removes.

The result also carries a positive characterisation, which is the more interesting contribution.
Task-aligned training closes the gap in-period and forward transfer erases it. The dense
text-to-volatility mapping a supervised encoder learns is era-specific: it captures the
relationship between the training era's risk vocabulary and volatility, and that relationship
drifts. The TF-IDF lanes are immune to this not because bag-of-words is a better representation in
any single year, but because the lexical features are transparent levels that an annually refit
linear head can re-weight each year at negligible cost. This reading is consistent with the
concept-drift framework (Lu et al. 2019) and with the time-varying text-return relationships
documented on this exact disclosure section by Magner et al. (2025).

One caveat is stated for fairness. The frozen ftvol2018 checkpoint faces a staleness handicap of
one to eight years relative to the annually refit TF-IDF lanes. Removing it would require
retraining the encoder inside every backtest window (twelve GPU fine-tunes), which was out of
scope. The one deploy-realistic data point available, the original ftvol trained on everything
before 2025 and evaluated on 2025, reaches parity and not superiority even with zero staleness,
which bounds how much the handicap can be hiding.

---

## Part V — Disclosure-change features (the Lazy-Prices branch)

Cohen, Malloy and Nguyen (2020) showed that year-over-year changes in 10-K text predict returns,
with the alpha arising because most filings change very little. The boilerplate-persistence
finding of Part I.1 makes this branch a natural extension: if the level of risk language is
persistent, perhaps the changes carry incremental information about volatility. Kravet and Muslu
(2013) report exactly that for risk-word counts.

For each consecutive firm-year filing pair, the following features were built
(`dataset_config/build_change_features.py`): TF-IDF cosine similarity between the two filings
(`chg_lex_cos`), embedding cosine similarity under the admissible encoders
(`chg_enc_cos_dual/sbert/bge`), the Jaccard overlap and length ratio, and the fraction of
paragraphs with no close match in the prior filing (`chg_new_para_frac`), over 6,713 firm-year
pairs. The features are valid by construction checks: the median lexical similarity dips visibly
in fiscal 2020 (0.964 against roughly 0.976 in normal years), which is the COVID rewrite of risk
sections showing up exactly where it should.

| condition | 2018–2024 IC (t) | 2013–2024 IC (t) |
|---|---|---|
| struct+change [hgb] | 0.492 (4.8) | 0.575 (8.7) |
| struct+tfidf+change [sparse] | 0.550 (5.9) | 0.606 (10.5) |
| struct+tfidf [sparse] (reference) | 0.561 (5.9) | 0.614 (10.4) |

The verdict is null. Adding change features to the count model sits slightly below the count model
on both windows (paired differences of −0.011 and −0.007, p = 0.37 and 0.28), and change features
without TF-IDF sit below the structured baseline. Disclosure change, whether lexical or semantic,
adds no incremental volatility signal over the level representation. The interpretation is
consistent with the source literature: the Lazy-Prices effect is a returns phenomenon driven by
investor inattention to changes, whereas for volatility the informative quantity is the level of
risk language, not its year-over-year shift.

---

## Part VI — Earnings-call tone (ongoing)

Stage C asks whether a second text modality, the earnings call, adds signal. The 2025 call
collection turned out to postdate the 10-K filings, so a filing-level test was not possible.
The question was instead tested call-anchored: does call tone predict the 30-day volatility
following the call? A first, crude test (tone against persistence, linearly) found nothing.
Re-tested inside the full model (structured block plus tone under an HGB head, leave-one-quarter-
out), tone adds +0.039 IC over structured alone (0.594 against 0.555), roughly the same increment
that 10-K text adds. The methodological lesson is worth keeping in the report: a feature can add
nothing on its own and still add real value in combination, so incremental tests must be run
inside the full model. The status is suggestive and unconfirmed (n = 152 calls, a single 2024 to
2025 regime). A small confirmation collection (the 406 pre-filing 2025 calls) is specified in
`dataset_collection_discussion/calls_collection_spec_2025.md` and remains pending.

---

## Part VII — Conclusions and contributions

### VII.1 The defended benchmark

The final model of the study is the structured feature block plus full-text TF-IDF under a sparse
ridge head. Its credentials: mean IC 0.614 with t = 10.4 over the twelve-year expanding-window
backtest, R²_log 0.226 on the 2025 validation year, a text increment over the fair structured
baseline of +0.011 IC (p = 0.057) with a decisively significant accuracy gain, and survival
against every challenger fielded across the study: two heads, eight text representations, three
pooling schemes, five encoders including a modern general-purpose embedder and two task-supervised
ones, topic exposures, change features, and full fusion models, all under leakage-audited
conditions extending to the training provenance of the encoders themselves.

### VII.2 What the study contributes

1. **A defended benchmark rather than an asserted one.** The original "TF-IDF wins" was one
   comparison on a flawed panel; the same conclusion now stands on a corrected panel against an
   exhaustive and fair grid.
2. **A characterisation of why the count model wins.** The volatility signal in 10-K risk text is
   lexical-level, persistent and count-representable. Task-aligned encoders can learn it in-period
   but the learned mapping is era-specific and does not transfer forward, while annually refit
   lexical features stay current by construction. To our knowledge this includes the first honest
   out-of-period evaluation of a volatility-supervised text encoder on this task.
3. **Three methods findings from the audit,** each of independent use to practitioners:
   correcting a 74-percent mispairing of text and labels moved the headline by one thousandth
   (boilerplate persistence, corroborating the premise of Lazy Prices from a new direction);
   equalising the text budget between representations did not rescue the encoders (a fair-input
   protocol for count-versus-embedding comparisons); and holding the prediction head fixed
   shrank the reported text increment fivefold (a fair-head protocol for ablation ladders).
4. **A null result with a mechanism.** Disclosure-change features do not help volatility
   prediction, and the returns-versus-volatility contrast with Lazy Prices explains why not.

### VII.3 Honest caveats

The text increment on ranking is statistically suggestive, not conclusive (p = 0.057 over twelve
years); each additional evaluation year tightens it. The frozen-encoder staleness handicap in the
clean-protocol test is real and only bounded, not removed, by the original-ftvol data point.
Cross-sectional level R² in individual backtest years is noisy and sometimes negative; the ranking
metric is the reliable lens. The Stage C call-tone result is a pilot. The 122 orphaned filings
remain excluded pending label recomputation.

### VII.4 Open questions

1. Does per-window encoder retraining (twelve GPU fine-tunes) close any of the out-of-period gap,
   or is the era-specificity fundamental? Out of scope for the current timeline; the design is
   specified.
2. Does call tone survive a filing-level test on the pre-filing 2025 collection?
3. Does the struct+tfidf increment cross p = 0.05 as evaluation years accumulate?
4. How does the dual-contrastive replication compare quantitatively with Chiu et al. (2025) on
   their own task, as opposed to on ours?

---

## Appendix — Correction ledger (old numbers → corrected numbers)

Every number below changed between the IPP-era write-ups and this document, for the reasons given
in Part I. The old values combine the broken panel with head-confounded comparisons and should no
longer be cited.

| quantity | old value | corrected value | driver |
|---|---|---|---|
| structured baseline, val-2025 IC | 0.570 | 0.591 (ridge) / 0.558 (hgb) | panel fix + head reported explicitly |
| struct+tfidf, val-2025 IC | 0.611 | 0.603–0.610 | panel fix |
| struct+tfidf, val-2025 R²_log | 0.276 | 0.220–0.226 | panel fix |
| struct+encoder(dual), val-2025 IC | 0.602 | 0.582 | panel fix + full-text encoding |
| text increment, val-2025 | +0.041 | +0.012 | same-head comparison |
| text increment, 7-yr backtest | +0.050 (p = 0.090) | +0.016 (p = 0.098) | same-head comparison |
| text increment, 12-yr backtest | — | +0.011 (p = 0.057) | new primary estimate |

*Sources for verification: `phase5/STRESS_TEST_RESULTS.md` (all stress-test tables),
`phase5/out/stress_grid_*.json` (machine-readable results), `original_pipeline_details.md`
(original pipeline), `Literature_agent_study_extended.md` (citation provenance; the verification
log at the end of that file records the claims that failed PDF verification across two rounds on
2026-07-07 and their final replacements, all already reflected in this document: the
long-document windowing precedent is Beltagy et al. 2020 Section 2 with Yang et al. 2020 and
Chiu et al. 2025, with SBERT and BERT both dropped from that claim, and FinMTEB contains no
temporal-shift analysis and is cited only for its bag-of-words-versus-dense finding). Momentum is deliberately cited through Ang et al. (2006), where it
appears as a standard control variable, rather than through Jegadeesh and Titman (1993), which
could not be accessed for reading; since momentum enters this study only as a control feature,
the accessible precedent is the appropriate one. Add Jegadeesh and Titman as a secondary
origin citation only if a copy is obtained.*
