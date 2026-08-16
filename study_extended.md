# Predicting Forward Volatility from 10-K Risk Disclosures: The Extended Study

**Author:** Sarthak Vishnu (s2880814) · **Supervisor:** Prof Tiejun Ma · University of Edinburgh
**Date:** 2026-07-06 (last revised 2026-08-05: volatility-window spectrum, the calls-versus-insiders
comparison, and the bounded generative-LLM pilot)
**Status:** Report-draft narrative. This document supersedes `study.md` as the working account of the
study. All citations are grounded in `Literature_agent_study_extended.md`, `Literature_agent.md`,
and `Literature_agent_phase3.md`. All numbers trace to `phase5/STRESS_TEST_RESULTS.md` and the
Phase 1 to 5 result files on this branch.

**How this document is organised.** The reader is assumed to know the original pipeline from the
IPP report and nothing after it. Part 0 therefore restates the original pipeline in full, so this
document stands alone, and closes with the analysis of its results that set the direction for the
rest of the study. Part I describes the reframing of the research question around a structured
financial baseline, together with the evaluation protocol of the extended study. Part II presents
the primary results under that protocol. Part III describes the robustness extension, which is a
deliberate methodological adaptation beyond the original design, and the final verdict it
produced. Part IV and Part V cover the disclosure-change branch and the earnings-call analysis.
Part VI reports the insider-trading extension, which asks when — for which firms — the text
increment is earned, and closes by setting the two event-data extensions against each other on the
common horizon axis introduced in Part III.4. Part VII states the conclusions and contributions,
and records in VII.5 why the generative-language-model route is scoped out, including the bounded
pilot run to test that judgement rather than assert it.

---

## Part 0 — The original pipeline

### 0.1 Research framing and target

The task is to predict a firm's 30-day forward realised volatility from the text of its SEC 10-K
Item 1A "Risk Factors" disclosure. Volatility is defined as the standard deviation of daily log
returns over the 30 trading days strictly after the filing date, annualised by multiplying by the
square root of 252. The corresponding lagged measure uses the 30 trading days strictly before the
filing date.

The window length is close to the one-month realised-volatility horizon of the HAR-RV literature
without being identical to it: Corsi (2009, §3.2) defines the monthly component as 22 trading
days, and this study uses a slightly longer 30-trading-day window so that every filing's label
closes comfortably inside the sample. The distinction is worth stating rather than glossing,
because 30 trading days is approximately six calendar weeks and is therefore this study's own
design choice rather than a convention it inherits. Part III.4 removes the need to defend the
choice at all, by re-running the entire evaluation across eight windows from 3 to 90 trading days.

The modelling target is the natural logarithm of forward volatility, because
volatility is approximately log-normal and strictly positive, and using the log once and
everywhere keeps error metrics comparable across all models.

Two metrics are reported throughout. The first is the within-year cross-sectional Spearman rank
correlation between predicted and realised volatility, referred to as the information coefficient
(IC). It answers the question "did the model rank firms correctly from least to most volatile
within a year?". The second is R² on the log target (R²_log), which measures how much of the
variance in the volatility level the model explains. Ranking is the primary lens because the
economic use of such a model is cross-sectional (which firms are riskier than which), while the
level is dominated by market-wide conditions.

For context on magnitudes, Grinold and Kahn (1999) provide a concrete benchmark: an information
coefficient of about 0.06 already implies an information ratio above one, which places a forecaster
in the top decile of active managers, even though such an IC corresponds to calling the direction
of returns correctly only about 53 percent of the time. Volatility is a far more forecastable
quantity than returns: Corsi (2009, Table 4) reports in-sample HAR-RV R² of 0.565 for USD/CHF,
0.707 for the S&P 500 and 0.236 for US Treasury bonds using only past realised volatility, whereas
return-prediction R² is typically below 2 percent. ICs in the range 0.5 to 0.6 for volatility ranking are therefore
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
lexical task: the frequency of specific risk language in the text carries the signal, and a bag-of-words
representation captures it directly, while semantic embeddings blur it.

### 0.7 What the original results implied

The original results set the agenda for the rest of the study. The headline finding is a
statement about text representations only, and analysing it against the literature exposed two
limits on what it could mean. First, the only non-text predictor in the entire pipeline was
lagged volatility, one number, even though the asset-pricing literature establishes a set of
structured firm characteristics as first-order determinants of volatility. Any comparison between
text representations was therefore a comparison between models that all lacked the dominant
predictors, so no conclusion about the value of disclosure text could yet be drawn from it.
Second, the question itself ("which text representation ranks best?") is weaker than the question
the literature actually cares about, which is whether disclosure text adds predictive power over
the strong quantitative baseline.

Analysing the original results against those two limits pointed to a specific programme, and it
is the programme the remainder of this document reports: build the structured baseline the
original pipeline never had, re-pose the question as one of incremental value over that baseline,
and tighten the evaluation protocol so that the small increments such a question involves are
measured fairly and with statistical confidence. Part I describes the reframing and the
protocol, and the results follow from Part II onward.

---

## Part I — The reframing: incremental value over a structured baseline

### I.1 Why the question changed

The original pipeline asked which text representation ranks firms best. The literature's actual
question is different: given that a set of structured firm characteristics is known to drive
volatility, does disclosure text add anything on top? Answering the second question requires
building the structured baseline that the original pipeline never had.

### I.2 The structured feature block

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

### I.3 The ablation ladder and its protocol

The evaluation is an ablation ladder over feature blocks: structured features alone, then
structured plus a text representation, where the text representation is TF-IDF, an encoder
embedding, topic exposures, change features, or combinations of these. Each condition is scored
on the validation year and, in Part III, on expanding-window backtests, over a single evaluation
panel keyed uniquely by firm and filing year. Because the quantity of interest is a small
increment on top of a strong baseline, the protocol is designed so that every comparison isolates
exactly one difference. Three rules apply throughout.

First, the prediction head is held fixed within every comparison. When the model class and the
feature set change at once, the credit assignment between them is arbitrary, which is the
ablation principle articulated by Lipton and Steinhardt (2019). An increment attributed to
"adding text" must therefore come from a comparison in which only the feature set changes, so
both heads (a ridge regression and a gradient-boosted tree, HGB) are reported for every
condition, and increments are always quoted between same-head arms.

Second, every text representation receives the full text. TF-IDF ingests the complete filing by
construction, so the encoders are given the same input through windowed encoding
(`topics/encode_paragraphs.py`): each long paragraph is split into overlapping 256-token windows
with a stride of 192 tokens, every window is encoded, and the window vectors are mean-pooled back
to a paragraph vector before the usual paragraph-to-filing pooling, for a total of 462,590
windows over 176,616 paragraphs. Segment-encode-pool is the established treatment of documents
that exceed a transformer's context window (Beltagy et al. 2020, Section 2); the financial-NLP
precedent is the hierarchical encoding of long transcripts in Yang et al. (2020, HTML), and the
256-token paragraph unit follows Chiu et al. (2025). Equal input budgets ensure that a comparison
between representations is a comparison between representations, not between the amounts of text
each was allowed to read.

Third, statistical comparisons are paired. Within each evaluation year the two models predict the
same firms, so their per-year ICs are compared as matched pairs across years, and their per-firm
squared errors are compared by Diebold-Mariano tests.

---

## Part II — Primary results

These are the primary results of the study. The evaluation design is the single-split design of
the original pipeline (train before 2025, evaluate on 2025), which is the design most comparable
to the literature, run under the protocol of Part I.3.

### II.1 How to read the tables

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

### II.2 The anchor results

| condition | IC | 95% CI | R²_log | DM p vs struct+tfidf |
|---|---|---|---|---|
| lagged [hgb] | 0.457 | — | −0.130 | — |
| structured [hgb] | 0.558 | [0.487, 0.628] | 0.094 | 0.001 |
| structured [ridge] | 0.591 | [0.522, 0.659] | 0.175 | 0.000 |
| tfidf+lag [sparse] | 0.508 | [0.428, 0.585] | 0.069 | 0.000 |
| **struct+tfidf [sparse]** | **0.603** | [0.534, 0.670] | **0.226** | reference |

Three statements summarise this table. First, the structured baseline alone beats the entire
original text-only pipeline: 0.591 against the TF-IDF-plus-lagged floor of 0.508, a gap of 0.083.
The standard quantitative characteristics rank firm volatility better than any text-only method,
which retroactively explains why the original pipeline's comparisons were all fought below this
level. (Earlier drafts quoted this floor as 0.541, which is the same condition evaluated at a fixed
sparse-ridge penalty of 10. The harness tunes that penalty on 2024 and selects 30, which costs
0.033 IC on 2025; 0.508 is therefore the figure produced by the same tuned pipeline as every other
row of this table, and the one anchored in `phase5/stress_grid.py`. The correction widens the gap
rather than narrowing it.)
Second, text on top of the structured baseline adds: struct+tfidf reaches 0.603 IC and raises
R²_log from 0.175 to 0.226, and the Diebold-Mariano test says this accuracy gain is highly
significant. Third, the IC increment (+0.012 on this single year) is inside the single-year
confidence interval, which is exactly why Part III moves to multi-year paired tests before claiming
anything about it.

### II.3 The encoder grid

The full grid scores every encoder under both heads with the full-text windowed inputs of
Part I.3. The encoders are: `dual` and `sbert` (the original similarity-trained contrastive encoder
and the off-the-shelf sentence transformer), `bge` (BAAI bge-base-en-v1.5, a modern strong
general-purpose embedder, included to address the objection that the original encoders were
simply not good enough), `volaware` (the Stage B contrastive encoder whose positive pairs are
paragraphs from firms in the same within-year forward-volatility decile), `three` and `three_lora`
(the project's own three-view contrastive encoders of Part 0.4, adding the sector view under full
fine-tuning and under LoRA respectively), and `ftvol` (an encoder fine-tuned end-to-end on the
volatility regression target).

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
| struct+enc[three] [ridge] | 0.564 | [0.489, 0.634] | 0.100 | 0.000 |
| struct+enc[three] [hgb] | 0.600 | [0.532, 0.662] | 0.245 | 0.587 |
| struct+enc[three_lora] [ridge] | 0.588 | [0.517, 0.659] | 0.145 | 0.001 |
| struct+enc[three_lora] [hgb] | 0.596 | [0.527, 0.661] | 0.198 | 0.430 |
| struct+enc[ftvol] [ridge] | 0.606 | [0.540, 0.672] | 0.185 | 0.207 |
| struct+enc[ftvol] [hgb] | 0.598 | [0.529, 0.666] | 0.167 | 0.155 |
| struct+enc[ftvol, topk_risk] [hgb] | 0.607 | [0.538, 0.677] | 0.194 | 0.454 |
| struct+tfidf_svd [ridge] | 0.594 | [0.526, 0.661] | 0.167 | 0.000 |
| EVERYTHING svd+enc[three]+chg [hgb] | 0.608 | [0.542, 0.673] | 0.215 | 0.777 |
| EVERYTHING tfidf+enc[three]+chg [sparse] | 0.553 | [0.476, 0.623] | −0.004 | 0.000 |

The pattern is clean. The generic semantic encoders (dual, sbert, bge) sit at or below the no-text
structured baseline and are significantly less accurate than struct+tfidf under a Diebold-Mariano
test in at least one head. The inclusion of bge settles the "you never tried a strong modern
embedder" objection: it loses too, even with the full text delivered through the windowed
protocol. The direction of this result has independent support: the FinMTEB benchmark reports that
bag-of-words representations outperform dense embedding models on financial semantic-similarity
tasks (Tang and Yang 2025), so count-based representations beating dense ones on financial text is a
documented phenomenon, not an idiosyncrasy of this pipeline. Which encoders reach statistical
parity with the count model depends on the head. Under the ridge head only ftvol does, at p = 0.207;
every other encoder is significantly less accurate, both three-view variants included. Under the
tree head the picture inverts, and the only two conditions that fail parity are sbert (p = 0.043)
and bge (p = 0.023), the two off-the-shelf general-purpose embedders, with dual marginal at 0.051.
volaware, three, three_lora and ftvol are all indistinguishable from the reference there. The line
therefore runs along domain adaptation rather than along task alignment: an encoder that has been
trained on this corpus reaches parity under a flexible head whether or not it ever saw the
volatility label. One row leads the table on a single axis, and it is the project's own sector-view
encoder — struct+enc[three] [hgb] posts the highest R²_log in the grid at 0.245, above the
reference's 0.226, while its IC of 0.600 sits mid-pack. That is a level-accuracy gain rather than a
ranking gain, and it is not accompanied by any IC advantage. None of the parity rows beats the
reference on IC, and the kitchen-sink fusion of every representation at once does not either: under
the sparse head it collapses to 0.553 IC with a negative R²_log, the worst text-carrying condition
in the table. The encoder named inside the two EVERYTHING rows and inside the attention-pooling row
is selected by maximising tree-head IC on the same validation rows the row is then scored on, so
that identity is a within-sample artefact and carries no evidential weight; `three` takes it at
0.600 against ftvol's 0.598, a margin of 0.002 inside a confidence interval roughly ±0.06 wide.
This grid also supersedes the original
pipeline's encoder comparison: task-supervised fine-tuning, which the original post-mortem
speculated would have worked better, was built (following the precedent of end-to-end
volatility-supervised text encoders in Qin and Yang 2019 and Yang et al. 2020), and it reaches
parity but not superiority.

One caveat belongs to this table and it matters for everything in Part III. The parity rows are
in-period results for encoders whose training saw pre-2025 volatility labels, and the ftvol
checkpoint was additionally epoch-selected on the validation year itself. These rows are honest as
exploratory evidence and inflated as claims. Establishing what a task-aligned encoder can really
do requires the clean protocol of Part III.3.

---

## Part III — The robustness extension

Everything in this part goes beyond the original evaluation design. It is presented as a
deliberate methodological adaptation, adopted for a specific reason: the single-year comparison in
Part II cannot statistically separate models 0.01 apart, and, more seriously, it cannot detect a
class of leakage that turned out to be present. Both problems require evaluation across many
years.

### III.1 The expanding-window backtest

The protocol is the standard walk-forward design for financial prediction (the reference treatment
of why k-fold cross-validation is inappropriate for financial time series is Lopez de Prado 2018,
chapter 7, and of walk-forward evaluation and lookahead bias generally, chapters 11 and 12). For each test
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
superiority over pure persistence is unambiguous (+0.078 IC over lagged [hgb] on the twelve-year
window, p = 0.001). This is the
honest shape of the headline result: a strong structured baseline, plus a small, consistent,
statistically suggestive lexical text increment, with a much clearer text advantage on level
accuracy than on ranking.

### III.2 The admissibility audit

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

The practical consequence: the in-period parity of ftvol and volaware in Part II.3 could not be
confirmed or refuted by any backtest using the existing checkpoints. It required a retrain under a
clean protocol.

### III.3 The clean-protocol retrain and the final verdict

The decisive experiment retrains the supervised encoder with a strict temporal cutoff: training
only on filings from years before 2017, epoch selection on 2017 (the last pre-cutoff year), the
checkpoint then frozen, all on the study corpus. Years 2018 to 2024 are never touched by the
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
Combined with Part II.3, the verdict of the whole study is: **no encoder configuration beats the
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

### III.4 The horizon of the text increment

Every result so far is stated at one horizon: realised volatility over the 30 trading days after
the filing. As Part 0.1 notes, that is close to but not the same as the one-month horizon of the
realised-volatility literature, whose monthly component Corsi (2009) sets at 22 trading days; it
is this study's own design choice rather than an inherited convention, and the last robustness
question is whether the study's conclusions are specific to it. Labels were therefore regenerated across a spectrum of seven further horizons — 3, 5, 7,
10, 20, 60 and 90 trading days — under the identical definition (standard deviation of daily log
returns over the window strictly after — and, for the lagged predictor, strictly before — the
filing date, annualised), from the same price sources as the structured features. The
construction was validated by requiring it to reproduce the study's 30-day labels exactly before
any new horizon was read (agreement to rounding precision on every comparable filing), and
windows are required to lie adjacent to the filing date, so a firm whose price history has ended
receives no label rather than a stale one. The fair pair of Part III.1 was then rerun per horizon
— persistence, structured [ridge], struct+tfidf [sparse], each with the horizon-matched lagged
volatility — on the same 2018–2024 expanding-window backtest and the 2025 validation year.

The spectrum begins at three days rather than one for a reason of construct rather than of
convenience. The label is a sample standard deviation, which is undefined for a single
observation; answering the one-day case would mean substituting a different estimator, the
absolute return scaled by the annualisation factor, partway along the curve. One estimator across
the whole spectrum is worth more than one extra point on it, and at a one-day horizon the target
would in any case cease to be a dispersion measure and become a single signed magnitude — a
different construct rather than a shorter version of the same one.

| horizon | lagged IC | structured [ridge] | struct+tfidf [sparse] | text ΔIC (paired) | p | val-2025 struct / struct+tfidf |
|---|---|---|---|---|---|---|
| 3 days | 0.193 | 0.330 | 0.351 | +0.021 | 0.020 | 0.366 / 0.386 |
| 5 days | 0.289 | 0.417 | 0.441 | +0.023 | 0.017 | 0.514 / 0.523 |
| 7 days | 0.299 | 0.444 | 0.470 | +0.027 | 0.014 | 0.564 / 0.582 |
| 10 days | 0.314 | 0.462 | 0.482 | +0.020 | 0.076 | 0.578 / 0.596 |
| 20 days | 0.401 | 0.513 | 0.532 | +0.020 | 0.119 | 0.638 / 0.637 |
| 30 days | 0.461 | 0.546 | 0.561 | +0.016 | 0.098 | 0.591 / 0.610 |
| 60 days | 0.528 | 0.592 | 0.591 | −0.001 | 0.851 | 0.709 / 0.718 |
| 90 days | 0.534 | 0.607 | 0.602 | −0.005 | 0.477 | 0.739 / 0.751 |

Three findings. First, the model ranking is horizon-robust: struct+tfidf at or above structured,
structured above persistence, at every one of the eight horizons on the backtest, with the
validation year agreeing throughout. No conclusion of Parts II and III is an artifact of the
30-day choice.

Second, the text increment itself has a term structure, and with the short end filled in it
resolves into a single-peaked curve rather than a decline: it rises from +0.021 at three days to
a maximum of +0.027 at seven, then falls monotonically to +0.016 at thirty and to zero at sixty
and ninety. Measured relative to the structured baseline the shape is sharper still, text being
worth six percent of the structured information coefficient at the short end and minus one
percent at the long end. The direction is the opposite of the naive expectation that a
slow-moving annual disclosure should matter more at longer horizons. The resolution is visible in
the first column: overall predictability rises steeply with horizon (persistence alone climbs
from 0.193 to 0.534), because longer realised-volatility windows are smoother and increasingly
dominated by the persistent component of volatility — and that component is precisely what the
structured block's multi-horizon trailing measures already carry. What the text contributes is
the transient, near-filing component of uncertainty, which washes out of the target as the
horizon lengthens. The four-point version of this table supported that reading by inference; the
eight-point version exhibits it, with the peak located rather than assumed.

Third, and unexpectedly, the increment attains conventional significance at the short end: p =
0.020, 0.017 and 0.014 at three, five and seven days. Part VII.4 poses as an open question
whether the increment crosses the five percent threshold as evaluation years accumulate; the
answer turns out to be that it crosses by shortening the horizon instead. The mechanism is
visible in the per-year dispersion — the information-coefficient t-statistic of the full model
rises from 5.9 at thirty days to 11.7 at three — because short realised-volatility windows are
far less exposed to the 2020 regime break that dominates the thirty-day panel's year-to-year
variance. The qualification is stated plainly and is not a formality: eight horizons were swept
without a multiplicity adjustment, so p = 0.014 at seven days is not a five-percent-level claim
standing alone. The evidence this table offers is the shape of the curve, which no multiplicity
argument touches, with the short-end significance as corroboration rather than as the headline.

Read alongside Parts V and VI, this completes the same absorption logic in a third dimension:
earnings-call tone showed *when* in time a text signal stops adding value, the
insider-conditioning analysis showed *for which firms* it adds most, and the term structure shows
*at which horizon* it exists at all. The headline increment is thus a property of the horizon at
which the study states it — near its maximum from three to twenty days, still present at thirty,
and demonstrably absent by sixty. The comparison at the close of Part VI takes this
further, because the three effects turn out not to peak at the same horizon.

---

## Part IV — Disclosure-change features (the Lazy-Prices branch)

Cohen, Malloy and Nguyen (2020) showed that year-over-year changes in 10-K text predict returns,
with the alpha arising because investors under-react to the changes that do occur, and their
starting observation is that most 10-K filings change very little from year to year. Kravet and
Muslu (2013) report the volatility-side analogue: annual increases in the volume of risk
disclosure are associated with higher stock-return volatility. This corpus shows the same
persistence, and a stale-text experiment quantifies it directly: re-pairing roughly three
quarters of the filings with the adjacent year's risk text instead of their own leaves the
struct+tfidf validation IC essentially unchanged (0.611 against 0.610). Year-stale risk text
predicts volatility almost exactly as well as current risk text, so the text signal measured in
Part II sits in the persistent level of the risk language rather than in its year-over-year
freshness, which is consistent with Campbell et al. (2014), who established that these
disclosures are informative rather than empty boilerplate. That persistence sharpens the question
of this part: if the level of risk language barely moves, do the movements that do occur carry
incremental information about volatility?

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

## Part V — Earnings-call tone: a horizon-contrast finding

Stage C asks whether a second text modality, the earnings call, adds signal. Predicting
volatility from earnings-call text is itself an established task: Qin and Yang (2019) and Yang et
al. (2020, HTML) supervise text encoders on call transcripts against a volatility target, and
Sawhney et al. (2020, VolTAGE) model call tone for volatility specifically. Stage C imports that
modality into the present incremental-value protocol, asking not whether call tone predicts
volatility in isolation — which the prior work establishes — but whether it adds signal over the
structured baseline and the 10-K text. The question was answered in two steps at two different
anchors, and the contrast between them is the finding.

The first step was call-anchored. The initially available calls postdated the 10-K filings, so
the pilot asked whether a call's tone predicts the 30-day volatility following the call itself.
A crude test (tone against persistence, linearly) found nothing; re-tested inside the full model
(structured block plus tone under an HGB head, leave-one-quarter-out), tone adds +0.039 IC over
structured alone (0.594 against 0.555), roughly the same increment that 10-K text adds. The
methodological lesson stands: a feature can add nothing on its own and still add real value in
combination, so incremental tests must be run inside the full model.

The second step became possible when the full transcript history was obtained (39,501 unique
transcripts from a single corpus, covering 2006 to early 2025), allowing the pre-registered
filing-level test at full scale. Tone features were attached to each filing from the most recent
call strictly preceding the filing date (a leakage-free construction; 88.3 percent of the panel
matched, at least 95 percent in every backtest year). Two independent evaluations were run. Within the 2025 validation
cross-section, a five-fold cross-validated comparison with identical folds and identical heads
per pair found every tone increment flat or negative (for example, minus 0.017 IC under the HGB
head on the full panel; every confidence interval straddles zero, and the one significant
accuracy test favours the model without tone). Across the 2018 to 2024 expanding-window backtest,
where the historical coverage makes calls a fully backtestable family, structured plus tone
matches structured alone (0.515 against 0.518, paired p = 0.395) and the full model with tone
matches it without (0.561 against 0.561, p = 0.436).

A natural objection to the second result is that its null might be an artifact of the thirty-day
label, the tone signal being real but living at a horizon the study does not measure. The horizon
machinery of Part III.4 answers this at no additional data cost, and it answers it decisively at
the filing anchor. Across the full spectrum from three to ninety days, adding tone to the
structured model produces increments of +0.011, +0.006, −0.003, −0.001, −0.006, −0.005, +0.002
and −0.003, and adding it to the full model with 10-K text produces increments whose absolute
value never exceeds 0.001 at any horizon. The only cells that are not null run in opposite
directions — a weak positive at three to five days and a small but nominally significant negative
at twenty — which is what an uncorrected eight-horizon sweep of a true zero is expected to look
like. Stage C's filing-level null is therefore a null at every horizon, not a property of the one
at which it was first stated. The short-horizon flicker is worth one further observation: it
appears over the structured model alone and vanishes over the full model, so where tone carries
anything at the filing anchor, the 10-K's own text has already captured it. As second text
modalities go, calls are a substitute for the filing text rather than a complement to it.

At the call anchor the same sweep can be run, and is reported as the pilot it is, on 152 filings
in a single 2025 regime. Tone's increment over the structured model traces a hump across the
spectrum — negative at three days, then +0.079, +0.084, +0.142 and +0.134 at five, seven, ten and
twenty, +0.039 at thirty, and negative again at sixty and ninety. Two things follow, both modest.
The headline +0.039 at thirty days sits on the decaying right shoulder of that hump rather than at
its peak, so the call-anchored effect, such as it is, is a two-to-four-week phenomenon rather than
a monthly one. And the sign reversal at three days is the first indication that the call signal
and the 10-K text signal do not live at the same horizon at all — a point Part VI's comparison
returns to with a second, better-powered instance of it. At this sample size the curve is jagged
and no individual cell should be read; the shape is the most that can be claimed.

Both filing-anchor and call-anchor results are correct; they measure different horizons. A
plausible reading, offered as an interpretation rather than an established mechanism, is that
management tone carries genuine
volatility information at the call date, but that by the filing date, one to three months later,
the structured block has already absorbed it: the realised-volatility windows measured at filing
span exactly the post-call period the tone predicted. Stated as a finding, Stage C identifies the
condition under which a second text modality stops adding value: an anchor placed after the
market has had time to impound the signal into the structured features. At the call horizon tone
is informative; at the filing horizon the 10-K task already possesses fresher information.

---

## Part VI — Insider trading and the risk text: when the text increment is earned

Stage E extends the study with a third information source, insider transactions reported on SEC
Form 4, chosen not as another candidate feature family but because it speaks directly to the
mechanism the study has converged on. Insider trading is the classic observable proxy for
information asymmetry — Frankel and Li (2004) use the profitability and intensity of insider
trades for exactly this purpose — and information asymmetry is a volatility construct: it
measures how much material information is concentrated in few hands, not which direction that
information points.
Every feature is accordingly framed as intensity or dispersion rather than as a trading signal.
Eight features are computed per filing: transaction and distinct-insider counts, net direction by
count and by value, insider disagreement (the fraction of trading insiders on the minority
buy-or-sell side of their firm's window), the officer sell fraction, abnormal trading intensity
(the window trade rate against the firm's own trailing-365-day rate), and the opportunistic
fraction, adapted from Cohen, Malloy and Pomorski's (2012) routine-trade classification: a trade
is scored opportunistic unless the same insider traded in the same calendar month in at least
three prior years. This is a deliberately weaker criterion than CMP's original test — the three
qualifying years need not be consecutive and are not confined to a fixed look-back — applied here
at the trade level, as CMP's own robustness check (their Table III) also does, rather than at the
trader level of their primary specification.

The construction inherits every discipline of the study. Features aggregate a trailing window
from 180 to 3 days before each filing date; the three-day margin covers Form 4's two-business-day
disclosure deadline, statutory since Sarbanes-Oxley §403 (codified at 15 U.S.C. §78p(a)(2)(C)),
so every feature is public at prediction time. Matching is by CIK with a
ticker fallback. Coverage is effectively complete — every panel filing falls inside the Form 4
universe, in every year from 2006 — so, unlike the calls family, these lanes are valid over both
backtest windows. One construct-validity observation from the build: median abnormal intensity is
mildly negative in every year, which is the pre-filing blackout period made visible in the data —
insiders trade less immediately before a 10-K than in their own trailing baseline. This is the
documented behaviour rather than an artifact: Bettis, Coles and Lemmon (2000) find that 78
percent of firms operate explicit blackout policies and that insider trading within them falls
to under a third of its level in permitted windows.

The first question is the level question, and its answer is null. Adding the insider block to the
structured model changes nothing in either window (paired ΔIC +0.000 at p = 0.897 over the
seven-year backtest, +0.001 at p = 0.768 over twelve years), and adding it to the full model
likewise (+0.000 at p = 0.886 and −0.001 at p = 0.585); the 2025 validation rows agree. This is
itself consistent with the study's architecture: the structured block already contains realised
volatility, drawdown and liquidity measures, and whatever elevated insider activity accompanies,
those features appear to reflect.

The second question is the one Stage E was designed for, and it was pre-registered with a
directional hypothesis before the run: does the text increment — struct+tfidf against structured
under matched heads, the study's fair pair — concentrate in the high-insider-activity half of
each year's cross-section? Because the conditioning variables are computed strictly pre-filing,
splitting each test year at its median is knowable at prediction time, so the analysis is
leakage-free by construction. Within each half the paired per-year ΔIC is computed on the same
firms, and the high-minus-low difference is tested across years.

The design itself appears to be this study's own rather than a replication. Both of its ingredients
are standard — median splits of a cross-section on a pre-filing conditioner, and paired-by-year
tests of an incremental information coefficient — but no precedent was found, in the reading behind
this study, for conditioning a *text increment* on an insider-trading state in order to ask where
in the cross-section that increment is earned. The claim is stated as a search result rather than
as established novelty, and it cuts both ways: an untested design carries no external validation of
its behaviour, which is part of why the results below are reported as suggestive and why the
horizon sweep that follows was allowed to overturn one of them.

The two conditioners initially appeared to answer in opposite directions. Conditioned on insider
*disagreement*, the hypothesis holds: the increment is larger where trading insiders are split
(high-minus-low ΔIC of +0.029, t = +2.04, p = 0.088 over seven years, positive in six of seven;
+0.016 at p = 0.158 over twelve). In the disagreement half the per-year text increment reaches
three to four times its unconditional size (+0.083 in 2021, +0.132 in 2023). Conditioned on
abnormal trading *intensity*, the hypothesis appeared to be reversed, the increment being larger
in the low-intensity half (−0.012, t = −2.67, p = 0.037 over seven years, negative in six of
seven; −0.006 at p = 0.227 over twelve).

**That second result does not survive, and is withdrawn.** Re-running the conditioning test at the
seven further horizons of Part III.4 places the intensity effect at −0.001, −0.005, −0.008,
−0.009, −0.007, +0.002 and +0.013 across three to ninety days: no cell significant, none larger in
magnitude than 0.013, and the sign turning positive at both ends of the spectrum. The nominally
significant thirty-day cell has no support from any adjacent horizon, which is the signature of a
single-cell finding in a family that is otherwise null rather than of an effect that happens to be
measured most cleanly at one horizon. It is reported here as withdrawn rather than quietly
dropped, because the discipline that retires it is the same one that credentials its counterpart:
insider disagreement is corroborated at twenty days (+0.029) as well as at thirty (+0.030), and
that agreement between neighbouring horizons is the evidence that the thirty-day cell is not
isolated. The horizon sweep was run to answer a question about robustness; what it also did was
distinguish one of the two conditioning results from the other.

A plausible reading of what remains, offered as an interpretation rather than an established
mechanism, is that disagreement marks unresolved dispersion among the best-informed parties —
precisely the state in which narrative disclosure would still carry incremental information, and
the state Shalen (1993) models as generating additional volatility, dispersion of expectations
being shown there to measure "both the additional volatility and the additional expected volume of
trade associated with noisy information". Read this way, Stage E extends the Stage C absorption
narrative from the time dimension (tone informative at the call anchor, absorbed by the filing
anchor) to the cross-section: the text increment is earned on the firms whose information the
market has not yet impounded. The complementary proposition that had been attached to the
intensity conditioner — that heavy insider trading is itself a transmission channel, Form 4s
becoming public within two business days, so that where insiders have traded the disclosure text
has less left to add — is a coherent mechanism and may well be true, but this study no longer
has evidence for it and does not assert it.

The bounds of the surviving claim are stated plainly. Two conditioners were tested over two
windows and eight horizons with no multiplicity adjustment; the disagreement effect attenuates as
the backtest extends to 2013 (the pattern is concentrated from 2018 onward), its strongest
p-value is 0.088, and the disagreement high-cell is small (52 to 89 firms per year). Stage E
therefore closes as a null level effect together with suggestive, horizon-corroborated evidence
that the text increment is state-dependent — largest where informed parties disagree.

### Comparing the two extensions

The two extensions were built to test one hypothesis along two different dimensions, and it is
worth setting them against each other explicitly rather than leaving them as consecutive
appendices. Earnings calls test the **time** dimension: whether a text signal that is informative
when spoken survives to the filing date one to three months later. Form 4 tests the
**cross-section**: which firms still hold unimpounded information at the filing date. The horizon
spectrum of Part III.4 supplies a third dimension, the timescale of the target itself, and it is
what turns the pair into a comparison rather than a juxtaposition, because both extensions can be
measured on it.

| | earnings calls (Part V) | insider trading (Part VI) |
|---|---|---|
| coverage | 6,859 of 7,367 filings, 2006–2025 | 7,367 of 7,367, 2006–2026 |
| backtestable windows | seven-year only | seven- and twelve-year |
| level effect | null at the filing anchor, at all eight horizons | null on all four arms, both windows |
| conditional or anchored effect | +0.039 IC at the call anchor, n = 152 | +0.030 disagreement conditioning |
| relation to the 10-K text | substitute: nothing added on top of TF-IDF at any horizon | complement: conditions where the TF-IDF increment is earned |

The comparison produces one result that neither extension produced alone. The three effects do
not peak at the same horizon. The unconditional 10-K text increment peaks at seven days and is
near its maximum at three. Insider-disagreement conditioning peaks at twenty to thirty days and
reverses sign, to −0.038, at three. Call tone at the call anchor peaks at ten to twenty days and
likewise reverses at three. The filing's own text, in other words, carries **days**, while both
event families carry **weeks**, and both are invisible or inverted over the first three trading
days after the filing.

This refines the absorption reading rather than restating it. The 10-K prices the transient
uncertainty immediately following its own publication, and that contribution is spent within a
fortnight. What insiders and management convey shows up over a horizon long enough for the market
to work through it, and does not register at all on the timescale where the disclosure text is at
its strongest. The three curves also give the shared account a falsifiable prediction that it
passes: were the mechanism simply that more text signal is better, all three would peak together.
They do not.

One asymmetry should be stated rather than smoothed over, because the two extensions are not
equally credentialled. The Form 4 result rests on seven backtest years at complete coverage; the
call-anchored result rests on 152 filings in a single 2025 regime, and its horizon curve is
correspondingly jagged. The well-powered calls finding is the *filing-anchor null* — 6,859 filings
across seven years and eight horizons, null throughout — and it is that null, not the
call-anchored hump, that stands beside the Form 4 conditioning result as evidence.

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

1. **A defended benchmark rather than an asserted one.** The original "TF-IDF wins" was a single
   comparison on a single evaluation year; the same conclusion now stands against an exhaustive
   and fair grid of challengers and across twelve years of walk-forward backtests.
2. **A characterisation of why the count model wins.** The volatility signal in 10-K risk text is
   lexical-level, persistent and count-representable. Task-aligned encoders can learn it in-period
   but the learned mapping is era-specific and does not transfer forward, while annually refit
   lexical features stay current by construction. To our knowledge this includes the first honest
   out-of-period evaluation of a volatility-supervised text encoder on this task.
3. **A persistence finding.** Year-stale risk text predicts volatility almost exactly as well as
   current risk text (Part IV), which corroborates the low-churn premise of Lazy Prices from the
   volatility side.
4. **A null result with a mechanism.** Disclosure-change features do not help volatility
   prediction, and the returns-versus-volatility contrast with Lazy Prices explains why not.
5. **A horizon-contrast finding for a second text modality.** Earnings-call tone predicts
   volatility at the call anchor (+0.039 IC over structured in combination) yet adds nothing at
   the filing anchor, in 2025 and across seven backtest years. The plausible reading is that the
   signal is absorbed into the structured features during the call-to-filing gap, which
   identifies when an additional text source is worth attaching: only while its information has
   not yet been impounded into cheaper, fresher predictors.
6. **A design for asking where a text increment is earned, and suggestive evidence from it.**
   Conditioning the text increment on pre-filing insider activity (Part VI) indicates it
   concentrates in firms whose insiders disagree — the cross-sectional counterpart of the
   absorption reading, subject to the bounds stated in Part VI, and corroborated across two
   adjacent horizons. The conditioning design is a contribution in its own right: its ingredients
   are standard, but no precedent was found for using an insider-trading state to locate *where in
   the cross-section* a text increment lives, as opposed to testing whether it exists on average.
7. **A term structure for the text increment, and a horizon axis that discriminates.**
   Relabelling the task across eight horizons from 3 to 90 trading days (Part III.4) shows the
   model ranking is horizon-robust while the text increment itself is single-peaked at seven days
   and gone by sixty, reaching conventional significance (p = 0.014 to 0.020) at the short end
   where the 2020 regime break contributes least variance. The study's headline is thereby scoped
   by measurement rather than assumption. The same axis turns out to discriminate among the
   study's own findings: it retired one previously significant conditioning result as a
   lone-horizon artifact, corroborated the other, and showed the three surviving effects to peak
   at *different* horizons — the filing text at seven days, insider disagreement and call tone at
   two to four weeks — which is the substance of the comparison closing Part VI.
8. **A negative result on generative feature extraction, obtained rather than argued.** A bounded
   pilot (Part VII.5) had an open-weights model read the raw insider-transaction record and score
   it directly, on a post-cutoff slice with identifiers stripped. Its scores are statistically
   indistinguishable from the eight hand-crafted features, and both from omitting the data
   altogether: feature engineering was not the binding constraint, the information content of the
   record was. The pilot also yields a stability measurement on real financial records — the score
   moves 59 to 96 percent as much between re-runs of an identical prompt as it does between
   different companies — which is the kind of evidence the instability literature is usually cited
   for in the abstract.

### VII.3 Honest caveats

The text increment on ranking is statistically suggestive, not conclusive (p = 0.057 over twelve
years); each additional evaluation year tightens it. The frozen-encoder staleness handicap in the
clean-protocol test is real and only bounded, not removed, by the original-ftvol data point.
Cross-sectional level R² in individual backtest years is noisy and sometimes negative; the ranking
metric is the reliable lens. The Stage C horizon-contrast interpretation (tone absorbed between
call and filing) is a plausible reading supported by the results, not a demonstrated mechanism.
The same holds for the Stage E conditioning result: conditioners tested over two windows and eight
horizons with no multiplicity adjustment, effects concentrated from 2018 onward — suggestive and
directionally consistent, not confirmatory. One member of that family did not survive the horizon
sweep and is withdrawn in Part VI; it is reported as withdrawn rather than removed, because a
study that only ever adds findings is not being audited. The horizon term structure (Part III.4)
carries its own qualification, which the added short-horizon significance makes more rather than
less important to state: eight horizons were swept without a multiplicity adjustment, so the
evidence is the shape of the curve, not the individual p-values at three, five and seven days. The
generative pilot of Part VII.5 is bounded by design to 2024 and 2025, so its contamination probe
is a placebo check on a post-cutoff slice and says nothing about the backtest years where the
contamination objection actually applies.

### VII.4 Open questions

1. Does per-window encoder retraining (twelve GPU fine-tunes) close any of the out-of-period gap,
   or is the era-specificity fundamental? Out of scope for the current timeline; the design is
   specified.
2. Answered (Part V): call tone does not survive the filing-level test, in 2025 or across the
   2018–2024 backtest, at any of the eight horizons, while remaining informative at the call
   anchor. The open follow-on is whether a shorter call-to-anchor gap (for example a mid-quarter
   anchor) recovers the signal. The horizon sweep now gives that follow-on a target: the
   call-anchored effect peaks at ten to twenty days, so a mid-quarter anchor should be paired with
   a label of that length rather than with the study's thirty-day headline window.
3. Answered in an unexpected form (Part III.4): the struct+tfidf increment crosses p = 0.05 at the
   three-, five- and seven-day horizons rather than by accumulating evaluation years, because
   short realised-volatility windows are far less exposed to the 2020 regime break. Whether it
   crosses at the thirty-day headline horizon as years accumulate remains open.
4. How does the dual-contrastive replication compare quantitatively with Chiu et al. (2025) on
   their own task, as opposed to on ours?
5. Does the state-dependence of the text increment (Part VI) replicate under other
   information-environment conditioners — most naturally news coverage intensity and news tone
   dispersion — and does the disagreement effect strengthen as evaluation years accumulate? Any
   such replication should be run across the horizon spectrum from the outset: that is what
   separated the disagreement result from the abnormal-intensity result, and a single-horizon
   replication would not have been able to tell them apart.

   This question is left open deliberately rather than by omission, and the reason is worth
   recording because news data was assessed for inclusion and declined on two independent grounds.
   The natural corpus, GDELT's Global Knowledge Graph 1.0, identifies firms only as free-text
   organisation names, carrying neither a CIK nor a ticker. Linking it to this study's panel would
   therefore require a temporally valid name-to-firm bridge: renames and restructurings resolving
   to the right entity in the right period, subsidiaries rolling up to their parents, and genuine
   disambiguation for firms whose names are ordinary English words, which naive string matching
   floods with false positives. Every downstream number would inherit the quality of that mapping,
   and the audit described in Part III.2 is this study's own demonstration of how far a subtle join
   defect propagates before it becomes visible; building and validating such a layer to the
   standard applied everywhere else here is a project-sized prerequisite rather than a
   preliminary. Second, the coverage does not align. Every other source in this study — filings,
   structured market features, call transcripts, Form 4 transactions — spans 2006 to 2026, whereas
   GKG 1.0 begins in April 2013 and, after trailing feature windows, is usable from about 2014.
   That forecloses the twelve-year backtest that every other result reports alongside the
   seven-year window, so a news finding would rest on strictly weaker evidence than the results
   standing next to it, and the cross-stage comparability the evaluation protocol is built on
   would not hold for it. The design is therefore kept on the record as specified future work
   whose data-engineering prerequisite exceeds the present study, rather than presented as an
   untested idea.

### VII.5 The generative-LLM route, and why it is scoped out

Every text representation in this study is an *encoder*: TF-IDF, SBERT, BGE, the DAPT model, the
two task-aligned encoders. A distinct use of language models is untested here — prompting a
*generative* model to score each filing on interpretable risk dimensions and feeding those scores
in as dense features. It is the natural next question, and it differs from the encoder route in a
way that could matter: the output is low-dimensional and interpretable, so it might resist the
era-specific drift that defeats the dense encoders (Part III.3) in the way the lexicon-based tone
features do. This subsection records why it is nevertheless out of scope, because the reasons are
substantive rather than merely practical, because one of them is a result this study has already
measured, and because a bounded pilot was run to make sure the position rested on evidence rather
than on assertion. (Sources verified in `Literature_agent_llm_feasibility.md`.)

**The elicited score is not a stable measuring instrument, measured here rather than assumed.**
Before turning to the literature it is worth reporting what this study observed directly. In the
pilot described below, an open-weights instruction model was asked to score 865 insider-trading
windows on three zero-to-hundred dimensions, five independent times per filing, under a fixed
prompt, a fixed model version and a fixed seed — conditions as favourable to stability as a
deployment could realistically be. Not a single one of the 4,325 responses failed to parse, so
what follows is not a formatting artifact. The median standard deviation across the five re-runs
of the *same* prompt on the *same* filing was 59 percent of the standard deviation across
*different* filings on the volatility-risk dimension, 88 percent on information asymmetry, and 96
percent on the model's own stated confidence. On the last of these the elicited score is very
nearly pure noise: re-asking the identical question about one company moves the answer as much as
switching to a different company. Averaging the five draws recovers almost none of it, lifting the
information coefficient from 0.733 to 0.736. A feature must be a fixed measurement; a quantity
whose re-measurement error approaches its cross-sectional signal is not one.

The literature explains why this should be expected and shows that it is worse along axes the
pilot held fixed. Sclar et al. (2024) find open-source
models "extremely sensitive to subtle changes in prompt formatting", with performance differences
"of up to 76 accuracy points" on LLaMA-2-13B under changes that are explicitly *meaning-preserving*
— the same question, typeset differently. Ouyang et al. (2024) find repeated identical requests
disagreeing on 47.6–75.8% of tasks and report that "setting the temperature to 0 does not guarantee
determinism" (their domain is code generation, so this establishes the decoding mechanism rather
than a magnitude for a scoring task). Wang et al. (2023) show LLM scorers being flipped by
reordering the candidates, in a pairwise-comparison setting rather than the absolute scoring
proposed here. Across releases, Chen, Zaharia & Zou (2023) record GPT-4 falling from 84% to 51%
accuracy on an unchanged task in three months. The pilot measured dispersion holding prompt, model
version and decoding configuration constant; each of these results says that relaxing any one of
those would make it worse.

**Contamination is the binding constraint.** This is the stronger objection, because it attacks the
backtest rather than the tooling. Glasserman & Lin (2023) show that when a model's training window
overlaps the backtest period the result is biased through *two* channels: look-ahead bias, where the
model knows the returns that followed, and a distraction effect, where general knowledge of the
named company interferes with reading the text. The important point is counter-intuitive and worth
stating precisely, because the naive version of this argument is wrong: the two channels run in
*opposite* directions. In-sample they find that *anonymised* headlines outperform, "indicating that
the distraction effect has a greater impact than look-ahead bias", and that this is "particularly
strong for larger companies". The net bias therefore cannot be signed in advance on an S&P 500
panel, which is precisely the large-firm regime where they report distraction dominating — and an
unsignable bias cannot be bounded or corrected for, only avoided. The obvious mitigation, prompting
the model to answer as of the filing date, is not dependable either: Wongchamcharoen & Glasserman
(2025) show that such defences "implicitly assume that models understand chronology" and that models
"struggle to maintain a single globally consistent timeline", on GPT-4.1, Claude-3.7 Sonnet and
GPT-5 alike. Kong et al. (2026), reviewing 164 financial-LLM papers from 2023–2025, find no single
one of these biases discussed in more than 28% of them.

**Why this study is entitled to weight that second obstacle heavily.** Part III.2–III.3 measured the
price of lookahead on this exact task: roughly 0.06 IC of apparent skill for an encoder trained on
all pre-2025 labels and epoch-selected on the evaluation year, several times the size of the genuine
text increment (+0.011 to +0.016). The encoder case was *fixable* — retrain under a strict pre-2017
cutoff, freeze, and the apparent skill disappears, which is how the 0.06 figure was obtained. The
generative case is not fixable the same way, because retraining the language model on a rolling
window is computationally infeasible, a point Glasserman & Lin make directly. A route whose central
validity threat this study can measure but cannot remove is the right one to decline.

**The bounded pilot, and what it settles.** Declining a direction on argument alone invites the
reply that the argument is a rationalisation, so the cheapest decisive version of the experiment
was run before the position was fixed, with its stopping rule written down before any score was
looked at. Scope was chosen so that the contamination objection could not apply to the pilot
itself: only 2024 and 2025 filings were scored, both post-dating the model's training cutoff, and
the prompts were identifier-stripped in the manner Glasserman & Lin propose — no ticker, no
company or insider names, no absolute dates, only day offsets relative to the filing and
pseudonymised insider labels. The model read the same pre-filing transaction window that the eight
hand-crafted Form 4 features summarise, and its three scores were then compared against those
features under one head and one set of folds, in the same five-fold within-year design used for
the 2025 call gate and for the same reason: a score that exists only on the evaluation rows cannot
be learned under a train-before-year split.

The comparison is a null in every direction that matters. Against the structured baseline
(information coefficient 0.736), the eight hand-crafted features give 0.730 and the model's scores
give 0.736. The difference between the two feature blocks is +0.006 with a paired bootstrap
interval of [−0.000, +0.013], inside the half-width its own stopping rule set, so the pilot closes
as a measured null. The finding is not that the language model is bad at reading Form 4 records; it
is that feature engineering was never the bottleneck. A hand-built summary of the insider record
and a generative model's own reading of the raw record are statistically indistinguishable from
each other, and both are indistinguishable from omitting insider information altogether — which is
exactly what the level test of Part VI already found over seven and twelve backtest years. What
binds is the information content of the record relative to a structured block that already carries
realised volatility, drawdown and liquidity.

One further reading deserves care, because it is easy to overclaim. Scoring the identified prompts
as a control produced an information coefficient of 0.737 against the anonymised 0.736, a premium
of +0.001 with an interval of [−0.004, +0.006]. On a post-cutoff slice there should be no
look-ahead to find, so this is a placebo check that passed — evidence that the scoping decision
worked, not a measurement showing contamination to be small. The years where the objection actually
bites are 2018 to 2024, and by construction the pilot says nothing about them.

A credible treatment would therefore require principled prompt-schema design, calibration and
stability analysis across model versions, an identifier-anonymisation control of the kind
Glasserman & Lin propose, explicit separation of contamination from signal, and inference over the
full filing corpus for at least two model families under the admissibility discipline of Part
III.2. That is a study in its own right, and it is identified here as a direction for future
research rather than as an extension of this one. What this study contributes to it is a negative
result obtained cheaply and a stability measurement obtained on real financial records.

---

## Appendix — Sources and citation verification

*Sources for verification: `phase5/STRESS_TEST_RESULTS.md` (all stress-test tables, including the
volatility-window spectrum, the Form 4 versus calls comparison and the Round 8 pilot),
`phase5/out/stress_grid_*.json` (machine-readable results), `original_pipeline_details.md`
(original pipeline). The horizon spectrum of Part III.4 and the sweeps it feeds in Parts V and VI
come from `dataset_config/compute_horizon_labels.py`, `phase5/run_horizon.py`,
`phase5/form4_text_conditioning.py` and `phase5/call_combined_gate.py` under the `RISK_HORIZON` and
`RISK_CALL_WINDOW` switches (job 3584156, `logs/run_horizon_spectrum.log`); the generative pilot of
VII.5 from `dataset_config/build_form4_llm_prompts.py`, `phase5/llm_score_form4.py` and
`phase5/llm_vs_manual.py` on branch `exp/llm-event-scoring` (job 3584249,
`logs/run_llm_form4.log`), whose stopping rule is stated in the docstring of the last of these and
was fixed before any score was read.
`Literature_agent_llm_feasibility.md` (Round 8, 2026-08-04 — the seven
generative-LLM sources behind VII.5, each downloaded from arXiv and checked by text extraction
against the claim it supports, since the `Literature/` PDF directories are not on the cluster;
it records one material correction, that Glasserman & Lin's in-sample finding runs opposite to
the intuitive reading of look-ahead bias, two domain caveats on Ouyang et al. and Wang et al.,
and one candidate rejected for being downloaded but not verified in detail),
`Literature_agent_study_extended.md` (citation provenance; the verification
log at the end of that file records a full four-round PDF pass completed on 2026-07-07, in which
every content, numeric and bibliographic claim was checked against its source PDF, and all
corrections are already reflected in this document. The load-bearing ones: the long-document
windowing precedent is Beltagy et al. 2020 Section 2 with Yang et al. 2020 and Chiu et al. 2025
(SBERT and BERT dropped from that claim); FinMTEB contains no temporal-shift analysis and is cited
only for its bag-of-words-versus-dense finding; the Corsi 2009 R² figures are the specific Table 4
values rather than a blanket range; the 30-trading-day label window is stated as this study's own
design choice rather than as a convention inherited from that literature, since Corsi's monthly
component is 22 trading days and no source supports 30 as "monthly" — the Round 7 correction, now
applied in Part 0.1 and Part III.4; the Lazy-Prices alpha is attributed to investor under-reaction
rather than to filings changing little; the Grinold and Kahn benchmark is the concrete IC-to-
information-ratio result; and the Lopez de Prado references are chapters 7, 11 and 12. Two
bibliographic housekeeping items remain open in the citations file, neither a content error: the
Grinold and Kahn edition year and the Diebold and Mariano 1995-versus-2002 reprint choice.)
Momentum is deliberately cited through Ang et al. (2006), where it
appears as a standard control variable, rather than through Jegadeesh and Titman (1993), which
could not be accessed for reading; since momentum enters this study only as a control feature,
the accessible precedent is the appropriate one. Add Jegadeesh and Titman as a secondary
origin citation only if a copy is obtained.*
