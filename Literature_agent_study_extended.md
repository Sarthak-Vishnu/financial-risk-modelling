# Literature Citations for Dissertation Design Choices
**Date:** 2026-07-06  
**Context:** Predicting 30-day forward realised stock volatility from SEC 10-K Item 1A text combined
with structured financial features. This file maps each design choice to grounded citations.

**Provenance key:**
- ✅ **IN DIRS** — PDF confirmed in `Literature/` or `Literature (new)/`
- ⚠️ **CANONICAL** — well-established, not in dirs; safe to cite as book/knowledge reference
- ❌ **FLAG** — no strong citation available; noted explicitly

---

## 1. Structured Volatility Predictors

### (a) Volatility Persistence — Multi-Horizon Realised Volatility

> ✅ **Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility."
> *Journal of Financial Econometrics*, 7(2), 174–196.**
> `Literature (new)/`

The HAR-RV model uses realised volatility at 1-day, 5-day, and 22-day horizons as predictors of
future realised volatility, achieving R² of 40–70% at daily horizons. The standard benchmark model
for realised volatility forecasting, and the direct precedent for the `rv_21d / rv_63d / rv_126d /
rv_252d` feature block.

### (b) Firm Size and (c) Leverage

> ✅ **Ang, A., Hodrick, R.J., Xing, Y. & Zhang, X. (2006). "The Cross-Section of Volatility
> and Expected Returns." *Journal of Finance*, 61(1), 259–299.**
> `Literature (new)/`

Ang et al. (2006) include firm size, B/M, and leverage as control variables in all their
cross-sectional regressions of IVOL and returns — establishing them as standard covariates in this
exact type of analysis. Using this paper covers (b) and (c) without requiring the older
Christie (1982) or Schwert (1989).

### (d) Book-to-Market

> ✅ **Fama, E.F. & French, K.R. (1992). "The Cross-Section of Expected Stock Returns."
> *Journal of Finance*, 47(2), 427–465.**
> `Literature (new)/`

Foundational paper establishing B/M and size as cross-sectional risk factors. Every finance referee
expects this citation for B/M. No modern substitute.

### (e) Market Beta and Idiosyncratic Volatility

> ✅ **Ang, A., Hodrick, R.J., Xing, Y. & Zhang, X. (2006)** (full citation above, §1b)

The canonical paper for both market beta and idiosyncratic volatility (IVOL) as cross-sectional
signals. They decompose total volatility into market and idiosyncratic components and show IVOL
strongly predicts future returns — the direct precedent for `market_beta` and `idio_vol` features.
One paper covers (b), (c), (d), and (e) as a package.

### (f) Amihud Illiquidity

> ✅ **Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects."
> *Journal of Financial Markets*, 5(1), 31–56.**
> `Literature (new)/`

Defines the Amihud ratio `|R|/volume` and validates it as a predictor of returns cross-sectionally
and through time. Obligatory citation for the `amihud_illiquidity` feature — it is the definitional
paper for the measure being used.

### (g) Return Distribution Moments and Momentum

**Skewness and kurtosis as cross-sectional predictors:**
> ✅ **Conrad, J., Dittmar, R.F. & Ghysels, E. (2013). "Ex Ante Skewness and Expected Stock
> Returns." *Journal of Finance*, 68(1), 85–124.**
> `Literature (new)/`

Uses option-implied moments to show that ex ante skewness and kurtosis are priced cross-sectionally,
with negatively skewed stocks earning higher expected returns and volatility. More recent and
methodologically tighter than Harvey & Siddique (2000), which it supersedes.

**Momentum:**

Jegadeesh & Titman (1993) was not accessible. **Use Ang et al. (2006) instead.** Their Table 2
includes past-6-month return (momentum) as a standard control variable alongside size, B/M, and
beta in all cross-sectional regressions — establishing that momentum belongs in the same covariate
set. Since Ang et al. (2006) is already being cited for (b)–(e), this adds no new citation and
momentum is covered without needing JT (1993) as a separate reference.

> ❌ **FLAG — Kurtosis as a standalone predictor:** No single canonical paper establishes kurtosis
> as a first-order volatility predictor independent of skewness. Include it as an exploratory
> control but do not build a separate citation claim for it.

---

## 2. Text-Based 10-K Volatility Prediction

### Foundational Paper — TF-IDF Baseline

> ✅ **Kogan, S., Levin, D., Routledge, B.R., Sagi, J.S. & Smith, N.A. (2009). "Predicting Risk
> from Financial Reports with Regression." *Proceedings of NAACL-HLT 2009*, pp. 272–280.**
> `Literature (new)/`

Uses 10-K filings from 1996–2005 to predict log excess return variance in the 12-month post-filing
window using SVR on TF-IDF bag-of-words features, achieving R² of 0.30–0.40 from text alone.
**The paper that establishes TF-IDF + linear model as the standard strong baseline for this task.**

### Finance-Specific Word List

> ✅ **Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability? Textual Analysis,
> Dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35–65.**
> `Literature/[2011] When Is a Liability Not a Liability...`

Finance-specific negative word counts from 10-K filings predict returns, earnings, and volatility
more accurately than the Harvard-IV dictionary. Standard justification for treating TF-IDF as a
*strong* lexical baseline calibrated to financial language, not a naive one. Cite alongside
Kogan et al. (2009) whenever describing the TF-IDF baseline.

### Item 1A → Volatility (Empirical Foundation)

> ✅ **Campbell, J.L., Chen, H., Dhaliwal, D.S., Lu, H. & Steele, L.B. (2014). "The Information
> Content of Mandatory Risk Factor Disclosures in Corporate Filings." *Review of Accounting
> Studies*, 19(1), 396–455. DOI: 10.1007/s11142-013-9258-3**
> `Literature (new)/[2014] The Information Content of Mandatory Risk Factor Disclosures...`

Shows directly that Item 1A disclosures are reflected in post-disclosure idiosyncratic risk (stock
return volatility) and systematic risk (market beta). Key finding: the unexpected portion of risk
disclosures is positively associated with post-disclosure volatility. The cleanest empirical anchor
for why Item 1A text predicts volatility — the disclosures are informative, not boilerplate.

### Item 1A Sentiment → Returns (Nearest Prior Work)

> ✅ **Magner, N., Henríquez, C. & Sanhueza, J. (2025). "Decoding Risk Sentiment in 10-K Filings:
> Predictability for U.S. Stock Indices." *Finance Research Letters*, 81, Article 107472.
> DOI: 10.1016/j.frl.2025.107472**
> `Literature (new)/[2025] Decoding Risk Sentiment in 10-K Filings...`

LM-dictionary and GPT-based sentiment on Item 1A across 21,421 10-K reports (2002–2024) predicts
index returns. Positions the domain-adapted encoder approach as a methodological advance over
dictionary/GPT methods on the same section and task. Closest prior work to this dissertation.

**Summary:** Cite Kogan et al. (2009) + Loughran & McDonald (2011) together to establish TF-IDF +
ridge as the standard strong lexical baseline. Campbell et al. (2014) establishes why Item 1A is
the correct text source. Magner et al. (2025) is the nearest prior work to position against.

---

## 3. Disclosure Changes

### Primary Citation — Lazy Prices

> ✅ **Cohen, L., Malloy, C. & Nguyen, Q. (2020). "Lazy Prices."
> *Journal of Finance*, 75(3), 1371–1415. DOI: 10.1111/jofi.12885**
> `Literature (new)/[2020] Lazy Prices.pdf`

Measures year-over-year changes in 10-K text using cosine similarity and edit distance across all
10-K and 10-Q filings (1995–2017). Firms with large changes underperform non-changers by 188
bps/month (>22% annualised). Anchors two claims: (1) 10-K text changes are informationally
meaningful; (2) year-over-year Item 1A cosine similarity is a defensible scalar feature. The alpha
arises because most filings change very little — the empirical documentation of Item 1A stickiness.

### Item 1A Stickiness — Rebuttal of Boilerplate Criticism

> ✅ **Campbell et al. (2014)** (full citation above, §2)

Their introduction quotes Reuters analysts calling risk factors "boilerplate" and empirically
rejects this — showing disclosures meaningfully reflect firm-specific risk types. Sharpest available
rebuttal of the boilerplate criticism from within the literature dirs.

### Successor — Changes in Risk Language → Volatility

> ⚠️ **Kravet, T. & Muslu, V. (2013). "Textual Risk Disclosures and Investors' Risk Perceptions."
> *Review of Accounting Studies*, 18(4), 1088–1122.**

Studies changes in risk words in 10-Ks specifically; finds increases in risk-word frequency predict
higher subsequent return volatility. Not in dirs — use only if a specific risk-change citation is
needed beyond Lazy Prices.

---

## 4. Evaluation Methodology

### (a) Expanding-Window / Walk-Forward Backtesting

> ⚠️ **Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
> (Chapters 4 and 7.)**

Standard reference for why k-fold CV is inappropriate for financial time series, and for
walk-forward / combinatorial purged CV protocols. Cite as a book reference — no PDF required.

### (b) Diebold-Mariano Test

> ✅ **Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
> *Journal of Business & Economic Statistics*, 13(3), 253–263.**
> `Literature (new)/`

The original DM test. Old but obligatory — every forecast comparison paper cites this. The
finite-sample correction (Harvey, Leybourne & Newbold 1997) is a minor refinement; citing DM
(1995) alone is standard practice in finance ML papers.

### (c) Lookahead Bias / Temporal Leakage

> ⚠️ **Lopez de Prado, M. (2018)** (same reference as 4a — Chapters 4 and 7.)

The recognised reference for temporal leakage in financial ML. Cite as book reference alongside
a brief procedural description of how the expanding-window protocol was implemented.

### (d) IC Magnitudes and Volatility Predictability

**IC benchmark context (book reference — no PDF required):**
> ⚠️ **Grinold, R.C. & Kahn, R.N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.**

Grinold & Kahn establish that IC of 0.05–0.10 is considered good for cross-sectional *return*
prediction. Cite as a book reference to establish that volatility ICs of 0.50–0.61 are far higher
than the return-prediction benchmark — as expected given volatility's well-documented persistence.

**Why volatility ICs of 0.5+ are not surprising — use Corsi (2009):**
> ✅ **Corsi (2009)** (full citation §1a)

Corsi's HAR-RV achieves R² of 40–70% for realised volatility forecasting using only past
volatility. This directly demonstrates that volatility is a far more forecastable quantity than
returns (which have R² typically below 2%), providing the theoretical context for why within-year
Spearman ICs of 0.5–0.6 are plausible for volatility ranking. No additional paper is needed for
this point.

---

## 5. Long-Document Encoding

### Financial-NLP Precedent — Hierarchical Encoding

> ✅ **Yang, Z., Ng, K., Smyth, B. & Dong, R. (2020). "HTML: Hierarchical Transformer-based
> Multi-task Learning for Volatility Prediction." *The Web Conference (WWW 2020)*, pp. 441–451.**
> `Literature (new)/[2020] HTML Hierarchical Transformer-based Multi-task Learning...`

Encodes long earnings call transcripts via a two-level architecture: token-level BERT on each
sentence (≤512 tokens), then a sentence-level Transformer aggregates sentence embeddings. Direct
financial-NLP precedent for encoding long documents by processing fixed-length segments and
pooling. The flat sliding-window + mean-pool approach used here is the simplified variant.

### General NLP Precedent — SBERT

> ✅ **Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese
> BERT-Networks." *EMNLP-IJCNLP 2019*.**
> `Literature/[2019; EMNLP-IJCNLP] Sentence-BERT...`

Section 4.2 explicitly describes the standard approach for documents longer than the BERT context
window: split into overlapping segments, encode each, mean-pool the representations. Canonical
general-NLP precedent for flat chunking + mean-pooling.

### Paragraph-Level Encoding in Financial Filings

> ✅ **Chiu, Y.-H. et al. (2025). "Financial Risk Relation Identification through Dual-view
> Adaptation." *EMNLP 2025*.**
> `Literature/[2025; EMNLP] Financial Risk Relation Identification through Dual-view Adaptation.pdf`

Uses 256-token paragraphs as the encoding unit throughout, corroborating the paragraph-as-unit
principle and the convention of truncating/padding to a fixed token budget.

**Note:** No single paper canonically defines sliding-window + mean-pooling specifically for 10-K
filings. Cite SBERT (general NLP precedent) + HTML (financial application) together — sufficient.

---

## 6. Fair Ablation Design

> ❌ **FLAG — no strong dedicated paper for "hold the prediction head fixed when ablating feature
> sets." This is methodological best practice, not a published protocol.**

Describe the ablation design explicitly in the methods section (HGB/Ridge head fixed; only the
input feature set varies across conditions). For any referee who asks:

> ⚠️ **Lipton, Z.C. & Steinhardt, J. (2019). "Troubling Trends in Machine Learning Scholarship."
> *ACM Queue*, 17(1).**

Argues that model comparisons typically conflate architectural and representational differences and
calls for controlling all factors except the one under study. Not in dirs; use only if a citation
is explicitly requested by examiners.

For small IC differences being within noise: reference the IC benchmark discussion in §4d
(Grinold & Kahn 2000) — IC differences of ±0.002 are below any meaningful threshold.

---

## 7. Supervised / Task-Aligned Text Encoders in Finance

### Established Precedent for Task-Supervised Encoders

> ✅ **Qin, Y. & Yang, Y. (2019). "What You Say and How You Say It Matters: Predicting Financial
> Risk Using Verbal and Vocal Cues." *ACL 2019*, pp. 390–401.**
> `Literature (new)/[2019] What You Say and How You Say It Matters...`

Trains a BiLSTM encoder end-to-end on earnings call → volatility prediction. The earliest paper
in the corpus that fine-tunes a text encoder directly on a volatility target, establishing the
task-supervised approach as a recognised method with positive results.

> ✅ **Yang et al. (2020) HTML** and **Sawhney et al. (2020) VolTAGE**
> (full citations in §5 and `Literature (new)/`)

Both fine-tune BERT / FinBERT directly on the earnings call → volatility task. Together with Qin
(2019) they confirm task-supervised fine-tuning as current SOTA prior to this work.

### Temporal Non-Stationarity / Out-of-Period Degradation

> ✅ **Magner et al. (2025)** (full citation §2)

Uses TVP-VAR to show the predictive relationship between Item 1A tone and returns varies across
time regimes, justifying why a model trained on one period may not transfer cleanly to another.

> ✅ **Su, Y. et al. (2025). "FinMTEB: Finance Massive Text Embedding Benchmark." *EMNLP 2025*.**
> `Literature/[2025; EMNLP] FinMTEB Finance Massive Text Embedding Benchmark.pdf`

Evaluates sentence encoders across time periods and documents temporal distribution shift as a
systematic challenge in financial NLP, including for task-specific fine-tuned models.

**For the general concept drift framing:**
> ✅ **Lu, J., Liu, A., Dong, F. & Gama, J. (2019). "Learning under Concept Drift: A Review."
> *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346–2363.**
> `Literature (new)/`

Peer-reviewed survey formally defining concept drift as a change in the joint distribution
P(X, Y) over time, and cataloguing mitigation strategies. More citable than the often-cited
Tsymbal (2004) tech report it supersedes.

**Recommended framing:** Task-supervised fine-tuning (Lever 3, IC 0.456) overfit to the training
regime's distributional properties — consistent with the concept drift framework (Lu et al. 2019)
and supported by the time-varying predictive relationships documented in Magner et al. (2025).
Similarity-trained encoders learned a more regime-invariant geometry and therefore transfer better
across years.

---

## Quick Reference Table

| # | Claim | Paper | Venue / Year | In Dirs? |
|---|---|---|---|---|
| 1a | Volatility persistence / HAR-RV | Corsi (2009) | J. Financial Econometrics | ✅ Yes |
| 1b–e | Size, leverage, B/M, beta, IVOL | Ang, Hodrick, Xing & Zhang (2006) | JF | ✅ Yes |
| 1d | Book-to-market as risk factor | Fama & French (1992) | JF | ✅ Yes |
| 1f | Amihud illiquidity | Amihud (2002) | J. Financial Markets | ✅ Yes |
| 1g | Skewness / kurtosis cross-section | Conrad, Dittmar & Ghysels (2013) | JF | ✅ Yes |
| 1g | Momentum (control variable) | Ang et al. (2006) Table 2 | JF | ✅ Yes |
| 2 | TF-IDF → 10-K volatility (baseline) | Kogan et al. (2009) | NAACL-HLT | ✅ Yes |
| 2 | Finance word list → volatility/returns | Loughran & McDonald (2011) | JF | ✅ Yes |
| 2 | Item 1A → idiosyncratic volatility | Campbell et al. (2014) | Rev. Account. Stud. | ✅ Yes |
| 2 | Item 1A sentiment → returns | Magner et al. (2025) | Finance Res. Lett. | ✅ Yes |
| 3 | 10-K text changes → returns / alpha | Cohen, Malloy & Nguyen (2020) | JF | ✅ Yes |
| 3 | Item 1A stickiness rebuttal | Campbell et al. (2014) | Rev. Account. Stud. | ✅ Yes |
| 3 | Risk word changes → volatility | Kravet & Muslu (2013) | Rev. Account. Stud. | ⚠️ No |
| 4a | Walk-forward backtesting protocol | Lopez de Prado (2018) | Wiley (book) | ⚠️ Book |
| 4b | Forecast comparison test | Diebold & Mariano (1995) | JBES | ✅ Yes |
| 4c | Lookahead bias / leakage | Lopez de Prado (2018) | Wiley (book) | ⚠️ Book |
| 4d | IC magnitude benchmarks | Grinold & Kahn (2000) | McGraw-Hill (book) | ⚠️ Book |
| 4d | Volatility forecastability (R² 40–70%) | Corsi (2009) | J. Financial Econometrics | ✅ Yes |
| 5 | Hierarchical chunk + pool (financial) | Yang et al. HTML (2020) | WWW 2020 | ✅ Yes |
| 5 | Sliding-window chunk + mean-pool (NLP) | Reimers & Gurevych SBERT (2019) | EMNLP-IJCNLP | ✅ Yes |
| 5 | Paragraph-level encoding (financial) | Chiu et al. (2025) | EMNLP | ✅ Yes |
| 6 | Fair ablation / confounded comparisons | Lipton & Steinhardt (2019) | ACM Queue | ⚠️ No |
| 7 | Task-supervised encoder → volatility | Qin & Yang (2019) | ACL | ✅ Yes |
| 7 | Task-supervised encoder → volatility | Yang et al. HTML (2020) | WWW 2020 | ✅ Yes |
| 7 | Temporal distribution shift in fin. NLP | Magner et al. (2025) | Finance Res. Lett. | ✅ Yes |
| 7 | Temporal distribution shift in fin. NLP | FinMTEB Su et al. (2025) | EMNLP | ✅ Yes |
| 7 | Concept drift definition / survey | Lu, Liu, Dong & Gama (2019) | IEEE TKDE | ✅ Yes |

---

**Status summary:** 21 of 24 citations are now in `Literature/` or `Literature (new)/`.
Three remaining without a PDF are books (Lopez de Prado 2018, Grinold & Kahn 2000) or a low-priority
optional entry (Kravet & Muslu 2013, Lipton & Steinhardt 2019) — all citable without a local PDF.

*Last updated: 2026-07-06.*
