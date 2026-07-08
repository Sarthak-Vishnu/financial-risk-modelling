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

The HAR-RV model uses realised volatility at daily, weekly (5-day average), and monthly horizons
as predictors of future realised volatility. **PDF-verified (Round 3):** the monthly component is
22 working days — *"the HAR model considered here employs monthly realized volatility (which
corresponds to 22 working days)"* (§3.2, p. 186). So the file's "1-day / 5-day / 22-day" horizons
are correct. **Correction:** the "R² of 40–70%" figure is only loosely supported — Corsi's actual
in-sample one-day-ahead HAR(3) R² values are 0.565 (USD/CHF), 0.707 (S&P500 futures) and 0.236
(T-Bond futures) (Table 4, p. 189); out-of-sample they are lower (1-day USD/CHF R² = 0.264,
Table 5, p. 191). The "40–70%" range holds for the equity/FX series in-sample but not for T-Bond
and not out-of-sample; cite the specific values rather than a blanket "40–70%". The standard
benchmark model for realised-volatility forecasting, and the direct precedent for the
`rv_21d / rv_63d / rv_126d / rv_252d` feature block.

### (b) Firm Size and (c) Leverage

> ✅ **Ang, A., Hodrick, R.J., Xing, Y. & Zhang, X. (2006). "The Cross-Section of Volatility
> and Expected Returns." *Journal of Finance*, 61(1), 259–299.**
> `Literature (new)/`

Ang et al. (2006) control for firm size, book-to-market, and leverage in their cross-sectional
(portfolio double-sort) analysis of idiosyncratic volatility and returns — establishing them as
standard covariates in this exact type of analysis. **PDF-verified (Round 3):** these controls
appear in **Table VII**, *"Controlling for Various Cross-Sectional Effects"* (§I.C, p. 279):
*"we control for size, book-to-market, leverage, liquidity, volume, ... and momentum"*. (The
method is portfolio double-sorts, not Fama–MacBeth regressions.) Using this paper covers (b) and
(c) without requiring the older Christie (1982) or Schwert (1989).

### (d) Book-to-Market

> ✅ **Fama, E.F. & French, K.R. (1992). "The Cross-Section of Expected Stock Returns."
> *Journal of Finance*, 47(2), 427–465.**
> `Literature (new)/`

Foundational paper establishing B/M and size as cross-sectional risk factors. Every finance referee
expects this citation for B/M. No modern substitute.

### (e) Market Beta and Idiosyncratic Volatility

> ✅ **Ang, A., Hodrick, R.J., Xing, Y. & Zhang, X. (2006)** (full citation above, §1b)

The canonical paper for both market beta and idiosyncratic volatility (IVOL) as cross-sectional
signals. IVOL is measured as the standard deviation of residuals relative to the Fama–French
3-factor model. **Correction (Round 3) — direction:** the paper shows IVOL is *strongly and
negatively* related to future returns, not merely that it "predicts" them — *"stocks with high
idiosyncratic volatility relative to the Fama and French (1993) ... model have abysmally low
average returns"* (abstract, p. 259; the "idiosyncratic volatility puzzle"). It remains the direct
precedent for the `market_beta` and `idio_vol` features. One paper covers (b), (c), (d), and (e)
as a package.

### (f) Amihud Illiquidity

> ✅ **Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects."
> *Journal of Financial Markets*, 5(1), 31–56.**
> `Literature (new)/`

Defines the Amihud ratio and validates it as a predictor of returns cross-sectionally and through
time. **PDF-verified (Round 3):** the measure is absolute return over *dollar* volume, not raw
share volume — *"ILLIQ ... is the daily ratio of absolute stock return to its dollar volume,
averaged over some period"* (abstract / §2, p. 32). Write it as `|R|/dollar-volume`. Obligatory
citation for the `amihud_illiquidity` feature — it is the definitional paper for the measure
being used.

### (g) Return Distribution Moments and Momentum

**Skewness and kurtosis as cross-sectional predictors:**
> ✅ **Conrad, J., Dittmar, R.F. & Ghysels, E. (2013). "Ex Ante Skewness and Expected Stock
> Returns." *Journal of Finance*, 68(1), 85–124.**
> `Literature (new)/`

Uses option prices to estimate ex ante (risk-neutral) higher moments and shows that skewness and
kurtosis are priced cross-sectionally. **PDF-verified (Round 3), with a correction:** the paper's
findings are about *returns*, not volatility — *"more ex ante negatively (positively) skewed
returns yield subsequent higher (lower) returns ... a negative (positive) relation between ex ante
volatility (kurtosis) and subsequent returns"* (abstract, p. 85). So: negatively skewed stocks
earn higher subsequent returns (correct), but the phrase "and volatility" was an overreach and is
removed — Conrad et al. do not show negatively skewed stocks have higher volatility. More recent
and methodologically tighter than Harvey & Siddique (2000), which it supersedes.

**Momentum:**

Jegadeesh & Titman (1993) was not accessible. **Use Ang et al. (2006) instead.** **Corrected
(Round 3):** momentum is controlled for in **Table VII** (not "Table 2" — Table II is *"Factor
Correlations"*, an unrelated table), and the horizon is **past-12-month returns**, not 6-month
(*"we control for ... past 12-month returns"*, p. 279 / p. 273). It sits alongside size, B/M,
leverage, liquidity, and volume in the same double-sort robustness analysis — establishing that
momentum belongs in the same covariate set. Since Ang et al. (2006) is already being cited for
(b)–(e), this adds no new citation and momentum is covered without needing JT (1993) as a
separate reference.

> ❌ **FLAG — Kurtosis as a standalone predictor:** No single canonical paper establishes kurtosis
> as a first-order volatility predictor independent of skewness. Include it as an exploratory
> control but do not build a separate citation claim for it.

---

## 2. Text-Based 10-K Volatility Prediction

### Foundational Paper — TF-IDF Baseline

> ✅ **Kogan, S., Levin, D., Routledge, B.R., Sagi, J.S. & Smith, N.A. (2009). "Predicting Risk
> from Financial Reports with Regression." *Proceedings of NAACL-HLT 2009*, pp. 272–280.**
> `Literature (new)/`

Uses 10-K filings to predict stock-return volatility over the 12 months following the report, using
support-vector regression on bag-of-words features. **PDF-verified (Round 3), with three
corrections:**
(1) **Sample:** the corpus is 1996–**2006** (*"54,379 reports published over the period 1996–2006
from 10,492 different companies"*, §3), not 1996–2005.
(2) **Target:** the predicted quantity is log *volatility* — the standard deviation of returns
(*"stock return volatility ... measured as the standard deviation of a stock's returns"*, §2; they
*"predict log v instead of v"*, §3) — not "log excess return variance". It is standard deviation,
predicted in log form, over the 12 months after the filing.
(3) **Metric:** the paper reports mean-squared error of predicted log-volatility, **not R²**;
there is no "R² of 0.30–0.40" anywhere in it (that figure was unverified and is removed).
Features include TF, TFIDF, and LOG1P representations (TFIDF is one of them, so "TF-IDF baseline"
is fair). **The paper that establishes bag-of-words + linear regression as the standard strong
baseline for this task.**

### Finance-Specific Word List

> ✅ **Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability? Textual Analysis,
> Dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35–65.**
> `Literature/[2011] When Is a Liability Not a Liability...`

Finance-specific negative word counts from 10-K filings predict returns, earnings, and volatility
more accurately than the Harvard-IV dictionary. **PDF-verified (Round 3):** the Harvard-IV
mismatch — *"almost three-fourths of the words identified as negative by the widely used Harvard
Dictionary are words typically not considered negative in financial contexts"* (abstract) — and the
volatility link is explicit — *"We link the word lists to 10-K filing returns, trading volume,
return volatility, fraud, material weakness, and unexpected earnings"* (abstract; results §III).
Standard justification for treating TF-IDF as a *strong* lexical baseline calibrated to financial
language, not a naive one. Cite alongside Kogan et al. (2009) whenever describing the TF-IDF
baseline.

### Item 1A → Volatility (Empirical Foundation)

> ✅ **Campbell, J.L., Chen, H., Dhaliwal, D.S., Lu, H. & Steele, L.B. (2014). "The Information
> Content of Mandatory Risk Factor Disclosures in Corporate Filings." *Review of Accounting
> Studies*, 19(1), 396–455. DOI: 10.1007/s11142-013-9258-3**
> `Literature (new)/[2014] The Information Content of Mandatory Risk Factor Disclosures...`

Shows directly that Item 1A disclosures are reflected in post-disclosure idiosyncratic risk (stock
return volatility) and systematic risk (market beta). **PDF-verified (Round 3):** *"We find a
positive association between the unexpected portion of risk factor disclosures and the
post-disclosure level of market beta and stock return volatility"* (p. 398); *"the information
conveyed by risk factor disclosures is reflected in systematic risk, idiosyncratic risk,
information asymmetry, and firm value"* (abstract, p. 396). The cleanest empirical anchor for why
Item 1A text predicts volatility — the disclosures are informative, not boilerplate.

### Item 1A Sentiment → Returns (Nearest Prior Work)

> ✅ **Magner, N., Henríquez, C. & Sanhueza, J. (2025). "Decoding Risk Sentiment in 10-K Filings:
> Predictability for U.S. Stock Indices." *Finance Research Letters*, 81, Article 107472.
> DOI: 10.1016/j.frl.2025.107472**
> `Literature (new)/[2025] Decoding Risk Sentiment in 10-K Filings...`

LM-dictionary and GPT-based sentiment on Item 1A across 21,421 10-K reports (2002–2024) predicts
index returns. **PDF-verified (Round 3):** *"We analyzed the tone of 21,421 10-K reports from
publicly traded U.S. companies between 2002 and 2024"* (§2.1); five tone indicators built with
*"the Loughran-McDonald dictionary, and AI-calibrated alternatives (GPT-3.5-turbo-0125, GPT-4,
GPT-4o, and GPT-4o-mini)"* (abstract); target is weekly returns on the S&P 500, Nasdaq, Russell
2000, and Dow Jones. Positions the domain-adapted encoder approach as a methodological advance
over dictionary/GPT methods on the same section and task. Closest prior work to this dissertation.

**Summary:** Cite Kogan et al. (2009) + Loughran & McDonald (2011) together to establish TF-IDF +
ridge as the standard strong lexical baseline. Campbell et al. (2014) establishes why Item 1A is
the correct text source. Magner et al. (2025) is the nearest prior work to position against.

---

## 3. Disclosure Changes

### Primary Citation — Lazy Prices

> ✅ **Cohen, L., Malloy, C. & Nguyen, Q. (2020). "Lazy Prices."
> *Journal of Finance*, 75(3), 1371–1415. DOI: 10.1111/jofi.12885**
> `Literature (new)/[2020] Lazy Prices.pdf`

Measures year-over-year changes in 10-K/10-Q text using cosine similarity and edit distance.
**PDF-verified (Round 3), with two corrections:**
(1) **Sample period is 1995–2014, not 1995–2017** — *"systematic across the entire cross-section
of U.S. publicly traded firms from 1995 to 2014"* (p. 1377). (The "2017" in the earlier draft was
a citation to Loughran & McDonald 2017, not the sample end.) Similarity measures verified:
*"(i) cosine similarity, (ii) Jaccard similarity, (iii) minimum edit distance, and (iv) simple
similarity"* (§I).
(2) **Alpha mechanism:** *"A portfolio that shorts 'changers' and buys 'nonchangers' earns up to
188 basis points per month in alpha (over 22% per year)"* (abstract, p. 1371) — this 188 bps
figure is correct. But the alpha arises from investor **inattention / under-reaction to the
changes that do occur** (the "lazy prices" thesis) — *"stock prices exhibit little to no reaction
at the time of public filing"* (p. 1378) — not because "most filings change very little." The
stickiness (high year-over-year similarity) is the backdrop; the return predictability comes from
under-reaction, not from the filings being unchanged.
Anchors two claims: (1) 10-K text changes are informationally meaningful; (2) year-over-year
Item 1A cosine similarity is a defensible scalar feature.

### Item 1A Stickiness — Rebuttal of Boilerplate Criticism

> ✅ **Campbell et al. (2014)** (full citation above, §2)

Their introduction quotes Reuters analysts calling risk factors "boilerplate" and empirically
rejects this — showing disclosures meaningfully reflect firm-specific risk types. **PDF-verified
(Round 3):** the epigraph — *"Risk factors are looked upon as boilerplate ... The irony of it is
that risk factors are almost meant not to be read, or relied upon. — Tom Taulli, IPO analyst
(Reuters 2005)"* (p. 397) — and the rebuttal — *"risk factor disclosures are not boilerplate but
instead meaningfully reflect the risks a firm faces"* (p. 398). Sharpest available rebuttal of the
boilerplate criticism from within the literature dirs.

### Successor — Changes in Risk Language → Volatility

> ✅ **Kravet, T. & Muslu, V. (2013). "Textual Risk Disclosures and Investors' Risk Perceptions."
> *Review of Accounting Studies*, 18(4), 1088–1122.**
> `Literature (new)/[2013] Textual Risk Disclosures and Investors' Risk Perceptions.pdf`

Studies changes in risk disclosures in 10-Ks. **PDF-verified (Round 4):** *"We find that annual
increases in risk disclosures are associated with increased stock return volatility and trading
volume around and after the filings"* (abstract, p. 1088); more precisely *"the annual increase in
the number of risk sentences in a company's 10-K filing is associated with higher return volatility
(particularly higher volatility of negative returns) and higher trading volume during the 60
trading-day period after"* (§1). **Correction:** the mechanism is the *number of risk sentences*
(the earlier draft's "risk-word frequency" was imprecise). This confirms the direct
risk-change → volatility link, corroborating what Campbell et al. (2014) cite it for. Use if a
specific risk-change citation is needed beyond Lazy Prices.

---

## 4. Evaluation Methodology

### (a) Expanding-Window / Walk-Forward Backtesting

> ✅ **Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
> (Chapters 7, 11, and 12.)**
> `Literature (new)/[2018, book] — Advances in Financial Machine Learning.pdf`

Standard reference for why k-fold CV is inappropriate for financial time series, and for
walk-forward / combinatorial purged CV protocols. **PDF-verified (Round 4), with a chapter
correction — the earlier "Chapters 4 and 7" was wrong:** Chapter 4 is *"Sample Weights"* (nothing
to do with CV/backtesting). The correct chapters are:
- **Chapter 7, "Cross-Validation in Finance"** — §7.3 *"Why K-Fold CV Fails in Finance"*, §7.4
  *"A Solution: Purged K-Fold CV"* (Purging §7.4.1, Embargo §7.4.2). This is the "why k-fold fails"
  + purged-CV reference.
- **Chapter 12, "Backtesting through Cross-Validation"** — §12.2 *"The Walk-Forward Method"* (and
  its pitfalls, §12.2.1), §12.4 *"The Combinatorial Purged Cross-Validation Method"*. This is the
  walk-forward / combinatorial-purged-CV reference.

Cite Ch. 7 and Ch. 12 (not Ch. 4).

### (b) Diebold-Mariano Test

> ✅ **Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
> *Journal of Business & Economic Statistics*, 13(3), 253–263.**
> `Literature (new)/[1995] Comparing Predictive Accuracy.pdf`

The original DM test. Old but obligatory — every forecast comparison paper cites this. The
finite-sample correction (Harvey, Leybourne & Newbold 1997) is a minor refinement; citing DM
(1995) alone is standard practice in finance ML papers. **PDF-verified (Round 4) — the Round-3
issues are now resolved:** a searchable copy of the **1995 original** was supplied (front matter:
*"Journal of Business & Economic Statistics, Jul., 1995, Vol. 13, No. 3, pp. 253-263"*), so the
on-disk PDF now matches the citation. The loss-differential mechanism is confirmed: *"The null
hypothesis of equal forecast accuracy for two forecasts is E[g(eᵢₜ)] = E[g(eⱼₜ)], or E[dₜ] = 0,
where dₜ = [g(eᵢₜ) − g(eⱼₜ)] is the loss differential"* (§1, p. 253). The test compares forecast
accuracy via the mean of the loss-differential series, with *"a wide variety of accuracy measures
... the loss function need not be quadratic and need not even be symmetric"* (abstract).

### (c) Lookahead Bias / Temporal Leakage

> ✅ **Lopez de Prado, M. (2018)** (same book as 4a — Chapters 7 and 11.)

The recognised reference for temporal leakage in financial ML. **PDF-verified (Round 4):** the
leakage/purging machinery is in **Chapter 7** (§7.4.1 *"Purging the Training Set"*, §7.4.2
*"Embargo"*) and **Chapter 11, "The Dangers of Backtesting"** (§11.2 *"Mission Impossible: The
Flawless Backtest"*). Cite Ch. 7 / Ch. 11 (not Ch. 4). Pair with a brief procedural description of
how the expanding-window protocol was implemented.

### (d) IC Magnitudes and Volatility Predictability

**IC benchmark context:**
> ✅ **Grinold, R.C. & Kahn, R.N. *Active Portfolio Management* (2nd ed.). McGraw-Hill.**
> `Literature (new)/[1999, book] Active Portfolio Management (2nd ed, McGraw-Hill).pdf`

Grinold & Kahn are the usual source for the point that a *small* information coefficient is
already valuable in cross-sectional *return* prediction. **PDF-verified (Round 4):** *"an
information coefficient of 0.0577 can lead to an information ratio above 1.0 (top decile ...).
Using Eq. (6.6), an IC = 0.0577 corresponds to correctly forecasting direction only 52.885 percent
of the time — a small edge indeed"* (Ch. 6, *The Fundamental Law of Active Management*); the
exercises use *"an IC = 0.05 and an IR = 1.0"* (Ch. 6). So an IC of ~0.05–0.06 already puts a
return forecaster in the top decile — which is the comparison being drawn: volatility ICs of
0.50–0.61 are an order of magnitude above the return-prediction benchmark.
**Bibliographic note:** the on-disk copy is the **1999** 2nd edition (this file previously cited
"(2000)"; the 2nd edition is frequently cited as either 1999 or 2000 — reconcile the year with
the copy you hold).

**Why volatility ICs of 0.5+ are not surprising — use Corsi (2009):**
> ✅ **Corsi (2009)** (full citation §1a)

Corsi's HAR-RV achieves high in-sample R² for realised-volatility forecasting using only past
volatility — 0.565 (USD/CHF), 0.707 (S&P500 futures), 0.236 (T-Bond futures) one-day-ahead
in-sample (Table 4, p. 189). **Correction (Round 3):** do not state a blanket "40–70%"; the
equity/FX in-sample values sit in that range but T-Bond and out-of-sample values are lower. Even
so, these R² far exceed the sub-2% typical of return prediction, which is the point being made:
volatility is a far more forecastable quantity than returns, giving the theoretical context for
why within-year Spearman ICs of 0.5–0.6 are plausible for volatility ranking. No additional
paper is needed for this point.

---

## 5. Long-Document Encoding

### Financial-NLP Precedent — Hierarchical Encoding

> ✅ **Yang, Z., Ng, K., Smyth, B. & Dong, R. (2020). "HTML: Hierarchical Transformer-based
> Multi-task Learning for Volatility Prediction." *The Web Conference (WWW 2020)*, pp. 441–451.**
> `Literature (new)/[2020] HTML Hierarchical Transformer-based Multi-task Learning...`

Encodes long earnings-call transcripts via a two-level architecture. **PDF-verified (Round 3):**
*"the proposed HTML model ... contains four components: (1) token-level transformer encoder; ...
(3) sentence-level transformer encoder; ..."* (§4); the token-level encoder is Whole-Word-Masking
BERT (*"the Whole Word Masking BERT (WWM-BERT)"*, §4), so "token-level BERT" is correct — more
precisely WWM-BERT. **Correction:** the specific "≤512 tokens" was dropped — it is BERT's
architectural default, not a figure stated in the paper. Direct financial-NLP precedent for
encoding long documents by processing per-sentence segments and aggregating with a higher-level
transformer. The flat sliding-window + mean-pool approach used here is the simplified variant.

### General NLP Precedent — Sliding-Window Chunking

> ✅ **Beltagy, I., Peters, M.E. & Cohan, A. (2020). "Longformer: The Long-Document
> Transformer." arXiv:2004.05150.**
> `Literature (new)/` (added 2026-07-07)

Section 2 ("Task-specific Models for Long Documents", p. 3), PDF-verified quote (**re-verified
Round 3**): "Another approach chunks the document into chunks of length 512 (could be overlapping),
processes each chunk separately, then combines the activations with a task specific model (Joshi
et al., 2019)." Cite for the claim that chunk-encode-combine is recognised prior practice for long
documents.
Note the framing: Longformer presents it as a known workaround (motivation for their
architecture), not an endorsement. Do NOT cite BERT (Devlin et al. 2019) for this claim — the
paper contains no sliding-window or chunking discussion (verified; no Appendix E exists). Do NOT
cite SBERT either (failed verification, round 1). *(See Verification Log.)*

### Paragraph-Level Encoding in Financial Filings

> ✅ **Chiu, Y.-H. et al. (2025). "Financial Risk Relation Identification through Dual-view
> Adaptation." *EMNLP 2025*.**
> `Literature/[2025; EMNLP] Financial Risk Relation Identification through Dual-view Adaptation.pdf`

Uses 256-token paragraphs as the encoding unit, corroborating the paragraph-as-unit principle and
the convention of truncating/padding to a fixed token budget. **PDF-verified (Round 3):** *"Each
input text piece is truncated or padded to 256 tokens"* (§4, p. 5); a paragraph is defined as the
token sequence `p = [w1, ..., wn]` (§3.1).

**Note:** No single paper canonically defines sliding-window + mean-pooling specifically for 10-K
filings. Final verified citation set for the practice: Longformer §2 (general prior practice) +
HTML / Yang et al. 2020 (financial hierarchical application) + Chiu et al. 2025 (256-token
paragraph unit in filings).

---

## 6. Fair Ablation Design

> ❌ **FLAG — no strong dedicated paper for "hold the prediction head fixed when ablating feature
> sets." This is methodological best practice, not a published protocol.**

Describe the ablation design explicitly in the methods section (HGB/Ridge head fixed; only the
input feature set varies across conditions). For any referee who asks:

> ✅ **Lipton, Z.C. & Steinhardt, J. (2019). "Troubling Trends in Machine Learning Scholarship."
> *ACM Queue*, 17(1). (arXiv:1807.03341.)**
> `Literature (new)/[2019] Troubling Trends in Machine Learning Scholarship.pdf`

Argues that papers frequently fail to identify the *true source* of empirical gains — attributing
improvements to architectural changes when they actually stem from other factors — and calls for
proper ablation studies to isolate the responsible factor. **PDF-verified (Round 4), with a wording
refinement:** the paper's trend #2 is *"Failure to identify the sources of empirical gains, e.g.
emphasizing unnecessary modifications to neural architectures when gains actually stem from
hyper-parameter tuning"* (§1); *"Too frequently, authors propose many tweaks absent proper ablation
studies, obscuring the source of empirical gains. Sometimes just one of the changes is actually
responsible"* (§3.2). (The confound named by the paper is architecture-vs-hyper-parameter-tuning,
not "architectural vs representational" as the earlier draft put it — corrected.) Directly on point
for the head-asymmetry finding elsewhere in the study.

For small IC differences being within noise: reference the IC benchmark discussion in §4d
(Grinold & Kahn 2000) — IC differences of ±0.002 are below any meaningful threshold.

---

## 7. Supervised / Task-Aligned Text Encoders in Finance

### Established Precedent for Task-Supervised Encoders

> ✅ **Qin, Y. & Yang, Y. (2019). "What You Say and How You Say It Matters: Predicting Financial
> Risk Using Verbal and Vocal Cues." *ACL 2019*, pp. 390–401.**
> `Literature (new)/[2019] What You Say and How You Say It Matters...`

Trains a BiLSTM-based model end-to-end on earnings-call content → volatility prediction. **PDF-
verified (Round 3):** the model is the Multimodal Deep Regression Model (MDRM) — *"The MDRM model
utilizes BiLSTM layer to extract context-dependent unimodal features, and subsequently fuses
unimodal features together using another layer of BiLSTM"* (§3); it is *multimodal* (verbal text
via pre-trained GloVe embeddings + vocal/audio features), and the target is stock volatility
(*"stock price volatility, which is the standard deviation of a stock's returns"*, §1). Title
confirmed: *"What You Say and How You Say It Matters: Predicting Financial Risk Using Verbal and
Vocal Cues"* (ACL 2019, p. 390). The earliest paper in the corpus that trains a text encoder
directly on a volatility target.

> ✅ **Yang et al. (2020) HTML** and **Sawhney et al. (2020) VolTAGE**
> (full citations in §5 and `Literature (new)/`)

Both use a BERT-family text encoder on the earnings-call → volatility task. **PDF-verified
(Round 3):** HTML uses WWM-BERT for the token-level encoder (§4); VolTAGE uses FinBERT as its
transcript encoder — *"We use FinBERT (Araci, 2019) as a sentence encoder, which is a pre-trained
language model based on BERT, for language modeling specific to the financial domain"* (§4.1).
(Note: "fine-tune" softened to "use" — VolTAGE describes FinBERT as its sentence encoder; whether
its weights are further fine-tuned vs frozen was not separately confirmed. The GloVe+BiLSTM
approach belongs to MDRM/Qin & Yang, *not* VolTAGE.) Together with Qin (2019) they confirm
task-aligned neural text encoders as the prior SOTA before this work.

### Temporal Non-Stationarity / Out-of-Period Degradation

> ✅ **Magner et al. (2025)** (full citation §2)

Uses TVP-VAR to show the tone→index-return relationship is time-varying. **PDF-verified (Round 3):**
*"we used ... the time-varying parameter vector autoregressive (TVP-VAR) ... TVP-VAR captures
time-varying spillover effects"* (§2.5). The paper documents time-varying spillovers between Item 1A
tone and the indices; the further inference that "a model trained on one period may not transfer
cleanly to another" is our extrapolation for the temporal-drift argument, not a claim Magner et al.
state directly.

> ✅ **Su, Y. et al. (2025). "FinMTEB: Finance Massive Text Embedding Benchmark." *EMNLP 2025*.**
> `Literature/[2025; EMNLP] FinMTEB Finance Massive Text Embedding Benchmark.pdf`

*(Corrected in Round 2; BoW finding PDF-verified in Round 3.)* FinMTEB contains NO across-time
evaluation and no temporal-shift analysis; do not cite it for temporal drift. Its relevant finding
is that bag-of-words representations outperform dense embedding models on financial STS tasks —
**verified quote:** *"traditional Bag-of-Words (BoW) models unexpectedly surpass all tested dense
embedding models on financial STS tasks, highlighting persistent challenges for current embeddings
in capturing nuanced financial semantics"* (§1 / §2, insight 3). Cite it in the encoder-grid
discussion (why count-based text beating dense embeddings on financial text is a documented
phenomenon), not in the temporal-drift framing. The temporal-drift claim rests on Lu et al. (2019)
and Magner et al. (2025) alone.

**For the general concept drift framing:**
> ✅ **Lu, J., Liu, A., Dong, F. & Gama, J. (2019). "Learning under Concept Drift: A Review."
> *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346–2363.**
> `Literature (new)/`

Peer-reviewed survey formally defining concept drift as a change in the joint distribution
P(X, Y) over time, and cataloguing mitigation strategies. **PDF-verified (Round 3):** *"concept
drift at time t can be defined as the change of joint probability of X and y at time t. Since the
joint probability Pt(X, y) can be decomposed into two parts as Pt(X, y) = Pt(X)·Pt(y|X), concept
drift can be triggered by three [sources]"*, formally *"∃t: Pt(X, y) ≠ Pt+1(X, y)"* (§II). More
citable than the often-cited Tsymbal (2004) tech report it supersedes.

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
| 1g | Momentum (control, past-12-mo) | Ang et al. (2006) **Table VII** | JF | ✅ Yes |
| 2 | BoW → 10-K volatility (baseline; MSE not R²; 1996–2006) | Kogan et al. (2009) | NAACL-HLT | ✅ Yes |
| 2 | Finance word list → volatility/returns | Loughran & McDonald (2011) | JF | ✅ Yes |
| 2 | Item 1A → idiosyncratic volatility | Campbell et al. (2014) | Rev. Account. Stud. | ✅ Yes |
| 2 | Item 1A sentiment → returns | Magner et al. (2025) | Finance Res. Lett. | ✅ Yes |
| 3 | 10-K text changes → returns / alpha (188 bps/mo; 1995–**2014**) | Cohen, Malloy & Nguyen (2020) | JF | ✅ Yes |
| 3 | Item 1A stickiness rebuttal | Campbell et al. (2014) | Rev. Account. Stud. | ✅ Yes |
| 3 | Risk-sentence changes → volatility | Kravet & Muslu (2013) | Rev. Account. Stud. | ✅ Yes |
| 4a | Walk-forward / purged CV (Ch. 7 & 12) | Lopez de Prado (2018) | Wiley (book) | ✅ Yes |
| 4b | Forecast comparison test (loss differential) | Diebold & Mariano (1995) | JBES | ✅ Yes (1995 original) |
| 4c | Lookahead bias / leakage (Ch. 7 & 11) | Lopez de Prado (2018) | Wiley (book) | ✅ Yes |
| 4d | IC benchmark (IC≈0.058 → top-decile IR) | Grinold & Kahn (1999) | McGraw-Hill (book) | ✅ Yes |
| 4d | Volatility forecastability (in-sample daily HAR R² 0.24–0.71) | Corsi (2009) | J. Financial Econometrics | ✅ Yes |
| 5 | Hierarchical chunk + pool (financial) | Yang et al. HTML (2020) | WWW 2020 | ✅ Yes |
| 5 | Sliding-window chunk + mean-pool (NLP) | Beltagy et al. Longformer (2020) §2 | arXiv | ✅ Yes |
| 5 | Paragraph-level encoding (financial) | Chiu et al. (2025) | EMNLP | ✅ Yes |
| 6 | Fair ablation / sources of empirical gains | Lipton & Steinhardt (2019) | ACM Queue | ✅ Yes |
| 7 | Task-supervised encoder → volatility | Qin & Yang (2019) | ACL | ✅ Yes |
| 7 | Task-supervised encoder → volatility | Yang et al. HTML (2020) | WWW 2020 | ✅ Yes |
| 7 | Temporal distribution shift in fin. NLP | Magner et al. (2025) | Finance Res. Lett. | ✅ Yes |
| 7 | BoW > dense embeddings on financial STS | FinMTEB Su et al. (2025) | EMNLP | ✅ Yes |
| 7 | Concept drift definition / survey | Lu, Liu, Dong & Gama (2019) | IEEE TKDE | ✅ Yes |

---

**Status summary:** as of Round 4, **all 24 citations have a local PDF** in `Literature/` or
`Literature (new)/` — the four previously-missing items (Lopez de Prado 2018, Grinold & Kahn 1999,
Kravet & Muslu 2013, Lipton & Steinhardt 2019) were supplied, and the Diebold & Mariano scan was
replaced with a searchable 1995 original. Every content, numeric, and bibliographic claim has now
been verified against its PDF (Rounds 3–4).

---

## Verification Log (2026-07-07)

Two claims from the 2026-07-06 version were checked against the PDFs and failed verification:

1. **SBERT (Reimers & Gurevych 2019) long-document claim: NOT SUPPORTED.** Section 4.2 of the
   paper is "Supervised STS" (sentence-pair evaluation); the paper never discusses long documents,
   overlapping windows, or pooling of segment embeddings. The claim was fabricated. Replaced by
   Beltagy et al. (2020) Longformer §3.2 (which explicitly describes chunk-then-combine as the
   standard long-document baseline) and BERT Appendix E (Devlin et al. 2019).
2. **FinMTEB (Su et al. 2025) temporal-shift claim: NOT SUPPORTED.** The paper benchmarks 15
   models across 64 datasets by task type; it contains no across-time evaluation and no
   temporal-drift analysis. Dropped from the temporal-drift framing (Lu et al. 2019 and Magner
   et al. 2025 carry that claim); retained only for its bag-of-words-versus-dense finding on
   financial STS.

**Round 2 (2026-07-07, after downloading the Longformer and BERT PDFs):** the round-1
replacements themselves failed verification and were corrected again.

3. **Longformer §3.2 claim: NOT SUPPORTED (wrong section, fabricated quote).** §3.2 is
   "Implementation" (CUDA kernels). The real passage is in §2, p. 3: "Another approach chunks the
   document into chunks of length 512 (could be overlapping), processes each chunk separately,
   then combines the activations with a task specific model (Joshi et al., 2019)." Corrected to
   cite §2, with the caveat that Longformer frames chunking as a known workaround, not an
   endorsement.
4. **BERT Appendix E claim: NOT SUPPORTED (appendix does not exist).** The BERT paper has
   appendices A–C only and contains no sliding-window, chunking, or stride discussion anywhere.
   Dropped entirely from this claim.

Final verified citation set for long-document windowing: Longformer §2 + Yang et al. (2020)
HTML + Chiu et al. (2025). All three PDF-verified.

~~All other citations in this file were spot-checked for author/venue/year plausibility without
issues, but only the claims listed in this log were verified against PDFs.~~ *(Superseded by
Round 3 below — every content/numeric/bibliographic claim has now been checked against its PDF.)*

---

## Verification Log — Round 3 (full pass, 2026-07-07)

Every remaining content, numeric, and bibliographic claim was opened and checked against its PDF.
Rule: no assertion about a paper without a quotable passage; failed claims are corrected in-place
from the same PDF or marked NOT SUPPORTED; no repairs from memory. Verdicts below (SUPPORTED
entries listed too, so the whitelist is explicit).

**SUPPORTED (verified with a quotable passage):**

- **Fama & French (1992):** SUPPORTED. *"Two easily measured variables, size and book-to-market
  equity, combine to capture the cross-sectional variation in average stock returns"* (abstract).
- **Amihud (2002):** SUPPORTED, with a precision fix — measure is `|R|/dollar-volume`
  (*"the daily ratio of absolute stock return to its dollar volume"*, §2), not raw share volume.
- **Loughran & McDonald (2011):** SUPPORTED. Harvard-IV mismatch and the explicit volatility link
  both quoted (abstract).
- **Campbell et al. (2014):** SUPPORTED (both §2 and §3 uses). Positive unexpected-disclosure →
  volatility (p. 398) and the Reuters "boilerplate" epigraph (p. 397) quoted.
- **Magner et al. (2025):** SUPPORTED. 21,421 reports / 2002–2024 / LM+GPT / index returns /
  TVP-VAR time-varying spillovers all quoted (§2.1, §2.5, abstract).
- **HTML — Yang et al. (2020):** SUPPORTED. Two-level token-/sentence-level transformer confirmed
  (§4); token encoder is WWM-BERT. Minor fix: dropped the unstated "≤512 tokens".
- **Longformer §2 — Beltagy et al. (2020):** RE-VERIFIED (load-bearing). Quote confirmed at §2, p. 3.
- **Chiu et al. (2025):** SUPPORTED. *"Each input text piece is truncated or padded to 256 tokens"*
  (§4, p. 5).
- **Qin & Yang (2019):** SUPPORTED. BiLSTM-based MDRM, multimodal (GloVe text + audio), volatility
  target; title/venue confirmed (ACL 2019, p. 390).
- **VolTAGE — Sawhney et al. (2020):** SUPPORTED. FinBERT sentence encoder confirmed (§4.1);
  softened "fine-tune"→"use". (My prior suspicion that VolTAGE used GloVe was wrong — that is MDRM.)
- **FinMTEB — Su et al. (2025):** SUPPORTED for the BoW-vs-dense finding (quote, §1/§2, insight 3).
- **Lu et al. (2019):** SUPPORTED. Formal `Pt(X, y) ≠ Pt+1(X, y)` drift definition quoted (§II).
- **Corsi (2009) horizons:** SUPPORTED. Monthly = 22 working days (§3.2, p. 186); daily/weekly(5d)/
  monthly(22d) confirmed.

**CORRECTED (claim was wrong; fixed in-place from the same PDF):**

5. **Kogan et al. (2009): three errors.** (a) Sample is **1996–2006**, not 1996–2005
   (*"54,379 reports ... over the period 1996–2006"*, §3). (b) Target is **log stock-return
   volatility (standard deviation)**, not "log excess return variance" (§2–3). (c) The paper
   reports **MSE, not R²** — the "R² of 0.30–0.40" does not appear in it and was **removed**
   (fabricated numeric).
6. **Cohen, Malloy & Nguyen (2020) "Lazy Prices": sample period wrong.** It is **1995–2014**
   (*"from 1995 to 2014"*, p. 1377), not 1995–2017. The 188 bps/month alpha is correct (abstract),
   but the alpha mechanism was re-stated as investor **inattention/under-reaction** (the "lazy
   prices" thesis), not "most filings change very little."
7. **Ang et al. (2006): wrong table + wrong momentum horizon.** The size/B-M/leverage/momentum
   controls are in **Table VII** ("Controlling for Various Cross-Sectional Effects", p. 279), **not
   "Table 2"** — Table II is "Factor Correlations". Momentum is **past-12-month** returns, not
   6-month. Direction fix: high IVOL → *abysmally low* average returns (the IVOL puzzle), not a
   generic "predicts returns".
8. **Corsi (2009) "R² 40–70%": overreach.** Actual in-sample one-day-ahead HAR(3) R² is 0.565
   (USD/CHF), 0.707 (S&P500), 0.236 (T-Bond) (Table 4); out-of-sample lower. Replaced the blanket
   "40–70%" with the specific values (used in both §1a and §4d).
9. **Conrad et al. (2013) "and volatility": overreach.** Paper links moments to *returns*
   (negatively skewed → higher returns; abstract). "and volatility" removed.

**UNVERIFIABLE (flagged, not repaired from memory):**

10. **Diebold & Mariano (1995):** the on-disk PDF is the **2002 JBES reprint (20:1, 134–144)**, not
    the 1995 original (13(3), 253–263) the citation lists; and its **body is image-only** (no text
    layer), so the "loss differential" mechanism could not be quoted. Bibliographic mismatch flagged;
    content claim not asserted.
11. **Lopez de Prado (2018), Grinold & Kahn (2000), Kravet & Muslu (2013), Lipton & Steinhardt
    (2019): UNVERIFIABLE-NO-PDF.** Specific attributions softened/flagged — Lopez de Prado
    "Chapters 4 and 7", Grinold & Kahn "IC 0.05–0.10", the Kravet-Muslu finding (indirectly
    corroborated via Campbell 2014's citation of it), and the Lipton-Steinhardt paraphrase — all
    marked for download-and-verify before any specific claim is relied upon.

*Last updated: 2026-07-07 (verification round 3 — full PDF pass).*

---

## Verification Log — Round 4 (2026-07-07, the five downloaded PDFs)

The five items flagged UNVERIFIABLE in Round 3 were supplied and checked against their PDFs to the
same standard (open, quote, section/page).

**SUPPORTED (verified with a quotable passage):**

- **Diebold & Mariano (1995):** SUPPORTED, and the Round-3 problems are resolved — the supplied
  copy is the searchable **1995 original** (JBES 13(3), 253–263), matching the citation. Loss
  differential confirmed: *"E[dₜ] = 0, where dₜ = [g(eᵢₜ) − g(eⱼₜ)] is the loss differential"*
  (§1, p. 253).
- **Kravet & Muslu (2013):** SUPPORTED — *"annual increases in risk disclosures are associated with
  increased stock return volatility"* (abstract); mechanism is the *number of risk sentences* (§1).
  Upgraded from the Round-3 indirect (Campbell) corroboration to direct verification.
- **Lipton & Steinhardt (2019):** SUPPORTED — trend #2 *"Failure to identify the sources of
  empirical gains ... when gains actually stem from hyper-parameter tuning"* (§1); calls for
  ablation (§3.2).

**CORRECTED (claim was wrong; fixed in-place from the PDF):**

12. **Lopez de Prado (2018): wrong chapters.** "Chapters 4 and 7" was wrong — **Chapter 4 is
    "Sample Weights"**, unrelated to CV/backtesting. Correct references: **Ch. 7** ("Cross-Validation
    in Finance"; §7.3 "Why K-Fold CV Fails in Finance"; §7.4 Purged K-Fold CV) for §4a's CV point,
    **Ch. 12** ("Backtesting through Cross-Validation"; §12.2 Walk-Forward, §12.4 Combinatorial
    Purged CV) for walk-forward, and **Ch. 11** ("The Dangers of Backtesting") for §4c leakage.
13. **Grinold & Kahn: numeric attribution refined + year.** The book supports the *substance* — an
    IC of 0.0577 gives an information ratio > 1.0 (top decile) and corresponds to forecasting
    direction only 52.9% of the time (Ch. 6) — so cite that concrete result rather than a bare
    "IC 0.05–0.10 is good." Also the on-disk copy is the **1999** 2nd edition; the file previously
    cited "(2000)" (reconcile the year).

**Net result:** every citation in this file now has a local PDF and every content/numeric/
bibliographic claim has been checked against it across Rounds 1–4. Two open bibliographic
housekeeping items remain (both flagged inline, neither a content error): the Grinold & Kahn
edition year (1999 vs 2000), and — should you prefer to cite the reprint — the Diebold & Mariano
1995-vs-2002 choice.

*Last updated: 2026-07-07 (verification round 4 — five downloaded PDFs).*
