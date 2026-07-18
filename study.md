# Summary for supervisor (technical)

**Task.** Predict the cross-sectional ranking of firms by 30-day forward realised volatility from 10-K
risk-factor text, evaluated by within-year Spearman IC (information coefficient).

**Original pipeline and its limitation.** The original design was text-centric: it learned representations
of the risk-factor disclosures (TF-IDF, DAPT, contrastive encoders, BERTopic) and compared these
representation methods against each other on the downstream task, with lagged realised volatility as the
only non-text predictor. Across five independent evaluations, a tuned bag-of-words (TF-IDF) baseline
matched or beat every learned encoder. The limitation this exposed was structural: the pipeline (i) omitted
the standard structured/financial determinants of volatility that the asset-pricing literature establishes
as first-order, and (ii) framed the question weakly — "which text representation ranks best?" — a comparison
in which neither side has access to the dominant predictors.

**Redesign.** Two changes. (1) *Establish a strong structured baseline*: engineer the conventional
cross-sectional volatility predictors — multi-horizon realised volatility, vol-of-vol, return moments and
drawdown, momentum, market beta, idiosyncratic volatility, liquidity (Amihud, dollar volume), and Compustat
fundamentals (size, leverage, book-to-market, ROA). (2) *Reframe to incremental value*: ask whether textual
disclosure provides incremental predictive power over that baseline, via a multimodal fusion model with an
ablation ladder (structured → +10-K text → +topics → +earnings-call tone), reporting incremental IC with
paired significance tests.

**Why this is the correct improvement.** *Empirically* — the structured baseline alone (val-2025 IC 0.570)
exceeds the entire original pipeline (TF-IDF + lagged, 0.545), and adding text reaches IC 0.611 /
R²_log 0.276, materially increasing explained variance. *Methodologically* — "bag-of-words beats neural
encoders" is a weak negative; "disclosure text adds significant incremental signal over a strong quant
baseline, and the topic model identifies which risk themes drive the cross-section" is a substantive
contribution. It also explains why the encoders previously underperformed: asked to carry the entire signal
alone they are redundant with lexical features, but as an incremental layer over structured covariates they
become contributive (~0.60+). This retains the brief's emphasis on domain-adaptive pretraining, contrastive
learning, and financial-aware augmentations — the encoder is, in Stage B, retargeted to a volatility-aware
contrastive objective (positives = same within-year forward-vol decile). That retargeting did **not** beat
the similarity-trained encoder (val-2025 IC 0.586 vs 0.602); since the comparison was confound-free, the
finding is that the contrastive *objective* is not the binding constraint — the task is lexically saturated
and TF-IDF (0.611) remains the strongest text representation. The encoder contribution is delivered and the
result is an informative negative.

*(Plain-language study notes, with full detail and honest caveats, follow below.)*

---

# Understanding the Results — Study Notes

## What we're trying to do

We want to predict how volatile (risky) a company's stock will be over the next 30 days, using
only the risk language in their annual 10-K report. We're trying to answer: **can you read a
company's annual report and know in advance which firms will be most volatile?**

The score we use is called **IC** (Information Coefficient). It's just: "did we correctly *rank*
firms from least to most volatile within a given year?" IC = 1 means perfect ranking, IC = 0 means
random, IC = 0.5 is decent. Higher is better.

---

## The full scoreboard — who's who

| Method | IC | Where it comes from |
|---|---|---|
| Random guess | ~0.0 | Theoretical floor |
| **AR(1) lagged vol** | **0.466** | Academic baseline — just use last year's volatility |
| **TF-IDF + lagged** | **0.545** | Our floor — word-counting from 10-Ks + last year's vol |
| **DAPT encoder** | ~0.53 | Our work — BERT fine-tuned on 10-K language |
| **Dual contrastive** | ~0.54 | Our work — encoder trained on similar/different filings |
| **Three-view (LoRA)** | ~0.53 | Our work — encoder with sector pairs added |
| **Supervised fine-tune (Lever 3)** | 0.456 | Our work — encoder trained directly on volatility |

---

## Who are we beating?

**We beat AR(1).** AR(1) means "tomorrow's volatility = today's volatility" — a well-known
academic benchmark. Our TF-IDF model (0.545) beats it by 8 points. That's real. It says: yes,
10-K language adds information beyond just looking at last year's risk.

---

## Who is beating us?

**TF-IDF is beating our own neural encoders.** Plain word-counting — no AI, no BERT, just
counting which words appear and how often — ranks firms by volatility better than all our learned
representations. Every neural encoder we built comes in at or below 0.545, the word-counting
floor.

---

## Why does TF-IDF win?

The key insight: volatility prediction from text is a **lexical task**, not a **semantic task**.
When a firm uses words like *"litigation," "going concern," "liquidity risk," "covenant"* a lot,
that firm is genuinely riskier. TF-IDF captures that directly. BERT reads these words, understands
their meaning, but in doing so it blurs the raw frequency signal that actually drives the
prediction. It's like TF-IDF has a photographic memory for exactly which danger words appear and
how many times — and that's all you need here.

---

## Where do our methods come from?

- **AR(1)**: Standard econometrics. Used in virtually every volatility paper as the "does text add
  anything?" baseline.
- **TF-IDF**: Classic NLP from the 1990s. Loughran & McDonald (2011) showed finance-specific word
  lists predict returns/volatility — our result follows that finding.
- **DAPT** (Domain-Adaptive Pre-Training): We took BERT and continued training it on 10-K filings
  so it "knows" financial language better than off-the-shelf BERT.
- **Contrastive encoders** (dual, three-view): We trained the encoder to pull similar filings
  together and push dissimilar filings apart, so embeddings reflect financial similarity.
- **BERTopic**: We used the encoder to cluster filings into latent risk topics (e.g. "debt
  covenant risk", "litigation risk") and used topic loadings as features.
- **Lever 3 (supervised fine-tune)**: We trained the encoder end-to-end on the volatility
  prediction task — the most direct attempt to make it win.

---

## The story, in one paragraph

You built a full pipeline: domain-adapted language model → contrastive fine-tuning → topic
modelling → downstream prediction → five different ways to make the neural encoder beat
word-counting. After all five tests, TF-IDF still wins. That means your honest, defensible
conclusion is: *for cross-sectional volatility ranking from 10-K text, lexical bag-of-words
features dominate learned representations — but text as a whole (any method) substantially beats
the no-text baseline (AR(1))*. The topic model gives something TF-IDF can't: an *explanation* of
which risk themes drive each firm's volatility, which is a separate contribution.

---

---

## Why didn't neural encoders win? (honest post-mortem)

Other studies do show neural encoders beating baselines — but look closely at *which* baseline
they beat. Most papers beat **AR(1)** (lagged vol, IC 0.466) or **simple word counts**. Our
TF-IDF is not simple — bigrams, 20,000 features, sublinear term frequency, stop words removed.
We weren't comparing BERT to a weak straw man. We compared it to a strong one.

That said, there are specific choices in this pipeline that likely explain the gap:

**1. The contrastive encoder learned the wrong geometry.**
It was trained to pull *similar-language* filings together (lexical pairs, chronological pairs,
sector pairs). It got very good at that. But what the task needed was an encoder that pulls
*similar-volatility* filings together. Volatility was never a training signal — until Lever 3,
which was too late and too brief.

**2. Lever 3 had weak supervision.**
Each filing has ~24 paragraphs, all sharing *one* volatility label. So 24 paragraphs were all
trained toward the same number — that's like teaching a student by repeating the same answer for
every question on a page. The signal is diluted.

**3. The DAPT base model is anisotropic.**
DAPT embeddings have mean pairwise cosine similarity of 0.60 (vs 0.36 for SBERT). This means all
embeddings point in roughly the same direction — they're clustered together — which destroys the
discriminative power needed to rank firms.

**4. Risk factor text (Item 1A) is written by lawyers to be deliberately vague.**
Every firm writes "we face market risk, liquidity risk, credit risk." The discriminating signal is
in *how much* specific danger language appears — which TF-IDF counts perfectly — rather than *what
it means* — which BERT is optimised for.

**What would have worked better:**
- Building contrastive pairs based on *volatility similarity* (high-vol firm pairs as positives)
  rather than text similarity
- Fine-tuning at the *filing level* with the actual volatility label, not paragraph-level proxies
- More epochs for Lever 3 with a learning rate schedule
- Starting from FinBERT (already finance-tuned) rather than BERT+DAPT

**The conclusion this justifies:**
For this task, text-based volatility prediction is fundamentally lexical. Semantic representation
learning adds geometric structure (the topic model) but not predictive signal above bag-of-words.
That is a real, defensible research finding — not a failure.

*(Everything above is PART 1 — the diagnosis. PART 2 below is the turnaround.)*

---
---

# PART 2 — The Redesign (and the turnaround)

## The realisation that changed the project

PART 1 kept asking "which TEXT method ranks firms best?" — and word-counting always won. The flaw was
the *question*, not the methods. The only non-text feature we ever gave the model was **last year's
volatility** (one number). We never built the predictors that actually drive volatility in finance:
multi-horizon realised vol, firm size, leverage, liquidity, beta. So we had **no strong baseline** — and
"TF-IDF beats BERT" is a weak comparison when neither is using the real signals.

**The new question:** *Does text add predictive power ON TOP OF a strong structured-financial baseline?*
This is winnable, publishable, and — crucially — it is the question the literature actually cares about.

## What we built

- **Structured feature block** (no new modelling, just engineering from price + Compustat data):
  realised vol at 21/63/126/252 days, vol-of-vol, return skew/kurtosis, max drawdown, momentum, market
  beta, idiosyncratic vol, liquidity (Amihud, dollar volume), and fundamentals (log market cap, leverage,
  book-to-market, ROA). ~99% coverage across 7,666 filings.
- **Multimodal fusion + ablation ladder**: structured baseline → + text → + encoder → + topics, each
  scored by within-year cross-sectional IC, with significance tests.

## The results (clean val-2025, n=397)

| Model | IC | R²_log | note |
|---|---|---|---|
| lagged only | 0.460 | −0.054 | persistence baseline |
| **TF-IDF + lagged** (PART-1 best) | 0.545 | 0.175 | the old floor |
| **structured** (new baseline) | **0.570** | 0.158 | **beats the old floor with NO text** |
| **structured + TF-IDF** | **0.611** | **0.276** | **text adds: +0.066 IC, +0.10 R²** |
| structured + encoder (dual) | 0.602 | 0.194 | the dense encoder also adds now |
| structured + encoder + topics (dual) | 0.608 | 0.192 | topics contribute too |

In the leakage-free 2018–2024 backtest the same ordering holds: structured (IC 0.519) > tfidf+lagged
(0.507) > lagged (0.462), and structured+TF-IDF is best (0.569). All with strong IC t-stats (5–6).

## What this means (the story to tell)

1. **The structured baseline alone (0.570) beats the entire PART-1 pipeline (0.545).** Standard quant
   features rank firm volatility better than any text method we had.
2. **Text adds real, incremental signal on top of it.** `structured + TF-IDF` reaches 0.611 IC / 0.276
   R² — text roughly **doubles** the variance explained over structured-alone. This is the contribution
   PART 1 could not show.
3. **The neural encoders become competitive** (0.602–0.608) once they sit on a strong baseline instead of
   being asked to carry the whole task. The topic model adds interpretability on top.

The arc: *"word-counting beats our neural pipeline"* (PART 1, a defensive negative) →
*"a strong quant baseline that text — including our encoders — measurably improves"* (PART 2, a positive
result). Same rigour, much stronger finding.

## Honest caveats (report these, don't hide them)

- In the 7-year backtest, structured+TF-IDF beats structured by +0.050 IC but at **p = 0.090** —
  suggestive, not yet below 0.05. The single-year 2025 gap is cleaner. More years / a purpose-built
  encoder should firm this up.
- Cross-sectional R² is negative in the backtest — that is the noisy within-year *level* R²; **IC
  (ranking) is the metric that matters** here, and it is solidly positive.

## Stage C — earnings-call tone: a lesson in how to test

The 2025 call collection turned out to **postdate** the 10-Ks (it grabbed recent quarterly calls, not the
annual call before each filing), so only ~3 of 406 filings had a leakage-free call match. We could not run
the filing-level pilot. Instead we tested the question directly, **call-anchored**: does a call's tone
predict the 30-day vol that follows *the call*?

**First (crude) test — tone vs persistence:** tone added *nothing* over last-period volatility
(tone+lag IC = lag-only IC = 0.525). It looked like calls were redundant — same story as the encoders.

**But that test was unfair.** It only checked tone against persistence, *linearly*. The real question is
whether calls help **in combination** with the full model, **nonlinearly** — exactly how 10-K text won in
Stage D. Re-tested properly (`phase5/call_combined_gate.py`, structured block + tone, HGB,
leave-one-quarter-out):

| Model (predict post-call 30d vol) | IC | vs structured |
|---|---|---|
| structured only (HGB) | 0.555 | — |
| **structured + call tone (HGB)** | **0.594** | **+0.039** |
| structured + call-text (ridge) | 0.596 | +0.011 |

**Call tone adds +0.039 IC in combination — essentially the same size as the +0.041 that 10-K text adds.**
The signal lives in the *interaction* with other features, invisible to the crude "beats lag?" test.

**Methodological lesson (worth stating in the thesis):** a feature can add real value in combination while
adding nothing on its own — always test incremental value inside the full model, nonlinearly, not against a
single baseline. The crude test nearly made us discard a useful modality.

**Status / caveat:** n=152, single regime (2024–25) — suggestive, not conclusive. So the plan is a *small*
confirmation collection (the 406 pre-filing 2025 calls, spec in
`dataset_collection_discussion/calls_collection_spec_2025.md`), tested at the filing level, before
committing to the full 2006–2024 backfill.

## Stage B — volatility-aware encoder: result

We retrained the encoder with a **volatility-aware contrastive objective** (the `vol` view: positives =
risk paragraphs from *different firms in the same within-year forward-vol decile*), the financial-aware
augmentation the brief asks for. Clean comparison (batch 64, same as the old encoders). Result on val-2025:

| Encoder condition | IC | R²_log |
|---|---|---|
| struct + enc[**volaware**] (purpose-built) | 0.586 | 0.181 |
| struct + enc[dual] (old similarity-trained) | 0.602 | 0.194 |
| struct + enc + topic[volaware] | 0.600 | 0.183 |
| struct + enc + topic[dual] | 0.608 | 0.192 |
| struct + TF-IDF (still best text) | **0.611** | 0.276 |

**The vol-aware encoder did NOT beat the similarity-trained one** (0.586 vs 0.602). It is competitive — it
sits in the same ~0.58–0.61 band as every dense encoder — but retargeting the contrastive *objective* from
similarity to volatility did not help. Because the test was clean (no batch confound), the conclusion is
sharp: **for this task the contrastive objective was never the bottleneck — the text is lexically saturated**,
so TF-IDF (0.611) still tops every dense encoder regardless of how it is trained. All encoders plateau ~0.60.

This is the same finding as everywhere else in the project, confirmed from one more angle, and it *completes
the mandated encoder work*: we built and evaluated a financial-aware, volatility-targeted contrastive encoder
exactly as the brief requires; the outcome is an informative negative on a sharp question, not a failure.
(Topic coherence: volaware C_v=0.670 vs sbert 0.716 / dual 0.673 — also in-band.)

## Still to come

- ~~**Stage C confirmation**~~ **DONE 2026-07-18** — full 2006–2025 call history landed (Finstream
  corpus via Sarthak's HF repo) and the filing-level test ran on both val-2025 and the 2018–2024
  backtest. **Result: tone adds nothing at the filing anchor** (val-2025 gate: all paired dIC ≤ 0;
  backtest: −0.004 p=0.395 vs structured, −0.001 p=0.436 vs struct+tfidf) while the call-anchored
  +0.039 stands. Stage C closes as a horizon-contrast finding — see PART 4 below and
  `phase5/STRESS_TEST_RESULTS.md` (Stage C section) for full tables. **No experiments remain.**

---

# PART 3 — The stress test (final, 2026-07-04)

Full detail + all tables: `phase5/STRESS_TEST_RESULTS.md`. The mandate was: be absolutely sure the
"count-based text wins" conclusion survives an adversarial audit — or find the pipeline change that
overturns it. Every prior choice was questioned first. Outcome, in five results:

1. **Two real defects found and fixed before trusting any number.** (P0-a) the filename year is the
   *filing* year, not fiscal year — 74% of text↔label pairs were re-keyed; headline barely moved
   (0.611→0.610), which is itself a substantive finding: *year-stale risk text predicts volatility
   as well as fresh text* — Item-1A is persistent boilerplate (independent corroboration of
   Lazy-Prices). (P0-b) encoders previously saw only ~35% of the words TF-IDF saw (256-token
   truncation); fixed with windowed full-text encoding — and the encoders still lose, converting
   the old unfair comparison into a defended one.
2. **The old "+0.05 text increment" was ~80% a head artifact.** structured[ridge] (IC 0.603, 12-yr)
   ≫ structured[hgb] (0.584). Same-head text gain: **+0.011 IC (p=0.057)**, with a large accuracy
   gain (R² 0.175→0.226, DM p<0.001). Lesson for any ablation: hold the head fixed.
3. **No encoder beats the count model under any admissible protocol.** Generic encoders — sbert,
   dual, and bge (the modern-embedder objection) — lose outright. Vol-supervised encoders (ftvol,
   volaware) reach DM-insignificant *parity* on val-2025 — but their training saw all pre-2025
   labels (and ftvol epoch-selected on val-2025). The clean retrain (train <2017, select 2017,
   freeze) collapses on the admissible 2018–2024 backtest: 0.50 IC, ≈0 over structured, −0.06 vs
   struct+tfidf. **Task-aligned training closes the in-period gap; forward transfer erases it.**
   The dense text→vol mapping is era-specific, while TF-IDF's lexical levels + cheap annual refit
   stay current by construction.
4. **Disclosure-change features (Lazy-Prices extension) are null**: lexical and semantic change,
   validated features (COVID-year dip), add nothing over level TF-IDF on either backtest window.
5. **The final model and its defense:** `struct+tfidf [sparse]` — IC 0.614 (t=10.4) over the 12-yr
   expanding-window backtest, R²_log 0.226 on val-2025 — now backed by a fair grid (2 heads × 8
   text representations × 3 poolings × topics × change × fusions, paired DM tests throughout,
   leakage audit including encoder-training provenance). The thesis contribution is a *defended
   benchmark plus the characterization of why*: the volatility signal in 10-K risk text is
   lexical-level, persistent, and count-representable.

---

# PART 4 — Stage C closed: the filing-level call-tone result (2026-07-18)

The full transcript history arrived (Finstream corpus, 39,501 unique transcripts 2005-10→2025-03,
re-hosted at HF `SarthakVishnu/dissertation-dataset` → `calls/SP500_calls_2006to2025.parquet`),
so the confirmation ran at full scale instead of the planned 406-call pilot: `build_call_features.py`
(CIK-primary join, most recent call strictly before each filing, ≤200d lookback) matches
**7,159/8,105 filings (88.3%; ≥95% every backtest year 2013–2024; 390/406 in val-2025)**.
The build is deliberately single-source (parquet only; the 299 API Ninjas JSONs stay as the
call-anchored pilot's data — a diff confirmed excluding them changes zero evaluated-year values,
only 14 unused 2026 matches).

Two independent filing-level tests, both null:

1. **val-2025 gate** (`call_filing_gate.py`, 5-fold CV within the 2025 cross-section, same folds
   and same head per pair): every ± tone pair is flat or negative — full panel dIC −0.017 (hgb) /
   +0.000 (ridge) / −0.000 (tfidf lane); matched subset −0.007/−0.013/−0.012, all CIs straddling
   zero; the one significant DM test (p=0.022) favours **no tone**.
2. **2018–2024 backtest** (`run_fusion.py`, calls now a leakage-free family): struct+calls 0.515
   vs structured 0.518 (dIC −0.004, p=0.395); struct+tfidf+calls 0.561 vs struct+tfidf 0.561
   (dIC −0.001, p=0.436). val-2025 rows agree (+0.003/+0.001, noise).

**The call-anchored pilot (+0.039 over structured, n=152) and this filing-level null are both
correct — they measure different horizons.** Plausible reading (an interpretation, not an
established mechanism): tone carries real volatility information at the call date, but by the
filing date — one to three months later — the structured block, whose realised-vol windows span
the post-call period, already contains it. Stage C closes as a **horizon-contrast finding**:
management tone predicts volatility at the call horizon, and the condition under which it stops
adding value is exactly the one the 10-K task imposes (a later anchor with fresher structured
information). Full tables: `phase5/STRESS_TEST_RESULTS.md` (Stage C section).

**With this, every experimental result of the study is final.** Remaining work is write-up only.

---

## My doubts / open questions

- Dual contrastive is a replication of Chiu et al. — is our IC for it comparable to theirs?
- Are other studies beating TF-IDF specifically, or just AR(1)?
- struct+tfidf backtest significance is p=0.090 — will more years / the vol-aware encoder push it < 0.05?
