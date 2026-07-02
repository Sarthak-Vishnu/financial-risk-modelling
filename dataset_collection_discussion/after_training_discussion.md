# After-Training Data-Integration Decision Memo

**Project:** Hybrid Topic and Domain-Adaptive Modelling for Financial Risk and Forecasting
**Student:** S2880814 · **Supervisor:** Prof Tiejun Ma · **RA/verifier:** Sean Choi · **Collab:** Sunnie Li
**Status:** Phases 2–4 trained; Phase 5 baselines + scoring harness in place; domain encoders being scored.
**Purpose:** Internal decision memo. Re-reviews the 2026-06-17 data-collection mail *against empirical
results* and sets a **pilot → decide → integrate (with re-training if it pays)** path per dataset. Strategy:
**core-first** — prove the hybrid representation beats the simple baselines on the real task, run cheap pilots
in parallel, and spend collection/re-training budget where measured marginal value is highest.

> **North star.** Marks are won by **beating the TF-IDF+lagged floor and prior volatility-forecasting
> literature on the Phase-5 downstream task** — not by collecting more data or adding novel architecture for
> its own sake. Every decision below is judged against that.

---

## 0. The two rules that decide everything

Both came out of the post-training evidence and override the instincts in the original mail.

> **Framing correction (important).** The 1-year (2025–26) collections of GDELT, earnings calls, and full
> 10-K/10-Q were a **deliberate pilot strategy** — collect cheaply for one year, measure whether the signal
> helps, and *only then* decide whether to invest in the full historical pull. They are **not** "unusable
> mistakes." The two rules below are therefore **pilot-design constraints**, not disqualifiers — and
> **re-training is on the table**: re-doing DAPT / contrastive with new data is a live option whenever a pilot
> says it pays. Decisions here are **benefit-driven (pilot-gated), not cost-avoidant**.

> **UPDATE 2026-06-22 — earnings calls flipped from "strategic bet / not pilotable" to "collect (gate-confirmed)."**
> This memo originally judged earnings-call *tone value* unmeasurable at 1 year (only a "thin scalar pilot").
> That turned out to be too pessimistic. The 2025 calls postdate the 10-Ks (can't match at filing level), but
> we settled the question **call-anchored** instead: predict the 30-day vol *after each call*, computing the
> full structured block at the call date. Result (`phase5/call_combined_gate.py`, HGB, leave-one-quarter-out,
> n=152): **call tone adds +0.039 IC on top of the structured baseline (0.555 → 0.594)** — about the same as
> the +0.041 the 10-K text adds in the main ladder. The signal is in the *interaction* (it adds nothing vs
> persistence *alone*), which is why the original "scalar-only / marginal-R²" framing missed it. Caveat:
> single regime, suggestive not significant. **Decision: collect the 406 pre-filing 2025 calls**
> (`dataset_collection_discussion/calls_collection_spec_2025.md`), confirm at the filing level, and **only
> then** commit to the 2008/2006–2024 backfill described in the rows below. The "strategic bet / re-training"
> rows below still stand for the *DAPT-corpus* value (which genuinely needs scale) and the 4th contrastive
> view — what changed is that **call-tone is now an evidence-backed collect, not a blind bet.**

**Rule 1 — Temporal-coverage gating (a pilot-design constraint).** Phase 5 fits its prediction head on
**train = 2006–2024** and scores **val = 2025**. *A feature that does not cover the train period cannot have a
coefficient learned for it on the main split.* This does **not** mean a 1-year signal is useless — it means
its pilot must be designed **within the covered window** (or as a daily panel), §3.5. If the pilot is
positive, the full-history pull + a head re-fit (or re-training) follows.

**Rule 2 — Target is volatility, not returns.** The prediction target is `fwd_vol_30d` (30-day forward
realised volatility). Most evidence cited in the mail (insider-trading *alpha*, news→*return* variance) argues
for *direction/returns*. A signal that predicts returns does not automatically predict volatility. Every
retained signal — and every pilot — needs a **volatility-specific** rationale (information asymmetry,
disagreement, attention/news *volume*/dispersion), or its justification must be revised.

---

## 1. What changed since the mail was written

The mail (2026-06-17) predates DAPT. We now have results, and they reset the priorities:

| Evidence | Result | Consequence for data strategy |
|---|---|---|
| DAPT (Phase 2) | Improves in-domain STS, **degrades retrieval** (MLM anisotropy) | More domain text ≠ automatically better representations |
| Contrastive FT (Phase 3) | `three_lora` best; **Sector view (View 3) *hurt* full fine-tuning** | Adding a *4th* view is risky before the existing 3 are proven net-positive |
| FinMTEB | Generic SBERT still wins OOD retrieval | Domain adaptation has not yet "won" — decisive test is Phase 5 |
| **Phase 5 baselines (val 2025)** | **TF-IDF + lagged: R²_log 0.18 / Spearman 0.60**; plain TF-IDF best ranking (0.60); **generic SBERT embeddings *under*-perform TF-IDF** | The bar to clear is **0.18 / 0.60**. Cheap bag-of-words is already a strong competitor |
| 2026 test labels | Unavailable (no 2026 price feed) → eval on **val 2025** | Any "test-set" claim waits on CRSP-2026 |

**Implication:** the project's immediate risk is *not* "too few signals" — it is that the learned text
representations have **not yet beaten a bag-of-words baseline** on the actual task. Expansion must serve that
fight, or it is a distraction.

---

## 2. Critical review of the mail — claim by claim

| Mail claim | Verdict | Reasoning |
|---|---|---|
| **Full 10-K (beyond 1A/7A) excluded from NLP** | ✅ **Holds** | Campbell et al. 2014 (~11% of 10-K words are Item 1A but carry the idiosyncratic-risk signal); Magner et al. 2025 best results from Item 1A only. Dilution argument is sound. |
| **10-Q excluded from NLP** | ✅ **Holds** | No mandatory Item 1A; quarterly "risk factors" are largely "no material change" boilerplate. No precedent for 10-Q in risk-embedding pre-training. |
| **Form 4 → scalar feature in Stage 5** | ⚠️ **Holds, but revise the *why*** | Correct to exclude from NLP (purely tabular, no text). **But** the cited evidence (Cohen-Malloy-Pomorski 2012; Goldie et al. 2023) is *abnormal return* — a **returns** signal. For the **volatility** target, reframe: insider trading **intensity/dispersion** and **opportunistic-vs-routine** activity proxy information asymmetry → which predicts volatility, not the alpha sign. |
| **GDELT → scalar feature in Stage 5** | 🧪 **Pilot now — cheap & decisive** | The 1-year pull is exactly the right pilot. (1) Test it *within the covered window* (§3.5 Pilot A/B), not on the main split. (2) Switch the feature emphasis: the mail leans on `avg_tone` (a **returns** signal, Boudoukh 2019), but for volatility use **news *volume*/attention** (`n_articles`, `n_sources`; Da-Engelberg-Gao 2011) and **tone *dispersion***. **Decision rule:** positive pilot → authorise BigQuery 2015–2024 + integrate (head re-fit); null pilot → drop. |
| **Earnings calls → DAPT corpus + scalar** | 🧪 **Strategic bet — can't be settled by a 1-yr pilot** | Defensible (Fin-E5, FinGPT). **But** the *NLP-training* value (DAPT corpus) needs scale — it cannot be measured from 299 transcripts. Only a thin **scalar** pilot is possible on the ~400 labelled 2025 filings (§3.5). So this is a deliberate bet on **precedent + willingness to pull 2008–2024 and re-train**, sequenced **after** the core is proven — not a cheap-pilot decision. |
| **4th cross-document contrastive view (claimed novelty)** | 🧪 **Strategic bet — sequence after core** | Needs aligned Item-1A↔call pairs in the **training** period → same historic-calls dependency. **Re-training the contrastive stage with a 4th view is on the table**, but: View 3 (Sector) already *hurt* full FT, and the project's novelty (hybrid Topic + DAPT + contrastive) **does not depend on it**. So: prove the 3-view core beats 0.18/0.60 first, then add the 4th view as a measured upgrade — not before. |
| **Item 7A → conditional DAPT corpus expansion** | 🟡 **Pilot cheaply on the 400, then decide full extraction** | Lowest-friction expansion: a **scalar pilot** (does Item-7A text add over Item-1A on the 400 labelled 2025 filings?) is doable now from `sec/`. If positive, extract Item 7A for the **full history** from EDGAR 10-Ks (collectable, same source as `sp500_1A`) and **re-run DAPT** on the 1A+7A corpus (Jehnen 2026 precedent). Marginal-gain-uncertain but cheap to test. |

---

## 3. Per-dataset decision table

Verdicts are **provisional-pending-pilot** where a pilot exists; "Full-integration path" assumes the pilot is
positive and **includes re-training** where relevant.

| Dataset | Provisional decision | Pilot (what would prove it pays) | Full-integration path if positive (incl. re-training) |
|---|---|---|---|
| **Item 1A** (`sp500_1A`) | ✅ **Keep — core (locked)** | — (it *is* the core) | Already in all phases |
| **`feature_table.parquet`** | ✅ **Keep — spine** | — | New signals attach as *columns* (shared contract, §5b) |
| **Form 4 insider txns** | ✅ **Keep — scalar (full coverage)** | Marginal R²/Spearman over lagged-vol on train+val (it covers 2006–26, so pilots on the **main split**) | Stage-5 scalar: pre-filing 30/60/90d windows; net officer/director P-vs-S, **trade dispersion/count**, opportunistic-vs-routine (Cohen-Malloy 2012) |
| **GDELT** | 🧪 **Pilot now (cheap, decisive)** | §3.5 Pilot A (2025 daily panel: news volume/dispersion → fwd vol) + Pilot B (400 filings, within-2025 CV) | Positive → **BigQuery 2015–2024** pull → Stage-5 scalar (volume/dispersion), head re-fit |
| **Earnings calls** | 🧪 **Strategic bet (post-core)** | Only a *thin scalar* pilot on the ~400 (call/Q&A tone marginal R²) — can't measure DAPT value at 1 yr | Pull Finstream **2008–2024** → **re-run DAPT** on 1A+calls corpus + call-tone scalar + (optional) 4th view |
| **4th contrastive view** | 🧪 **Strategic bet (post-core)** | Not pilotable at 1 yr (needs aligned train-period pairs) | **Re-train contrastive** with View 4 (Item-1A↔call, TF-IDF matched, cross-sector hard negs) once calls pulled |
| **Item 7A** | 🟡 **Pilot cheaply, then decide** | Scalar pilot on the 400: Item-7A text marginal R² over Item-1A | Positive → extract Item 7A full-history from EDGAR → **re-run DAPT** on 1A+7A (Jehnen 2026) |
| **10-Q / full 10-K body** | ❌ **Excluded from NLP (HOLDS)** | — | Full-10-K text may still *source* Item 7A |
| **`crsp_dsenames`, `ccm`, FINSABER, CRSP-2025** | ✅ **Keep — infrastructure** | — | `dsenames` temporal name bridge is essential for any name-matched source (GDELT) |

---

## 3.5 Pilot experiments — how the 1-year data earns (or loses) its place

This is the operational heart of the memo. The 1-year collections exist *to be piloted*. But there is a
crucial **asymmetry** in what a 1-year pilot can decide:

> **GDELT is cheaply and *decisively* pilotable. Earnings calls / Item 7A / the 4th view are *not* — their
> value is in NLP-training, which needs scale a 1-year pilot can't supply.** Treat them differently.

### A. GDELT — run the decisive pilot now (no historical pull needed to decide)

**Pilot A — daily panel (strongest, filing-independent).** For 2025 (where CRSP-2025 prices exist), build a
per-`(permno, date)` panel: target = next-30-trading-day realised vol; features = GDELT **`n_articles`,
`n_sources`, tone dispersion** (+ trailing realised vol as control). Hundreds of PERMNOs × ~220 days = ample
n. Fit a panel regression / gradient-boost; report incremental R² and rank-IC of the news features over the
vol-only control.
*This answers the fundamental question — does company news carry forward-vol signal at all? — independent of
the filing pipeline.*

**Pilot B — pipeline-shaped (smaller, directly comparable).** On the **400 labelled 2025 filings**, append
pre-filing GDELT window features to the Tier-1 lagged-vol baseline; evaluate by **k-fold CV within 2025**
(can't train on 2006–24 where GDELT is absent — that's Rule 1). Report marginal R²_log / Spearman vs the
lagged-only model. Small-n — treat as corroboration of Pilot A, not standalone.

**Decision rule.** Clear incremental signal in A (corroborated by B) → **authorise the BigQuery 2015–2024
pull**, then integrate as a Stage-5 scalar with the head re-fit on the extended span. Null result → **drop
GDELT** from the supervised model with evidence, not assumption. Either way the supervisor gets a *data-backed*
recommendation. Use **volume/dispersion** features, not mean tone.

### B. Earnings calls & Item 7A — what the 1-year pilot can and cannot tell you

- **Cheap scalar pilot (possible now):** on the ~400 filings, does a call-tone / Q&A-tone feature (or Item-7A
  text features) add marginal vol R² over Item-1A + lagged? This is a **weak, small-n signal check** — useful
  if strongly positive, inconclusive if flat.
- **The real value (NLP-training) is *not* pilotable at 1 year.** DAPT-corpus expansion and the 4th contrastive
  view need the 2008–2024 scale to show any effect; 299 transcripts can't move a domain-adaptation metric.
- **Therefore decide these as strategic bets:** literature precedent (Jehnen 2026 for 7A; Fin-E5/FinGPT for
  calls) **+ willingness to pull the historical corpus and re-train**, sequenced **after** the core hybrid is
  shown to beat 0.18/0.60. Item 7A is the cheaper bet (EDGAR-collectable, no server dependency); earnings
  calls depend on the Finstream historic pull Prof Ma flagged for the server's sake.

### C. Sequencing
1. Finish Phase-5 scoring of the current encoders + topics vs the 0.18/0.60 floor (in progress).
2. Run **GDELT Pilot A/B** and the **Form-4** marginal-value ablation (both cheap, data on hand).
3. From results, decide: GDELT BigQuery pull? Item-7A full extraction + DAPT re-run? earnings-call pull + 4th
   view? — spending re-training budget on whatever shows the **largest measured marginal value**.

---

## 4. Design choices that actually move the score ("beat the literature")

The decisive metric is Phase-5 volatility prediction vs **(a)** the TF-IDF+lagged floor (0.18 R² / 0.60 ρ)
and **(b)** published vol-forecasting numbers. Levers, in priority order:

1. **Finish scoring the domain encoders + topic vectors (in progress).** This is the actual experiment. If
   `three_lora` / topic hybrid clears 0.18/0.60, the thesis is validated; if not, that is the real problem to
   solve — *before* any new dataset.
2. **Best encoder = `three_lora`** (best on both FinMTEB axes). If retraining, **fix the sector-view noise**
   (38% of sector pairs have TF-IDF cosine < 0.15) by filtering low-similarity pairs — the known lever behind
   View-3's full-FT damage.
3. **Topic axis must earn its place:** the domain BERTopic backend has to beat the SBERT Cᵥ baseline (0.716),
   or the interpretable axis adds nothing over the generic one.
4. **Hybrid feature vector** `[mean-pooled encoder | topic exposure | log_lagged | (Form 4 scalars)]`, with
   `log_lagged` as the persistence anchor (it lifted TF-IDF from −0.03 to 0.18 — it will anchor every tier).
5. **Volatility-specific framing for every scalar** (Rule 2): attention/volume and dispersion, not tone-mean.
6. **Re-training is an available lever** — re-do DAPT on a 1A+7A (or 1A+calls) corpus, re-do contrastive with
   a 4th view — but **core-first sequenced**: confirm the current 3-view + topic hybrid beats 0.18/0.60, then
   spend the re-training budget on whatever a pilot (§3.5) ranks highest. Re-training is justified by measured
   marginal value, not assumed.
7. **Temporal-coverage gating as a pilot-design rule** (Rule 1): a 1-year signal is piloted within its window;
   only a train-period-covering feature joins the main supervised head. Learned from GDELT.
8. **Build a literature target table** — the specific published Item-1A / financial-text vol-forecasting
   numbers we intend to beat — so "outperformed prior work" is a concrete, checkable claim, not a vibe.

---

## 5. Talking points mapped to Prof Ma's four pre-meeting items

**(a) Data description / summary / statistics** — from `feature_table.parquet` (authoritative, not the readme
estimates):

| Metric | Value |
|---|---|
| Filings with Item 1A text | **8,105** |
| Firms (unique CIK / ticker) | 482 / 485 |
| PERMNO assigned | 8,079 (99.7%) |
| Both vol labels | 7,695 (94.9%) — **but train-dominated** |
| **fwd_vol by split** | train 7,320/7,353 · val 400/406 · **test 1/346** |
| SIC-2 sectors | 53 |
| Filing-date range | 2006-02-22 → 2026-03-02 |
| Split | train ≤2024 · val 2025 · test 2026 |

> ⚠️ **Flag honestly:** the 94.9% "both labels" figure hides that **2026 test forward-vol is essentially
> unlabelled (1/346)** — no 2026 price feed yet. Interim Phase-5 eval is **val 2025**. (Reconcile minor
> count drift vs readme: 7,695 vs ~7,721 vol-labels, 8,105 vs 8,007 paragraph-bearing filings — counting
> rules differ, not errors.)

**(b) Collaborative group work / dataset integration** — propose a **shared feature-table contract**: one
row per `(cik, fiscal_year)` keyed also by `permno`, into which each collaborator's signal slots as a
*column* (text embedding, topic vector, Form-4 scalars, …). Standardise on join keys `cik / permno /
fiscal_year`; use `crsp_dsenames` (`namedt`/`nameendt`) as the **temporal name bridge** for any name-matched
source so everyone resolves names identically; Sean Choi's PERMNO-linkage verification is the shared
integrity check. This avoids redundant collection and guarantees signals are mergeable.

**(c) Initial technical framework + implementation plan** — present the **core-first sequence**: (1) finish
Phase-5 scoring of encoders + topics vs the floor; (2) decide add-ons by measured marginal R², gated by
Rule 1; (3) Form 4 first (full coverage, low risk); (4) earnings-calls / 4th-view / GDELT-historical only if
core results justify the cost. Emphasise the temporal-coverage gate and leakage discipline (fit on train
only) as framework invariants.

**(d) Updated literature summary** — reframe around the **volatility** target: attention/volume as vol
predictors (Da-Engelberg-Gao 2011), insider info-asymmetry → vol, Item-1A-only precedents (Campbell 2014,
Magner 2025), DAPT precedents (Gururangan 2020, FinGPT, Fin-E5), plus the **target table** of published
numbers to beat.

---

## 6. Open questions to raise with the supervisor

1. **GDELT:** approve running the cheap 1-year pilot (§3.5 A/B) to *decide with data* — and pre-authorise the
   BigQuery 2015–2024 pull **conditional on a positive pilot**? (The 1-year collection was always meant to
   answer exactly this.)
2. **Earnings calls / 4th view:** these can't be settled by a 1-year pilot (NLP-training needs scale). Approve
   the Finstream 2008–2024 pull + contrastive re-training as a **strategic bet** now, or hold until the core
   hybrid is shown to beat the floor? Recommend: hold (core-first), then pull if core results justify it.
3. **Item 7A:** approve the cheap scalar pilot on the 400; if positive, EDGAR full-history extraction + DAPT
   re-run. Lowest-friction expansion (no server dependency).
4. **Confirm val-2025 as the interim held-out set** until CRSP-2026 prices allow 2026 test labelling.
5. **Form 4 scope:** confirm volatility-framed feature set (intensity/dispersion) rather than the
   returns-alpha framing of the original mail.

---

### One-line summary
The original collection plan is well-reasoned, and the **1-year pulls were a sound pilot strategy** — they
just predate the results. The evidence says: **prove the core hybrid beats 0.18/0.60 first**, **pilot GDELT
cheaply now** (it can be decided with data on hand) and add it if positive, fold in **Form 4** (the one
fully-covered signal), and treat **earnings-calls / Item-7A / 4th-view** as *strategic, re-training-backed
bets* sequenced after the core — each gated by measured marginal value, not assumption.
