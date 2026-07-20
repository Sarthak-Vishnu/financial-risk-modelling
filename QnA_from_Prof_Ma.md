# Q&A — questions from Prof Ma

Each question is answered twice: first **as of the IPP-report pipeline** (the original design:
text-only lanes, single train-before-2025 / evaluate-on-2025 split, best model TF-IDF + lagged
volatility under a ridge head, R²_log ≈ 0.18 / Spearman ≈ 0.60, count-based features beating the
contrastive and sentence-transformer encoders), and then **under the extended pipeline** (the
structured baseline, the full encoder grid, the expanding-window backtests 2013–2024, the
clean-protocol encoder retrain, and the earnings-call study). Where a question can only be
settled by an experiment that has not been run, that is stated explicitly.

---

## Q1. The count-based model gave better results. Are those results consistent across different horizons — 30 days, 20 days, …?

**As of the IPP pipeline.** The horizon question was not tested. The label is a single fixed
horizon — realised volatility over the 30 trading days following the filing date
(`dataset_config/compute_volatility_labels.py`, `WINDOW = 30`) — and every comparison in the
report, including "TF-IDF beats the encoders", was made at that one horizon. So the honest
IPP-era answer is: *unknown empirically; 30 trading days was a design choice* (it matches the
monthly convention of the realised-volatility literature and gives every filing a label that
closes within the sample).

**Under the extended pipeline.** Still a single 30-day label, but two things sharpen the prior:

1. *The advantage of the count model is a representation result, not a horizon result.* The
   clean-protocol retrain (study_extended.md, Part III.3) shows the dense encoders lose because
   their text-to-volatility mapping is era-specific and drifts, while the TF-IDF lane survives
   because an annually refit linear head can re-weight transparent lexical levels each year.
   That mechanism has nothing to do with the label window, so the ranking
   (count model ≥ encoders) is expected to hold at 20, 60 or 90 days.
2. *The size of the text increment, by contrast, plausibly does depend on horizon.* At shorter
   horizons volatility persistence strengthens, so the structured block (which contains realised
   volatility at 21/63/126/252-day lookbacks, a HAR-style structure) should dominate even more
   and the text increment should shrink. At longer horizons persistence decays and the
   slow-moving annual disclosure signal should matter relatively more — consistent with Kogan
   et al. (2009) and Campbell et al. (2014), who find 10-K text informative at up to annual
   horizons. The study already contains one precedent for exactly this kind of
   horizon-dependence: earnings-call tone adds +0.039 IC at the call date but nothing at the
   filing date (Part V, the horizon-contrast finding).

**Verdict — now measured (job 3559526, 2026-07-20).** The experiment described above was run:
labels regenerated at 20/60/90 trading days (`compute_horizon_labels.py`, with a built-in
certification that the reimplementation reproduces the study's frozen 30-day labels exactly,
corr = 1.000000), and the fair pair rerun per horizon on the 2018–2024 backtest and val-2025.

| horizon | lagged IC | structured [ridge] | struct+tfidf | text ΔIC (paired) | p |
|---|---|---|---|---|---|
| 20d | 0.401 | 0.513 | 0.532 | **+0.020** | 0.119 |
| 30d | 0.461 | 0.546 | 0.561 | **+0.016** | 0.098 |
| 60d | 0.528 | 0.592 | 0.591 | −0.001 | 0.851 |
| 90d | 0.534 | 0.607 | 0.602 | −0.005 | 0.477 |

Two answers, one per layer of the question. (1) **The model ranking is consistent across
horizons**: struct+tfidf ≥ structured ≥ lagged at every horizon on the backtest, and val-2025
agrees (struct+tfidf 0.637 / 0.610 / 0.718 / 0.751 at 20/30/60/90 days). Nothing overtakes the
count-based model at any horizon. (2) **The size of the text increment is horizon-dependent** —
it decays monotonically from +0.020 at 20 days to zero at 60–90 days. Notably, this is the
*opposite* direction from the prior stated in point 2 above: overall predictability *rises* with
horizon (lagged IC 0.401 → 0.534) because longer realised-vol windows are smoother and more
persistent — and that persistent component is exactly what the structured block already carries.
What text contributes is the transient, near-filing component of uncertainty, which washes out
of the target as the horizon lengthens. See Q2 for the interpretation.

---

## Q2. If the results are *not* consistent across horizons — then what?

Inconsistency would not invalidate the 30-day result; it would bound its scope and become a
finding in its own right: the **term structure of the text increment**. The reporting plan would
be:

- Report the paired ΔIC (struct+tfidf minus structured) per horizon, with the same
  paired-by-year tests used everywhere else in the study. The deliverable is a curve —
  "text adds x IC at 20 days, y at 30, z at 90" — not a binary.
- Interpret it through the same lens as the earnings-call result: a text signal is valuable at
  the horizons where the market has not yet impounded it into cheaper quantitative predictors,
  and absorbed at the horizons where it has. The study already demonstrates this pattern for a
  second modality, so a horizon-dependent 10-K increment would corroborate rather than
  contradict the framework.
- The headline claims would then be stated *at their horizon*: "at the 30-day monthly horizon,
  text adds a small consistent increment" — which is already how the study phrases them.

If instead the *representation ranking* flipped at some horizon (an encoder beating TF-IDF at,
say, 90 days), that would be genuinely new information and would justify re-opening the encoder
comparison at that horizon — but the drift mechanism identified in Part III.3 gives no reason to
expect it.

**Now measured (see the Q1 table): the "not consistent" branch is the one that obtained, and it
resolved exactly as this reporting plan anticipated.** The deliverable curve is
+0.020 → +0.016 → −0.001 → −0.005 across 20/30/60/90 days: the text increment is a
short-horizon phenomenon, present at the monthly horizons (20–30 days) and fully absorbed by
60 days. The interpretation follows the framework above, with one refinement the data supplied:
the increment does not fade because the market impounds the text over calendar time — it fades
because longer-horizon realised volatility is increasingly dominated by the slow persistent
component that the structured block (HAR-style trailing realised-vol features) already captures,
leaving the transient near-filing uncertainty as the only part text can add, and that part
matters only in short windows. The headline claims were already stated at their horizon ("at the
30-day monthly horizon…") and now carry measured support on both sides: the increment is real
where claimed and demonstrably absent at 60–90 days. The representation ranking did **not** flip
at any horizon, so the encoder comparison stays closed.

---

## Q3. If consistent — is using an (open-source) LLM better than TF-IDF?

**As of the IPP pipeline.** The report could only say: the two encoders tried (the
similarity-trained contrastive encoder and off-the-shelf SBERT) sat below TF-IDF, with the open
caveat that stronger or task-supervised encoders had not been tried.

**Under the extended pipeline.** This question is now answered about as thoroughly as it can be
at the encoder scale, and the answer is **no — no LLM-derived dense representation beats TF-IDF
on this task under any admissible protocol** (Part II.3 and III.3):

- Modern general-purpose embedders included: `bge` (BAAI bge-base-en-v1.5) loses too
  (IC 0.580–0.584 vs 0.603), which settles the "you never tried a strong modern embedder"
  objection.
- Task-supervised open-source encoders were built (volaware, ftvol — contrastive on
  volatility deciles and end-to-end fine-tuned on the regression target). They reach
  *statistical parity* in-period, never superiority.
- Under the clean temporal protocol (train before 2017, freeze, backtest 2018–2024) the
  supervised encoder collapses out-of-period: −0.06 IC against struct+tfidf, statistically zero
  against the structured features alone.
- The kitchen-sink fusion of every representation at once also does not beat the count model.
- This direction has independent support: the FinMTEB benchmark (Su et al. 2025) reports
  bag-of-words outperforming dense embedding models on financial semantic tasks.

The positive characterisation matters more than the null: **task alignment, not model modernity
or scale, is what closes the gap — and forward transfer erases it**, because the dense mapping
is era-specific while annually refit lexical weights are not.

**Scope caveat.** "LLM" above means encoder-class models (~110M parameters) used as embedders.
Two LLM uses remain untested: (a) decoder-scale open models (7B+) as embedders, and (b) LLMs as
*feature generators* (see Q5, where this is identified as a future research direction beyond the
scope of the present study). Given the drift mechanism, the expectation for (a) is parity at
best, but it has not been measured.

---

## Q4. Is lookahead bias better or not?

Lookahead bias reliably makes *measured* results better and *real* results worse — it
manufactures apparent skill that a deployed model would not have. It is a validity threat, never
a legitimate design choice, and the extended pipeline both removes it and, usefully, **measures
how much it is worth**:

- The original ftvol encoder (trained on all pre-2025 labels, epoch-selected on the evaluation
  year itself) appears at parity with the count model in-period (IC 0.606). The clean-protocol
  retrain of the same architecture — strict pre-2017 training cutoff, epoch selection on 2017,
  checkpoint frozen — loses by ~0.06 IC out-of-period. That gap, roughly **0.06 IC of apparent
  skill, is the measured price of the lookahead** in this setting: several times larger than
  the genuine text increment (+0.011 to +0.016).
- The controls now in place: forward-window labels anchored strictly after the filing date;
  features computed only from data available at the filing date; expanding-window walk-forward
  evaluation (Lopez de Prado 2018, chs. 7, 11–12) instead of k-fold on time series; an
  **admissibility audit** of every encoder's training provenance before it may enter a backtest
  (representation-level lookahead is still lookahead, even when the downstream head is trained
  cleanly); and for earnings calls, only the most recent call *strictly before* each filing is
  used.

So the answer to "is it better": it flatters the numbers and invalidates the claim. Every
headline result of the study is quoted from the lookahead-free protocol.

---

## Q5. If we use an LLM for *feature extraction* instead of reasoning/prediction — how would financial-specific models compare to general models?

**What is already tested.** The entire encoder grid *is* LLM feature extraction: frozen
embeddings fed as features into the same heads as every other block. On the domain question the
pipeline holds direct evidence:

- *General-domain* extractors (SBERT, BGE): IC 0.562–0.591, at or below the no-text structured
  baseline.
- *Domain-adapted* extractor: an MLM domain-adaptively pretrained on the 10-K corpus was built
  for the topic pipeline; its embedding space is strongly anisotropic — a documented property
  of MLM representation spaces (Ethayarajh 2019) — and did not yield usable geometry for
  downstream clustering, so it was excluded from the final backends. Domain-adaptive
  pretraining, on this corpus, did not convert into a better feature space.
- *Task-aligned* extractors (volaware, ftvol): parity in-period, collapse out-of-period.

**The prediction for FinBERT-class models.** A financial-specific encoder sits between "general"
and "task-aligned": domain vocabulary without task supervision. The evidence says the domain
vocabulary is not the binding constraint — TF-IDF already captures financial-vocabulary
information essentially perfectly, at zero cost, in a form an annually refit head can re-weight —
and that the only thing that ever closed the gap was supervision on the volatility target
itself. The expectation is therefore that a FinBERT-class extractor lands with the general
encoders (parity at best with the structured baseline, below struct+tfidf), and FinMTEB's
finding that bag-of-words beats dense models *on financial text specifically* points the same
way. This is a well-grounded expectation, not yet a measured result.

**The genuinely untested variant.** Using a generative LLM as a *feature generator* — prompting
an open model (general: e.g. Llama/Qwen-class; financial: e.g. a finance-tuned variant) to score
each filing on interpretable risk dimensions (litigation exposure, leverage concern, demand
uncertainty, …) and feeding those scores as dense features. This differs from embeddings in that
the output is low-dimensional and interpretable, so it may in principle resist the drift problem
the same way the lexicon-based tone features do. Whether interpretable LLM-elicited features can
surpass count-based representations — and whether financial-domain tuning changes that answer —
is, however, a research question of considerable depth in its own right: a credible treatment
would require principled prompt-schema design, calibration and stability analysis across model
versions, inference over the full filing corpus for at least two model families, and the same
multi-year admissibility discipline this study applies to encoders. It is therefore identified
here as a substantive direction for future research rather than as an extension of the present
study, whose scope and time constraints it exceeds.

---

## Summary of experimental status

| question | answerable from existing results? | new experiment needed |
|---|---|---|
| Q1 horizon consistency | **yes — measured (2026-07-20)**: ranking consistent at every horizon; increment size is not | no — done |
| Q2 if inconsistent | **answered — the term structure IS the finding**: +0.020/+0.016/−0.001/−0.005 at 20/30/60/90d | no — done |
| Q3 LLM vs TF-IDF | **yes — answered (no, at encoder scale)** | optional: decoder-scale embedder |
| Q4 lookahead bias | **yes — answered and quantified (~0.06 IC)** | no |
| Q5 financial vs general LLM extractor | partially (embedding route tested; domain pretraining did not help) | identified as a future research direction — beyond the scope of the present study |

---

## Position note — GDELT GKG 1.0 (for discussion)

Both extension datasets raised earlier were assessed for integration. Form 4 was integrated in
full (Stage E: complete feature build, paired ladder rows on both backtest windows, and the
pre-registered text-increment conditioning analysis). For GDELT GKG 1.0 the assessment concluded
that a sound integration is not achievable within the remaining timeline, on two independent
grounds.

**1. Entity resolution is a project-sized prerequisite, and the study's own audit history shows
why it cannot be done hastily.** GKG 1.0 identifies firms only as free-text organisation names —
there is no CIK and no ticker. Linking those names to the panel therefore requires a temporally
valid name-to-firm bridge: corporate renames and restructurings must resolve to the right entity
in the right period (Google → Alphabet, Facebook → Meta), subsidiaries must roll up correctly,
and firms whose names are common English words (Target, Apple, Visa, Gap) need genuine
disambiguation, because naive string matching floods them with false positives. Every downstream
number would inherit the quality of this mapping, and the earlier audit phase of this study
demonstrated precisely how much a subtle join defect can propagate before it is visible. A
defensible entity-resolution layer — built, validated, and audited to the standard applied
everywhere else in this study — is weeks of dedicated work before the first experimental row
exists.

**2. The coverage window is incomplete relative to the rest of the study.** Every other data
source in the study — filings, structured market features, earnings-call transcripts, Form 4
transactions — spans 2006–2026. GKG 1.0 begins in April 2013, and with trailing pre-filing
feature windows its usable coverage starts around 2014. This forecloses the twelve-year
(2013–2024) backtest that every other result in the study reports alongside the seven-year
window, so any GDELT finding would rest on a strictly weaker evidential basis than the results it
would sit next to, and the cross-stage comparability that the study's evaluation protocol is
built on would not hold for it.

The scientific design itself remains on the record: `study_extended.md` (Part VII, open question
5) specifies news coverage as the natural independent conditioner for replicating the Stage E
state-dependence finding. The GDELT study is therefore positioned as specified future work whose
data-engineering prerequisite — the entity-resolution layer — exceeds the scope and time of the
present study, rather than as an open experimental gap.
