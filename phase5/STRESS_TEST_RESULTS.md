# TF-IDF Stress-Test — results log

**Question.** Is "count-based TF-IDF is the best text representation" really the final answer, or can a
learned/multimodal model beat it? All numbers below are on the **P0-corrected panel**
(`fix_filing_join.py`: filename year = FILING year; unique fiscal keys). Pre-fix numbers
(struct 0.570 / struct+tfidf 0.611) paired texts with the next filing's labels and are not comparable.

## Headline findings so far (2026-07-02, CPU experiments complete; encoder rows await GPU encode)

1. **The join fix changed 74% of text↔label pairs** but left the TF-IDF headline almost unchanged
   (struct+tfidf 0.610 val-2025) — fresh text ≈ year-stale text, i.e. Item-1A signal is persistent
   boilerplate. The error was conservative (no leakage), and is now repaired.
2. **Head asymmetry inflated the old "text increment."** `structured [ridge]` (0.591 val / 0.603
   12-yr backtest) is much stronger than the `structured [hgb]` the ladder used (0.558 / 0.584).
   Same-head comparison: text adds **+0.012 IC on val-2025** and **+0.011 IC (p=0.057) on the 12-yr
   backtest** — consistent in sign across years, but ~5× smaller than the head-confounded +0.05.
   On R²_log the text gain remains substantial (0.175 → 0.226 val-2025).
3. **α-tuning is noise at this scale:** tuning the sparse-ridge α on 2024 *hurt* 2025
   (tfidf+lag 0.508 tuned vs 0.541 at fixed α=10) — model-selection variance swamps these deltas,
   reinforcing that only paired multi-year tests count.
4. **Lexical change features don't add** (struct+change ≤ structured everywhere). The semantic
   (`chg_enc_cos_*`) and shape (`chg_new_para_frac`) columns need the GPU encode — still open.
   The features themselves are valid: clear COVID dip (median chg_lex_cos 0.964 in FY2020 vs ~0.976).

## Anchors (val-2025, n=393, corrected panel)

| condition | IC | R²_log |
|---|---|---|
| lagged [hgb] | 0.457 | −0.130 |
| structured [hgb] | 0.558 | 0.094 |
| **structured [ridge]** | **0.591** | 0.175 |
| tfidf+lag [sparse α=10] | 0.541 | 0.116 |
| struct+tfidf [sparse] | 0.603–0.610 | 0.220–0.226 |

(n≈393 ⇒ single-year IC has ±0.06 CI; val-2025 deltas are indicative only.)

### Sparse-ridge penalty provenance — read this before comparing any two sparse rows

Two ranges above (`0.541` beside `0.508`, and `0.603–0.610`) are the same condition measured under
two different penalties. Neither is an error, and neither should be reconciled by changing a
number.

`E.TextNumericRidge` takes `alpha=10.0` as its **class default**. Whether a lane is tuned depends
entirely on how it is constructed:

| path | penalty | why |
|---|---|---|
| `stress_grid.py --mode val2025`, via `fit_sparse(..., tune=True)` | **tuned** over `SPARSE_ALPHAS = (3.0, 10.0, 30.0)`, selected on 2024; picks **30** | the fair grid tunes every head it can |
| `stress_grid.py --mode backtest`, the `conds` dict in `run_backtest` | **fixed α=10** | constructs `TextNumericRidge` directly; `fit_sparse` is never called and `SPARSE_ALPHAS` is never consulted |
| `run_horizon.py`, `run_fusion.py`, `call_filing_gate.py` | **fixed α=10** | same — direct construction |

So the two ranges resolve as:

- **`tfidf+lag [sparse]`: 0.541 fixed vs 0.508 tuned on val-2025.** The tuned figure is the one the
  fair grid reports, and the one the write-up quotes. Tuning *hurt* here, which is finding 3 above.
- **`struct+tfidf [sparse]`: 0.610 fixed vs 0.603 tuned on val-2025.** The 0.610 is what
  `run_horizon.py` reports as its H=30 anchor, because run_horizon constructs the head directly and
  therefore inherits α=10. The **0.603 is the one to publish**: every other row in the val-2025 grid
  and every DM p-value in it is computed against that prediction vector, so quoting 0.610 beside
  them would compare a tuned grid to a fixed-α row. Preferring 0.610 would also mean selecting on
  the evaluation year, and the ordering reverses on R²_log anyway (α=10 gives 0.2197, α=30 gives
  0.2260), so it is not simply the better model. 0.610 is correct inside run_horizon's own
  internally consistent α=10 world and should not be changed there.

**Backtest rows are uniformly fixed α=10 on every sparse lane** — `tfidf+lag`, `struct+tfidf` and
`struct+tfidf+change` alike. Every paired ΔIC in the backtest tables therefore compares like with
like, and none of them is contaminated. The residue is cross-table only: the same condition name
denotes a tuned model in the val-2025 grid and a fixed-α model in the backtest tables.

**On leaving the value implicit.** α=10 is inherited from the class default rather than passed at
these call sites, and that is deliberate. Making it explicit would change no behaviour, and it
would break the byte-identity of `stress_grid.py` against the version that produced the committed
`out/stress_grid_val2025_fixed.json` — which is the strongest provenance claim the grid has. This
note exists so that nobody later "fixes" the apparent inconsistency by adding tuning to the
backtest lanes: doing so would silently invalidate every published backtest comparison.

## E3 — backtests (primary lens)

| condition | 2018–2024 IC (t) | 2013–2024 IC (t) |
|---|---|---|
| lagged [hgb] | 0.456 (4.6) | 0.526 (8.3) |
| structured [hgb] | 0.511 (5.0) | 0.584 (9.0) |
| **structured [ridge]** | **0.546 (5.6)** | **0.603 (9.9)** |
| tfidf+lag [sparse] | 0.499 (5.2) | 0.558 (9.3) |
| struct+tfidf [sparse] | 0.561 (5.9) | 0.614 (10.4) |
| struct+change [hgb] (full clean cols) | 0.492 (4.8) | 0.575 (8.7) |
| struct+tfidf+change [sparse] (full clean cols) | 0.550 (5.9) | 0.606 (10.5) |

(change lanes re-run 2026-07-04 with the complete semantic+shape columns, minus the
label-trained-encoder cols `chg_enc_cos_volaware/ftvol`, which would leak test-year label
information through the encoder into a pre-2025 backtest.)

Paired across-years vs **structured [ridge]** (the fair baseline):

| condition | 7-yr ΔIC (p) | 12-yr ΔIC (p) |
|---|---|---|
| struct+tfidf | +0.016 (0.098) | **+0.011 (0.057)** |
| struct+tfidf+change (full clean cols) | +0.005 (—) | +0.004 (—) |
| structured [hgb] | −0.034 (0.152) | −0.019 (0.180) |

**Semantic-change verdict (E2 closed, 2026-07-04):** with the complete clean change set
(lexical + `chg_enc_cos_dual/sbert/bge` + `chg_new_para_frac`), struct+tfidf+change sits *below*
struct+tfidf on both windows (−0.011 p=0.37 / −0.007 p=0.28 paired) and struct+change sits below
structured. Disclosure-change features — lexical or semantic — add no incremental volatility
signal over the level TF-IDF representation. The Lazy-Prices-extension branch is null.

The struct+change row's own paired test against structured [hgb], from `paired_vs_structured` in
the two backtest JSONs (previously quoted as a difference of means with no p-value):

| lane | 7-yr ΔIC | p | 12-yr ΔIC | p |
|---|---|---|---|---|
| struct+change [hgb] vs structured [hgb] | −0.0197 | 0.2302 | −0.0094 | 0.3369 |

Negative on both windows, neither significant — the same verdict the ΔIC alone implied, now with
the test behind it.

vs the weaker structured [hgb]: struct+tfidf +0.050 (p=0.114) 7-yr / +0.030 (p=0.113) 12-yr —
the old-style comparison, kept for reference. structured [hgb] vs lagged: +0.059 (p=0.007) 12-yr —
the strong-baseline claim is solid.

Baseline-vs-persistence, both heads (paired across years, recomputed from the stored per-year ICs
in `out/stress_grid_backtest_{2018,2013}_fixed.json`). The write-up defends **structured [ridge]**,
so that is the row it quotes; the [hgb] row above is the same comparison on the weaker head and is
kept only because the JSON's `paired_vs_structured` block is keyed on it:

| baseline vs lagged [hgb] | 7-yr ΔIC | p | 12-yr ΔIC | p |
|---|---|---|---|---|
| structured [ridge] | +0.0891 | 0.0177 | **+0.0776** | **0.0009** |
| structured [hgb] | +0.0549 | 0.1118 | +0.0586 | 0.0074 |

Note the seven-year [hgb] cell is *not* significant (p=0.112); only the twelve-year one is. Quoting
+0.059 with p=0.007 therefore pins the window to twelve years as well as the head to [hgb].

## E1 — val-2025 fair grid (encoder rows pending GPU encode)

| row | IC [95% CI] | R² | DM p vs struct+tfidf |
|---|---|---|---|
| struct+tfidf [sparse] (ref) | 0.603 [0.534, 0.670] | 0.226 | — |
| structured [ridge] | 0.591 [0.522, 0.659] | 0.175 | 0.000 |
| structured [hgb] | 0.558 [0.487, 0.628] | 0.094 | 0.001 |
| struct+tfidf_svd [ridge] | 0.594 [0.526, 0.661] | 0.167 | 0.000 |
| struct+tfidf_svd [hgb] | 0.569 [0.499, 0.639] | 0.167 | 0.123 |
| struct+change [hgb] | 0.551 [0.477, 0.624] | 0.110 | 0.004 |
| struct+tfidf+change [sparse] | 0.605 [0.537, 0.671] | 0.223 | 0.697 |
| struct+enc[dual] [ridge] — **full text** | 0.582 [0.509, 0.652] | 0.162 | 0.013 |
| struct+enc[dual] [hgb] | 0.582 [0.511, 0.651] | 0.160 | 0.051 |
| struct+enc[sbert] [ridge] | 0.562 [0.482, 0.636] | 0.017 | 0.000 |
| struct+enc[sbert] [hgb] | 0.591 [0.521, 0.658] | 0.149 | 0.043 |
| struct+enc[volaware] [ridge] | 0.587 [0.514, 0.656] | 0.147 | 0.003 |
| **struct+enc[volaware] [hgb]** | **0.597 [0.530, 0.663]** | **0.207** | 0.593 |
| **struct+enc[ftvol] [ridge]** | **0.606 [0.540, 0.672]** | 0.185 | 0.207 |
| struct+enc[ftvol] [hgb] | 0.598 [0.529, 0.666] | 0.167 | 0.155 |
| struct+enc[bge] [ridge] | 0.580 [0.506, 0.649] | 0.093 | 0.000 |
| struct+enc[bge] [hgb] | 0.584 [0.512, 0.651] | 0.145 | 0.023 |
| struct+enc[ftvol,risk_weighted] [hgb] | 0.604 [0.538, 0.672] | 0.185 | 0.324 |
| **struct+enc[ftvol,topk_risk] [hgb]** | **0.607 [0.538, 0.677]** | 0.194 | 0.454 |
| struct+enc[volaware,{risk_weighted,topk_risk}] [hgb] | 0.586–0.593 | 0.180–0.203 | ≥0.165 |
| pooling variants (risk_weighted/topk_risk × dual/sbert, earlier run) | 0.573–0.578 | 0.145–0.172 | ≤0.128 |
| struct+change [hgb] (full semantic+shape cols) | 0.557 [0.484, 0.630] | 0.084 | 0.001 |
| struct+tfidf+change [sparse] | 0.609 [0.542, 0.672] | 0.219 | 0.355 |
| **EVERYTHING svd+enc[ftvol]+chg [hgb]** | **0.608 [0.542, 0.675]** | 0.170 | 0.204 |
| EVERYTHING tfidf+enc[ftvol]+chg [sparse] | 0.590 [0.525, 0.662] | 0.165 | 0.135 |

**Encoder verdict, full grid (fair conditions — full text via windowing, corrected labels, both heads):**

1. **Generic semantic encoders lose.** dual/sbert/bge — including bge, the modern general-purpose
   embedder (the last "you never tried X" objection) — sit at or below the no-text structured
   [ridge] baseline (0.591) and are DM-significantly less accurate than struct+tfidf. Fixing the
   65% truncation did NOT rescue them.
2. **Task-aligned encoders reach parity — the only ones that do.** ftvol (supervised vol
   fine-tune) 0.606/0.607, volaware (vol-aware contrastive) 0.597 with the second-best R² (0.207).
   Clear pattern: what closes the gap is not model modernity but *training on the volatility task*.
   All differences vs the TF-IDF reference are DM-insignificant → **statistical tie, not a win**.
3. **Admissibility caveats on the parity rows (audit 2026-07-03):**
   - `finetune_vol.py` selected its best epoch by **val-2025 filing IC** → ftvol's val-2025 rows
     are inflated by model selection on the eval set.
   - Both ftvol and volaware were trained with volatility labels (regression target / within-year
     vol-decile pairing) from **all pre-2025 filings** → a 2013/2018-start backtest with the
     current checkpoints is **inadmissible** (encoder saw test-year labels), so the pre-registered
     "backtest lane before claiming anything" cannot run on these checkpoints.
   - Both were also trained on the **pre-fix** (P0-a broken) labels/corpus — a headwind, not a tail
     wind, so parity survived a handicap. A clean retrain could go either way.

**Decisive experiment (queued): clean-protocol retrain.** Fine-tune ftvol on filings < 2018 ONLY
(fixed corpus + fixed labels, epoch selection on the last train year, post-cutoff years never
touched), re-encode full-text, then a legitimate 2018–2024 expanding-window backtest with paired
DM tests vs struct+tfidf. This either produces "task-supervised encoder beats the count model"
with airtight protocol, or upgrades the tie/loss to a defensible final verdict.

DM p is on squared-error loss: the sparse TF-IDF reference is significantly *more accurate* (R²) than
every non-TF-IDF row so far — the lexical block earns its place on level accuracy, beyond ranking.

## FINAL VERDICT (2026-07-04, job 3527629 — all experiments complete)

**The clean-protocol supervised encoder collapses out-of-period.** ftvol2018 (train < 2017 on the
corrected corpus, epoch selected on 2017 at filing IC 0.635, frozen, full-text windowed encode)
on the admissible 2018–2024 expanding-window backtest:

| lane | mean IC | paired vs struct+tfidf | paired vs structured [hgb] |
|---|---|---|---|
| struct+enc[ftvol2018] [ridge] | 0.504 | −0.057 (p=0.087) | −0.007 (p=0.83) |
| struct+enc[ftvol2018] [hgb] | 0.497 | −0.064 (p=0.097) | −0.014 (p=0.64) |
| struct+tfidf [sparse] (ref) | 0.561 | — | +0.050 (p=0.11) |

Under the honest protocol the task-supervised embeddings add **nothing over structured features
alone** (Δ≈0 vs structured) and lose to the count model by ~0.06 IC. The in-period strength was
real (0.635 filing IC on the selection year) — it just does not transfer forward: the original
ftvol's val-2025 "parity" came from training on all pre-2025 labels + epoch selection on the eval
year. Combined with the val-2025 grid: **no encoder beats struct+tfidf under any admissible
protocol.** Decision-tree branch: the "all null" row — with two characterization upgrades:

1. **Task alignment closes the in-period gap, forward transfer erases it.** Generic encoders
   (dual/sbert/bge) lose everywhere; vol-supervised encoders tie only when their training window
   covers the evaluation era. The text→vol signal a dense encoder learns is era-specific, while
   TF-IDF's lexical level features + cheap annual refit stay current by construction. (One fair
   caveat: the frozen encoder faces a 1–8-year staleness handicap that the annually-refit TF-IDF
   lanes do not; per-window encoder retraining — 12 GPU fine-tunes — would remove it, but the
   deploy-realistic single data point of that kind, original-ftvol→2025, still only ties.)
2. **The benchmark is now defended, not just asserted:** struct+tfidf 0.614 IC (12-yr, t=10.4),
   R² 0.226 val-2025; text increment over the fair structured[ridge] baseline +0.011 (p=0.057)
   with a large accuracy (DM) edge; every challenger — heads, SVD, five encoders, three poolings,
   topics, change features (lexical + semantic), EVERYTHING fusions — beaten or tied by it under
   fair, leakage-audited conditions.

## Decision tree → conclusion (every branch a positive headline)

| Outcome | Headline |
|---|---|
| A learned combo beats struct+tfidf (backtest, paired p<0.05) | Learned/multimodal model beats the count-based ceiling. |
| EVERYTHING > struct+tfidf | Dense/topic signal is orthogonal & additive to bag-of-words. |
| Semantic change adds where lexical change doesn't | Novel disclosure-change signal for vol forecasting (Lazy-Prices extension). |
| All null | A strong structured-ridge baseline (IC 0.603, 12-yr) + a small but consistent lexical text increment (+0.011, p=0.057) + full head/representation/pooling grid *characterizing* the text signal as lexical-level — a sharp, honest contribution. |

## Stage C — earnings-call tone: filing-level result (2026-07-18, complete)

**Data (single source).** Full transcript history landed: the Finstream corpus (39,766 rows →
39,501 unique calls, 637 firms, 2005-10→2025-03; re-hosted at HF
`SarthakVishnu/dissertation-dataset`, `calls/SP500_calls_2006to2025.parquet`).
`build_call_features.py` (CIK-primary join, ticker fallback, most recent call strictly before the
filing, ≤200d): **7,159/8,105 filings matched (88.3%)**, ≥95% in every backtest year 2013–2024,
390/406 in val-2025. The 299 API Ninjas JSONs are deliberately excluded from the feature build
(they remain the call-anchored pilot's data): a two-source-vs-parquet-only diff confirmed their
only unique matches are 14 filings dated 2026 — untouched by any evaluation — with **zero**
feature values changed on 2006–2025 rows, so every number below is single-source.

**1) Filing-level gate, val-2025 (`call_filing_gate.py`; 5-fold CV within the 2025 cross-section,
identical folds per comparison, same head per pair):**

| pair (same head) | full panel (n=393) dIC [95% CI] | matched only (n=379) dIC | DM p |
|---|---|---|---|
| structured ± tone [hgb] | −0.017 [−0.034, +0.001] | −0.007 | 0.022 / 0.066 |
| structured ± tone [ridge] | +0.000 [−0.011, +0.012] | −0.013 | 0.677 / 0.062 |
| struct+tfidf ± tone [ridge] | −0.000 [−0.012, +0.011] | −0.012 | 0.539 / 0.061 |

Every CI straddles or sits below zero; the only significant DM test favours the **no-tone** model.

**2) Backtest 2018–2024 (`run_fusion.py`, calls now a leakage-free family — 6,845/7,367 panel
filings covered on the single-source build (the run printed 6,859 pre-cleanup; the 14 extra were
2026 filings, which never enter training or evaluation, so all metrics are unaffected); paired
across-year test vs the same-head counterpart):**

| lane | window | mean IC | paired dIC | p |
|---|---|---|---|---|
| struct+calls [hgb] vs structured [hgb] | 2018–2024 | 0.515 vs 0.518 | −0.004 | 0.395 |
| struct+tfidf+calls [ridge] vs struct+tfidf [ridge] | 2018–2024 | 0.561 vs 0.561 | −0.001 | 0.436 |
| struct+calls [hgb] vs structured [hgb] | 2013–2024 | 0.588 vs 0.589 | −0.001 | 0.698 |
| struct+tfidf+calls [ridge] vs struct+tfidf [ridge] | 2013–2024 | 0.613 vs 0.614 | −0.000 | 0.515 |

The twelve-year rows are transcribed from `logs/run_fusion_form4.log:93-95`. `run_fusion.py` emits
the call lanes unconditionally, so the `RISK_TEST_START=2013` pass in the Stage E job produced them
as a side effect; they were never read at the time because `run_fusion.py` printed a hardcoded
"BACKTEST 2018-2024" header over the twelve-year section (since fixed). The dIC column is the
paired across-year mean, which is why it does not always equal the difference of the two rounded
means beside it.

val-2025 rows agree: struct+calls 0.559 (vs 0.556), struct+tfidf+calls 0.611 (vs 0.610) — noise.

**Verdict.** Call tone adds nothing at the filing anchor, in 2025 and across all seven backtest
years — while the call-anchored pilot (+0.039 IC over structured predicting post-*call* 30d vol,
n=152, `call_combined_gate.py`) stands. The two results are not in tension: they measure different
horizons. A plausible reading (interpretation, not established mechanism): the tone signal is real
at the call date but is absorbed by filing time — the structured block at the filing date includes
realised-vol windows spanning the post-call period, so whatever tone predicted has already been
realised into prices. Stage C therefore closes as a **horizon-contrast finding**: management tone
carries volatility information at the call horizon, and the conditions under which it stops adding
value are precisely the ones the 10-K prediction task imposes (a later anchor with fresher
structured information). The null holds on **both** backtest windows: −0.004 (p=0.395) and −0.001
(p=0.436) over 2018–2024, −0.001 (p=0.698) and −0.000 (p=0.515) over 2013–2024. Calls therefore sit
on the same two-window footing as Stage E's Form 4 lanes, and coverage was never the constraint —
tone matches ≥96.5% of panel rows in every year 2013–2024, falling off only before 2013 (2011
82.0%, 2010 87.9%, 2007 34.9%), and those years enter training only.

## Stage E — Form 4 insider features × risk text (2026-07-19, complete)

**Data & features.** `form4_transactions.csv` (SEC insider-transactions structured flat files,
1.67M non-derivative rows, 645 S&P 500 issuers, 2006–2025; re-hosted at HF
`SarthakVishnu/dissertation-dataset`, `form4/`). `build_form4_features.py`: per filing, eight
`f4_*` intensity/dispersion features over the trailing window (filing−180d, filing−3d] — the 3-day
margin covers Form 4's 2-business-day disclosure lag, so every feature is public at prediction
time. CIK-primary join, ticker fallback. Coverage is effectively total: **7,367/7,367 panel
filings in the Form 4 universe** (8,039/8,105 filings with window activity), every year 2006–2026
— so unlike calls, Stage E rows are valid in *both* backtest windows. The mildly negative median
`f4_abn_intensity` in every year is the pre-10-K blackout period showing up in the data (insiders
trade less just before the filing than in their trailing-365d baseline) — construct validity, not
an artifact. (~10 keyed-date-error rows in the raw CSV span 1990–2031; none can fall in a valid
window.)

**1) Level effect (paired ladder rows, `run_fusion.py`, both windows):** null everywhere.

| lane (same head) | 7-yr mean IC | dIC | p | 12-yr mean IC | dIC | p |
|---|---|---|---|---|---|---|
| struct+form4 [hgb] vs structured [hgb] | 0.519 vs 0.518 | +0.000 | 0.897 | 0.591 vs 0.589 | +0.001 | 0.768 |
| struct+tfidf+form4 vs struct+tfidf [ridge] | 0.562 vs 0.561 | +0.000 | 0.886 | 0.613 vs 0.614 | −0.001 | 0.585 |

val-2025 agrees: struct+form4 0.551 (vs structured 0.556), struct+tfidf+form4 0.610 (vs 0.610).
Insider-activity levels are already spanned by the structured block — consistent with the
structured features themselves (realised vol, drawdown, turnover) reflecting whatever elevated
insider activity accompanies.

**2) Text-increment conditioning (`form4_text_conditioning.py` — the pre-registered question:
does the text increment concentrate where insider activity is high?)** Anchors reproduced inside
the script (unconditional +0.016 p=0.098 / +0.011 p=0.057). Within-year median splits of the test
cross-section (strict split; for `f4_disagreement`, whose within-year median is 0, this contrasts
"any insider disagreement" — ~22% of firms — against none):

| conditioner | window | mean(high−low ΔIC) | t | p | sign consistency |
|---|---|---|---|---|---|
| f4_abn_intensity | 7-yr | −0.012 (*withdrawn*) | −2.67 | 0.037 | negative 6/7 years |
| f4_abn_intensity | 12-yr | −0.006 | −1.28 | 0.227 | negative 7/12 |
| f4_disagreement | 7-yr | **+0.029** | +2.04 | 0.088 | positive 6/7 years |
| f4_disagreement | 12-yr | +0.016 | +1.51 | 0.158 | positive 8/12 |

> **Update 2026-08-05 — `f4_abn_intensity` withdrawn.** The window sweep (spectrum section below)
> re-ran this test at 3/5/7/10/20/60/90 days. The intensity effect is significant only in the 30d
> cell and is null at all seven others (|ΔIC| ≤ 0.013, sign flipping positive at both ends), so it
> is a lone-window result and is no longer reported as a finding. `f4_disagreement` survives, with
> the 30d cell corroborated by its neighbour at 20d (+0.029). Everything below that rests on the
> *contrast* between the two conditioners should be read as resting on disagreement alone.

**Verdict.** The pre-registered direction (larger text increment in the high half) holds for
**disagreement** and is reversed for **abnormal intensity** — and the two results are more
coherent together than either alone. A plausible reading (interpretation, not established
mechanism): the two conditioners proxy different stages of the information process. Abnormal
trading *intensity* is itself a transmission channel — Form 4s are public within two business
days, so where insiders have traded heavily, their information is already flowing into prices and
the disclosure text has less left to add (the low-intensity half is where the text increment
lives: ΔIC_low > ΔIC_high in 6/7 recent years). Insider *disagreement*, by contrast, marks
unresolved dispersion among the best-informed parties — precisely the state in which narrative
disclosure would still carry incremental information, and there the text increment is 3–4× its
unconditional size (e.g. +0.083 in 2021, +0.132 in 2023 within the disagreement half). This
extends the Stage C absorption narrative from the time dimension (call vs filing anchor) to the
cross-section (which *firms* text helps on). Bounds on the claim, stated plainly: two conditioners
× two windows with no multiplicity adjustment, p-values 0.037–0.227, both effects attenuate as the
window extends back to 2013 (the pattern is concentrated in 2018+), and the disagreement high-cell
is small (52–89 firms/year). Stage E therefore closes as: **no level effect, plus suggestive,
directionally consistent evidence that the text increment is state-dependent** — largest where
informed parties disagree. *(As of 2026-08-05 the intensity arm of that contrast is withdrawn per
the note above; the state-dependence claim now rests on disagreement alone, which the window sweep
corroborates at 20d as well as 30d.)*

## Volatility-window spectrum — Prof Ma Q1/Q2 + meeting items 1/3/4 (2026-08-05, complete)

Supersedes the four-window result of 2026-07-20 below, whose numbers are reproduced here
unchanged. At the 2026-08-04 meeting Prof Ma asked for the short end of the curve (1/3/5/7 days),
for the whole spectrum from 3 to 90 days presented in comparison, for the same window angle
applied to the earnings-call study, and for the Form 4 and calls extensions to be compared. All
four are one sweep, because `eval_common.load_panel()` already selects labels through
`RISK_HORIZON` and every downstream script reads labels through it.

**Why the spectrum starts at 3, not 1.** The label is `std(daily log returns, ddof=1) × √252`,
undefined for a single observation. Answering H=1 would mean swapping in a different estimator
(|r|·√252) partway along the curve; one estimator across the whole spectrum is worth more than one
extra point on it. At H=1 the target also stops being a dispersion measure and becomes a single
signed-magnitude return — a different construct, not a shorter version of the same one.

**Gates, all passed before any new number was read.** Label build self-certification at H=30:
corr = 1.000000 on both columns, max |diff| 3.9e-6. Coverage monotone in horizon on every filing
year (True). H=30 anchor reproduction in all three swept scripts: `run_horizon.py` 0.461 / 0.546 /
0.561, ΔIC +0.016 p=0.098, val-2025 0.591 & 0.610; `form4_text_conditioning.py` +0.016 p=0.098,
disagreement +0.030 p=0.089, abn_intensity −0.012 p=0.037; `call_combined_gate.py` at
`RISK_CALL_WINDOW=30` tone +0.039. Construct checks: `fwd_vol_3d` vs `fwd_vol_90d` correlation
0.517 (against 0.877 for 20 vs 90 — the short and long labels are genuinely different targets);
fiscal-2020 median forward vol elevated at every window, peaking at 20d (0.850) and decaying to
0.649 by 90d.

### 1. The 10-K text increment across the spectrum (item 1)

| window | lagged IC | struct [ridge] | struct+tfidf | text ΔIC | p | ΔIC / struct | val-2025 struct / s+tfidf |
|---|---|---|---|---|---|---|---|
| 3d | 0.193 | 0.330 | 0.351 | **+0.021** | **0.020** | +6.4% | 0.366 / 0.386 |
| 5d | 0.289 | 0.417 | 0.441 | **+0.023** | **0.017** | +5.5% | 0.514 / 0.523 |
| 7d | 0.299 | 0.444 | 0.470 | **+0.027** | **0.014** | +6.1% | 0.564 / 0.582 |
| 10d | 0.314 | 0.462 | 0.482 | +0.020 | 0.076 | +4.3% | 0.578 / 0.596 |
| 20d | 0.401 | 0.513 | 0.532 | +0.020 | 0.119 | +3.9% | 0.638 / 0.637 |
| 30d | 0.461 | 0.546 | 0.561 | +0.016 | 0.098 | +2.9% | 0.591 / 0.610 |
| 60d | 0.528 | 0.592 | 0.591 | −0.001 | 0.851 | −0.2% | 0.709 / 0.718 |
| 90d | 0.534 | 0.607 | 0.602 | −0.005 | 0.477 | −0.8% | 0.739 / 0.751 |

**Verdict — three findings.**
1. **The model ranking is window-robust.** struct+tfidf ≥ structured ≥ lagged at every one of the
   eight windows on the backtest; val-2025 agrees everywhere. Nothing overtakes the count model at
   any window, so the headline representation result is not an artifact of the 30-day design
   choice. This is the direct answer to Q1 and it is now measured over eight points, not four.
2. **The text increment is single-peaked at 7 days** and decays monotonically to zero by 60,
   +0.021 → +0.023 → **+0.027** → +0.020 → +0.020 → +0.016 → −0.001 → −0.005. In relative terms
   the shape is sharper still (last-but-one column): text is worth 6% of the structured IC at the
   short end and −1% at the long end. The direction is the *opposite* of the pre-run prior (QnA Q1
   predicted text mattering more at long horizons). The mechanism is visible in the first column:
   overall predictability rises with window length (lagged IC 0.193 → 0.534; longer realised-vol
   windows are smoother and dominated by the persistent component), and the persistent component
   is exactly what the structured HAR-style block already carries. Text contributes the transient
   near-filing component, which washes out of the target as the window lengthens.
3. **The increment clears p < 0.05 at 3, 5 and 7 days** — the first time anywhere in the study.
   VII.4 open question 3 asked whether the increment crosses 0.05 as evaluation years accumulate;
   the answer turns out to be that it crosses by *shortening the window* instead. Mechanically the
   short windows are far less exposed to the 2020 regime break: IC_t rises from 5.9 at H=30 to
   11.7 at H=3. **Stated with its bound**: eight windows, uncorrected, so p=0.014 at H=7 is not a
   0.05-level claim on its own. The defensible claim is the monotone shape of the curve, which no
   multiplicity argument touches, with the significance at the short end as corroboration.

### 2. Earnings calls across the spectrum (item 3)

**(a) Filing anchor** — the well-powered lane (6,859/7,367 filings matched, 7 backtest years).
Each pair is same-head: `struct+calls [hgb]` vs `structured [hgb]`, `struct+tfidf+calls` vs
`struct+tfidf [sparse]`.

| window | struct+calls ΔIC | p | struct+tfidf+calls ΔIC | p |
|---|---|---|---|---|
| 3d | **+0.011** | 0.079 | −0.001 | 0.818 |
| 5d | +0.006 | 0.100 | +0.000 | 0.661 |
| 7d | −0.003 | 0.376 | −0.000 | 0.989 |
| 10d | −0.001 | 0.781 | −0.000 | 0.892 |
| 20d | −0.006 | **0.027** | −0.001 | 0.078 |
| 30d | −0.005 | 0.289 | −0.001 | 0.372 |
| 60d | +0.002 | 0.739 | −0.000 | 0.821 |
| 90d | −0.003 | 0.350 | +0.000 | 0.741 |

Stage C's filing-anchor null is **not** a 30-day artifact — it is a null at every window. The only
non-null cells run in opposite directions (a weak positive flicker at 3–5 days, a small but
significant *negative* at 20 days), which is what an uncorrected eight-window sweep of a true zero
looks like. And on top of the full text model calls add nothing anywhere: |ΔIC| ≤ 0.001 at all
eight windows. Where tone flickers at the short end, 10-K TF-IDF has already captured it — the two
text modalities are substitutes, not complements.

**(b) Call anchor** — the pilot lane, n=152 (n=139 at 60/90d, where the window runs past the data
end), single 2025 regime, leave-one-quarter-out. Reported as a pilot; the shape is jagged, as
expected at this sample size.

| window | 3d | 5d | 7d | 10d | 20d | 30d | 60d | 90d |
|---|---|---|---|---|---|---|---|---|
| tone ΔIC over structured | −0.034 | +0.079 | +0.084 | **+0.142** | **+0.134** | +0.039 | −0.028 | −0.025 |
| call-text ΔIC | +0.008 | +0.004 | +0.014 | +0.014 | +0.004 | +0.011 | +0.007 | +0.004 |

Tone at the call anchor is a **hump peaking at 10–20 days**, positive across 5–30 and negative at
3, 60 and 90. The published +0.039 at 30d turns out to sit on the decaying right shoulder of that
hump rather than at its peak. Call-text embeddings add ~+0.01 at every window, i.e. nothing that
depends on the window.

### 3. Form 4 conditioning across the spectrum (item 4 input)

`form4_text_conditioning.py` swept under `RISK_HORIZON` with no code change, so both extensions now
sit on the same window axis. Unconditional column reproduces the item-1 table exactly (same test).

| window | uncond ΔIC | p | `f4_disagreement` high−low | p | `f4_abn_intensity` high−low | p |
|---|---|---|---|---|---|---|
| 3d | +0.021 | 0.020 | **−0.038** | 0.075 | +0.013 | 0.445 |
| 5d | +0.023 | 0.017 | −0.011 | 0.316 | −0.007 | 0.531 |
| 7d | +0.027 | 0.014 | −0.002 | 0.900 | −0.009 | 0.442 |
| 10d | +0.020 | 0.076 | +0.002 | 0.894 | −0.008 | 0.268 |
| 20d | +0.020 | 0.119 | **+0.029** | 0.135 | −0.005 | 0.575 |
| 30d | +0.016 | 0.098 | **+0.030** | 0.089 | **−0.012** | **0.037** |
| 60d | −0.001 | 0.851 | +0.001 | 0.919 | −0.001 | 0.820 |
| 90d | −0.005 | 0.477 | +0.002 | 0.801 | +0.002 | 0.574 |

Two consequences, one supportive and one corrective.

**`f4_disagreement` survives and acquires a shape.** The effect is a medium-window phenomenon,
+0.029 at 20d and +0.030 at 30d, decaying to zero by 60 and *reversing* to −0.038 at 3 days. Two
adjacent windows agreeing is real corroboration that the 30d cell is not isolated. The sign flip at
the short end is not noise-shaped either — it says the disagreement conditioning and the
unconditional text increment peak at *different* windows, which is the substance of item 4 below.

**`f4_abn_intensity` is downgraded and should no longer be reported as a finding.** It was the only
p < 0.05 cell in the Stage E conditioning table (−0.012, p=0.037 at 30d). Across the other seven
windows, in ascending order 3/5/7/10/20/60/90d, it is +0.013, −0.007, −0.009, −0.008, −0.005,
−0.001, +0.002 — none significant, none larger than 0.013, and the sign flips positive at both
ends. No neighbouring window supports it.
A single significant cell in a family that is null at every adjacent window is the signature of a
lone-window fluke, and the window sweep is exactly the test that distinguishes the two cases. The
interpretation previously attached to it (abnormal intensity as an already-firing transmission
channel, so text has less left to add) is therefore withdrawn as unsupported; it may still be true,
but this study no longer has evidence for it. The Stage E verdict rests on `f4_disagreement` alone.

*This is what the window sweep was worth beyond answering the question asked: it retired one
result and corroborated another, at no additional data cost.*

### Reproducibility caveat on the HGB lanes

The `structured [hgb]` level reproduced as 0.511 in this run against 0.518 published on 2026-07-18,
and `struct+calls [hgb]` correspondingly 0.507 vs 0.515 (so the H=30 call ΔIC is −0.005 p=0.289
here vs −0.004 p=0.395 published). Investigated and not resolved: all input file mtimes predate 18
July; `make_hgb` is unchanged since creation (`git log -L`); sklearn 1.8.0 was installed 2026-05-19,
before the July run; HGB is deterministic within a process (identical predictions, `n_iter_`=299 on
both fits); and re-running `run_fusion.py`'s own lane directly also gives 0.511. **Only HGB lanes
moved — every ridge and TF-IDF lane reproduced exactly**, and the paired ΔIC, which is what the
lanes exist to produce, is essentially unchanged (−0.005 vs −0.004). No conclusion depends on the
level. Recorded here rather than silently absorbed.

---

## Horizon robustness — Prof Ma Q1/Q2 (2026-07-20, superseded by the spectrum above)

**Question.** Is the count-model result (and the text increment) consistent across label
horizons, or specific to the 30-trading-day label? Labels regenerated at 20/60/90 trading days
(`dataset_config/compute_horizon_labels.py`; same definition and price sources as the frozen 30d
labels, plus a staleness guard requiring the window to touch the filing date within 10 calendar
days — no label rather than a stale one where a firm's price history ends). The build
self-certifies: recomputed H=30 vs the stored labels, corr = 1.000000 on both columns,
max |diff| ≈ 3e-6 (the 6dp rounding). The 30d labels and `feature_table_fixed.parquet` were never
modified; new horizons live in `datasets/volatility_labels_horizons.parquet`, selected at runtime
via `RISK_HORIZON` (`eval_common.load_panel`; default 30 = byte-identical legacy path). Fair pair
per horizon (`phase5/run_horizon.py`): lagged persistence, structured [ridge], struct+tfidf
[sparse], horizon-matched log_lag throughout; 2018–2024 backtest + val-2025. The H=30 lane ran
first off the frozen labels and reproduced every anchor exactly (0.561 / +0.016 p=0.098 /
val-2025 0.591 & 0.610) before any new horizon was read.

| horizon | panel | lagged IC | struct [ridge] | struct+tfidf | text ΔIC | p | val-2025 struct / s+tfidf |
|---|---|---|---|---|---|---|---|
| 20d | 7,366 | 0.401 | 0.513 | 0.532 | **+0.020** | 0.119 | 0.638 / 0.637 |
| 30d | 7,367 | 0.461 | 0.546 | 0.561 | **+0.016** | 0.098 | 0.591 / 0.610 |
| 60d | 7,364 | 0.528 | 0.592 | 0.591 | −0.001 | 0.851 | 0.709 / 0.718 |
| 90d | 7,359 | 0.534 | 0.607 | 0.602 | −0.005 | 0.477 | 0.739 / 0.751 |

**Verdict — two findings.**
1. **The model ranking is horizon-robust.** struct+tfidf ≥ structured ≥ lagged at every horizon
   on the backtest; val-2025 agrees everywhere. Nothing overtakes the count model at any horizon,
   so the headline representation result is not an artifact of the 30-day design choice.
2. **The text increment has a term structure** — +0.020 → +0.016 → −0.001 → −0.005 across
   20/30/60/90 days. Direction is the *opposite* of the pre-run prior (QnA Q1 predicted text
   mattering more at long horizons): overall predictability rises with horizon (lagged IC
   0.401 → 0.534; longer realised-vol windows are smoother and more persistent), and that
   persistent component is exactly what the structured HAR-style block already carries. Reading
   (hedged, consistent with Stages C/E): text contributes the transient near-filing component of
   uncertainty, which washes out of the target as the horizon lengthens — the study's 30-day
   headline sits at the horizon where the text signal lives, and the increment is demonstrably
   absent by 60 days. No individual ΔIC is significant (p 0.098–0.851); the monotone pattern
   across four horizons, not any single cell, is the evidence. Label construct validity: COVID
   fiscal-2020 median fwd vol decays with horizon (0.85 → 0.65, mean reversion); cross-horizon
   label correlations 0.88–0.98.

## Form 4 vs earnings calls — the matched comparison (item 4, 2026-08-05)

Prof Ma asked for the two extensions to be compared on results *and* interpretation. The window
sweep makes the results half quantitative rather than rhetorical, because both extensions are now
measured on the same axis.

| axis | earnings calls (Stage C) | Form 4 (Stage E) |
|---|---|---|
| what it proxies | management's own read of the quarter, at the moment they voice it | information asymmetry among the best-informed parties |
| anchor | signal at the call date, null at the filing date | filing date only |
| coverage | 6,859/7,367 filings (93%), 2006–2025 | 7,367/7,367 (100%), 2006–2026 |
| backtestable windows | 7-yr only (coverage) | both 7-yr and 12-yr |
| level effect | null at the filing anchor, at all 8 windows | null on all four arms, both windows |
| conditional / anchored effect | +0.039 IC at the call anchor at 30d, n=152 | +0.030 disagreement conditioning at 30d |
| **window signature** | **hump, peak 10–20d** (+0.142 / +0.134) | **rise to 20–30d** (+0.029 / +0.030), sign-flips to −0.038 at 3d |
| relation to the 10-K text | substitute — abs(ΔIC) ≤ 0.001 on top of TF-IDF at every window | complement — conditions *where* the TF-IDF increment is earned |
| status | informative at its own anchor, absorbed by the filing anchor | null in level, state-dependent in the cross-section |

**The result the comparison produces, which neither study produced alone.** The three effects have
*different* window signatures, and they do not line up:

| effect | peak window | value at 3d |
|---|---|---|
| 10-K text increment (unconditional) | **7d** (+0.027) | +0.021 — near its maximum |
| insider-disagreement conditioning | **20–30d** (+0.030) | −0.038 — sign reversed |
| call tone at the call anchor | **10–20d** (+0.142) | −0.034 — sign reversed |

The 10-K's own text carries **days**. Both event families carry **weeks**. That is a cleaner split
than the pre-sweep picture suggested, and it refines the absorption reading rather than repeating
it: the filing text prices the transient, immediately-post-filing uncertainty and is spent within
a fortnight, whereas what insiders and management convey shows up over a horizon long enough for
the market to work through it — and is invisible, or inverted, in the first three days.

**Interpretation, stated as interpretation.** Both extensions test the same hypothesis — that the
10-K text increment exists only where and when the market has not already impounded the
information — along different dimensions. Calls test the **time** dimension: tone is informative at
the call, and null one to three months later at the filing, because the structured block's
realised-volatility windows measured at filing span exactly the post-call period the tone
predicted. Form 4 tests the **cross-section**: the increment is largest on firms whose insiders are
split, i.e. whose information is demonstrably unresolved. The window spectrum adds a third
dimension, **the target's own timescale**, and it is what makes these a matched pair rather than
two appendices. It also gives the two studies a shared falsifiable prediction: if the mechanism
were simply "more text signal is better", all three curves would peak together. They do not.

**Where the comparison is asymmetric, and should be reported as such.** The calls result at the
call anchor rests on n=152 in a single 2025 regime and its window curve is correspondingly jagged;
the Form 4 result rests on 7 backtest years at 100% coverage. The two are not equally credentialled
and the write-up should not present them as such — the calls *filing-anchor* null is the
well-powered calls finding (6,859 filings × 7 years × 8 windows, null throughout), and it is that
null, not the call-anchored hump, that carries weight against the Form 4 conditioning result.

## Round 8 — bounded generative-LLM pilot (2026-08-05, job 3584249, MEASURED NULL)

**Question (Prof Ma, 2026-08-04).** Give the event dataset to an LLM, have it interpret and score,
compare with the manual model — does it improve the results? Run as a bounded pilot on
`exp/llm-event-scoring` with a stopping rule fixed before the scores were looked at.

**Design.** Form 4 pre-filing windows (the same (filing−180d, filing−3d] window as the eight
`f4_*` features) rendered as a compact transaction table and scored by Qwen2.5-7B-Instruct on three
0–100 dimensions (volatility_risk, information_asymmetry, confidence), K=5 samples at temperature
0.7. Scope is **2024+2025 filings only (865 prompts, 836 joined to the panel)** — both years
post-date the model's training cutoff, so look-ahead is controlled by construction rather than by
argument. Prompts are **identifier-stripped** (no ticker, no company or insider names, no absolute
dates — day offsets relative to the filing; insiders pseudonymised I1, I2, …), with a per-row
anonymisation assertion; an **identified control lane** was scored separately as the contamination
probe. Evaluation is 5-fold CV within year, identical folds and one head
(`E.make_imputed_ridge`) across all three conditions — the same admissible design
`call_filing_gate.py` uses, and for the same reason: the LLM score exists only on the evaluation
rows, so no train<year split could learn it. 2×`h200_1g.18gb`, 1h40m wall.

**1) Score stability — the measurement, and the most durable output.** Parse failures **0/4,325
samples (0.00%)**, so nothing here is a parsing artifact.

| dimension | across-filing sd | median within-filing sd (5 re-runs) | ratio |
|---|---|---|---|
| volatility_risk | 12.47 | 7.36 | 0.59 |
| information_asymmetry | 11.08 | 9.75 | 0.88 |
| confidence | 7.91 | 7.58 | **0.96** |

Re-running the *identical* prompt on the *identical* filing moves the score 59–96% as much as
swapping in a different company. On the confidence dimension the elicited score is very nearly
pure noise. This is study_extended.md VII.5's stability claim measured on this study's own data
instead of borrowed from Sclar et al. (2024) and Ouyang et al. (2024) — and it is measured under
conditions favourable to the model (fixed prompt, fixed model version, fixed seed, one release).
Averaging the 5 draws recovers almost nothing: single draw IC 0.733, K-averaged 0.736, gain +0.003.

**2) Does the LLM score beat the hand-crafted features?** No — and neither beats the baseline.

| condition | IC | R²_log |
|---|---|---|
| structured [ridge] | 0.736 | 0.532 |
| structured + f4_manual (the 8 `f4_*` features) | 0.730 | 0.527 |
| structured + f4_llm (3 LLM scores) | 0.736 | 0.532 |

| paired comparison | ΔIC | 95% CI | DM p |
|---|---|---|---|
| f4_manual vs structured | −0.006 | [−0.014, +0.000] | 0.167 |
| f4_llm vs structured | −0.000 | [−0.005, +0.004] | 0.920 |
| **f4_llm vs f4_manual** | **+0.006** | **[−0.000, +0.013]** | 0.118 |

**STOPPING RULE: ΔIC(llm vs manual) = +0.006, CI half-width 0.007 → MEASURED NULL.** The pilot
closes here per its pre-registered rule: no expansion to the full panel, no earnings-call lane, no
second model family. (Note the IC level of 0.736 is not comparable to the 0.546 backtest figure —
CV-within-year trains on contemporaneous filings and is a much easier design. It is the *paired
differences* that are the output.)

**3) Contamination probe (anonymised vs identified prompts).** identified IC 0.737, anonymised
0.736, premium **+0.001, 95% CI [−0.004, +0.006]**, n=836. Framed precisely: on a post-cutoff slice
there *should* be no look-ahead to find, so this is **a placebo check that passed**, not a general
measurement of contamination. It confirms the scope decision worked; it does not license
generalising to the 2018–2024 backtest years, where the whole objection lives.

**Verdict.** Feature engineering was **not** the bottleneck. The eight hand-crafted features and the
generative model's reading of the same raw record are statistically indistinguishable, and both are
indistinguishable from omitting insider information entirely — which is exactly what Stage E's level
test already found on 7 and 12 backtest years. The binding constraint is the information content of
the Form 4 record relative to a structured block that already carries realised volatility, drawdown
and liquidity. The pilot's value is therefore (a) the instability measurement above, which
strengthens VII.5's position with a number rather than a citation, and (b) a defensible answer to
the question actually asked. It is written up as the bounded answer to QnA Q5 and is not expanded.

## Run log

- 2026-08-05 — job 3584156 (Teaching, 2 CPU, 6h58m): **volatility-window spectrum complete**
  (meeting items 1/3/4). Label build at 3/5/7/10/20/60/90 (certification corr 1.000000 both
  columns, coverage monotone in horizon on every filing year), then the H=30 anchor gate across
  `run_horizon.py` + `form4_text_conditioning.py`, then all eight windows through `run_horizon.py`
  (now carrying two added call lanes), `form4_text_conditioning.py`, and `call_combined_gate.py`
  under `RISK_CALL_WINDOW`. Findings in the spectrum section above. Two results changed status:
  the text increment clears p<0.05 at 3/5/7 days (first time in the study), and
  `f4_abn_intensity` is withdrawn as a lone-window fluke. HGB level reproducibility caveat
  recorded above.
- 2026-08-05 — job 3584249 (`h200_1g.18gb` ×2, 1h40m): **Round 8 generative-LLM pilot** on branch
  `exp/llm-event-scoring`. 865 Form 4 windows × 5 samples × 2 lanes (anonymised + identified
  control) through Qwen2.5-7B-Instruct, 0 parse failures out of 8,650 samples, then
  `llm_vs_manual.py`. MEASURED NULL by its own stopping rule; the within-filing score spread is
  the keeper. (Three earlier submits failed on scheduling and sizing: 3584212/3584218 pended
  because a 48h request cannot backfill ahead of a higher-priority hold on `saxa`; 3584220 OOMed
  inside `from_pretrained` because a `1g.18gb` slice exposes 16.00 GiB, not 18, against a 15.2 GiB
  bf16 model plus a transient fp32 upcast of the 152k×3584 embedding. Fixed with 2 slices, a 2h
  wall request, and `set -e` so a scoring failure can no longer fall through to the evaluation
  step and bury the traceback.)
- 2026-07-20 — job 3559526 (Teaching, 2 CPU, 34m): horizon experiment complete — label build
  (certification corr 1.000000 both columns), H=30 anchor reproduction exact, then H=20/60/90.
  Findings above; QnA_from_Prof_Ma.md Q1/Q2 updated to measured answers. (First submit 3559518
  was stopped by the build's own certification gate: two ETN rows — a ticker absent from CRSP
  2025 — got stale-window lagged labels from FINSABER data ending 2024-12-31; fixed with the
  10-day staleness guard, after which agreement is exact. The gate doing its job — no horizon
  number was ever read off the ungated build. Attempt-1 log preserved as
  logs/run_horizon_attempt1.log.)
- 2026-07-19 — job 3557942 (Teaching, 2 CPU, 2h18m): full Stage E chain — feature build,
  fusion ladder both windows (first 12-yr run with the `RISK_TEST_START` switch), conditioning
  both windows. All prior anchors reproduced exactly (struct+tfidf 0.561/0.614; unconditional
  text increment +0.016 p=0.098 / +0.011 p=0.057). Level rows null; conditioning signed as above.
  STAGE E COMPLETE.

- 2026-07-18 — job 3557338 (Teaching, 2 CPU): full fusion ladder with calls rows. Backtest
  struct+calls −0.004 (p=0.395) vs structured; struct+tfidf+calls −0.001 (p=0.436) vs
  struct+tfidf. Same day: `call_filing_gate.py` (login, ~2 min) — filing-level 2025 gate NULL on
  both populations. Earlier: full Finstream corpus (1.07GB parquet, HF) downloaded;
  `build_call_features.py` rebuilt (CIK join). Final pass: features rebuilt parquet-only for a
  single-source data story — diff vs the two-source build showed 0 changed values on evaluated
  years (only 14 unused 2026 matches dropped; 7,159 matched). STAGE C COMPLETE — all
  experimental results of the study are now final.
- 2026-07-04 (evening) — job 3527629 (h200_3g.71gb, 20 epochs batch 128): ftvol2018 trained
  (best 2017 filing IC 0.635 @ epoch 19), windowed encode, admissible 2018–2024 backtest.
  RESULT: 0.497–0.504 IC, ≈0 over structured, −0.06 vs struct+tfidf → FINAL VERDICT above.
- 2026-07-04 — E3 backtests re-run (both windows) with the complete clean change columns +
  structured[ridge] lane + paired-vs-struct+tfidf table baked in. Anchors reproduced exactly.
  Semantic change adds nothing (E2 closed). Remaining open: the ftvol2018 clean retrain
  (`contrastive/run_ftvol2018.sh`, GPU) — its 2018–2024 paired DM test vs struct+tfidf is the
  final verdict.
- 2026-07-03 (morning) — job 3527358 completed: ftvol + bge full-text embeddings cached (all 5
  encoders done). change_features rebuilt with semantic (sbert/volaware) + shape columns;
  chg_new_para_frac populated for the first time (median 0.27). Full val-2025 grid run:
  bge loses decisively; ftvol/volaware reach DM-insignificant parity with struct+tfidf.
  Provenance audit found both vol-encoders trained on pre-2025 vol labels (backtest inadmissible
  with current checkpoints) + ftvol epoch-selected on val-2025 → clean-protocol retrain queued.
- 2026-07-03 — encode job 3527085: **volaware saved** (462,590 windows). ftvol failed — checkpoint
  had no tokenizer/module files; fixed by copying them from volaware (same mpnet base), smoke-tested.
  bge encoded fully but `np.save` hit EIO on the 95%-full OST; corrupt file removed and
  `topics/out` + `phase5` striped to OST2 (`lfs setstripe -i 2`). Third submit encodes only ftvol+bge.
- 2026-07-02 — P0-a join fix + re-baseline; P0-b encode scripts committed (GPU pending).
  E2 built (6,713 pairs, COVID-dip validated). E3 run both windows + structured[ridge] fair lane.
  E1 grid run (text/structured/change rows). Key discovery: head asymmetry inflated the old text
  increment; fair-head text gain is +0.011 IC (p=0.057, 12 yr) with a large R² gain.
