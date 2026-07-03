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

vs the weaker structured [hgb]: struct+tfidf +0.050 (p=0.114) 7-yr / +0.030 (p=0.113) 12-yr —
the old-style comparison, kept for reference. structured [hgb] vs lagged: +0.059 (p=0.007) 12-yr —
the strong-baseline claim is solid.

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

## Run log

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
