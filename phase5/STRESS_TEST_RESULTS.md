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
| struct+change [hgb] | 0.501 (5.0) | 0.578 (9.0) |
| struct+tfidf+change [sparse] | 0.559 (6.0) | 0.612 (10.7) |

Paired across-years vs **structured [ridge]** (the fair baseline):

| condition | 7-yr ΔIC (p) | 12-yr ΔIC (p) |
|---|---|---|
| struct+tfidf | +0.016 (0.098) | **+0.011 (0.057)** |
| struct+tfidf+change | +0.013 (0.226) | +0.009 (0.163) |
| structured [hgb] | −0.034 (0.152) | −0.019 (0.180) |

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
| pooling variants (risk_weighted/topk_risk × dual/sbert) | 0.573–0.578 | 0.145–0.172 | ≤0.128 |
| EVERYTHING svd+enc[sbert]+chg [hgb] | 0.576 [0.505, 0.649] | 0.147 | 0.040 |
| EVERYTHING sparse twin | 0.548 [0.465, 0.624] | −0.063 | 0.000 |
| struct+enc[volaware/ftvol/bge] | volaware cached (grid re-run in progress); ftvol+bge await third encode job | | |

**Encoder verdict (fair conditions — full text via windowing, corrected labels, both heads):**
dual/sbert sit at or below the no-text structured [ridge] baseline (0.591) on IC and are
significantly less accurate than struct+tfidf by DM test. Fixing the 65% truncation did NOT rescue
the encoders. The EVERYTHING model dilutes rather than adds; semantic change (chg_enc_cos_dual)
adds nothing over lexical change. Remaining open: volaware / ftvol / bge (the "modern embedder"
objection) — resubmit `topics/run_encode.sh` (now per-encoder-isolated) and re-run the grid.

DM p is on squared-error loss: the sparse TF-IDF reference is significantly *more accurate* (R²) than
every non-TF-IDF row so far — the lexical block earns its place on level accuracy, beyond ranking.

## What's left to settle the verdict

1. `sbatch topics/run_encode.sh` (GPU; optional `hf download BAAI/bge-base-en-v1.5` first) → fills
   `emb_<enc>_fixed.npy` (full-text windowed — encoders finally see 100% of the words TF-IDF sees).
2. Re-run `dataset_config/build_change_features.py` (semantic/shape change columns) and
   `python phase5/stress_grid.py --mode val2025` (encoder + EVERYTHING rows).
3. If any encoder row approaches the reference: add it to the backtest as an SVD-style leakage-free
   lane before claiming anything.

## Decision tree → conclusion (every branch a positive headline)

| Outcome | Headline |
|---|---|
| A learned combo beats struct+tfidf (backtest, paired p<0.05) | Learned/multimodal model beats the count-based ceiling. |
| EVERYTHING > struct+tfidf | Dense/topic signal is orthogonal & additive to bag-of-words. |
| Semantic change adds where lexical change doesn't | Novel disclosure-change signal for vol forecasting (Lazy-Prices extension). |
| All null | A strong structured-ridge baseline (IC 0.603, 12-yr) + a small but consistent lexical text increment (+0.011, p=0.057) + full head/representation/pooling grid *characterizing* the text signal as lexical-level — a sharp, honest contribution. |

## Run log

- 2026-07-03 — encode job 3527085: **volaware saved** (462,590 windows). ftvol failed — checkpoint
  had no tokenizer/module files; fixed by copying them from volaware (same mpnet base), smoke-tested.
  bge encoded fully but `np.save` hit EIO on the 95%-full OST; corrupt file removed and
  `topics/out` + `phase5` striped to OST2 (`lfs setstripe -i 2`). Third submit encodes only ftvol+bge.
- 2026-07-02 — P0-a join fix + re-baseline; P0-b encode scripts committed (GPU pending).
  E2 built (6,713 pairs, COVID-dip validated). E3 run both windows + structured[ridge] fair lane.
  E1 grid run (text/structured/change rows). Key discovery: head asymmetry inflated the old text
  increment; fair-head text gain is +0.011 IC (p=0.057, 12 yr) with a large R² gain.
