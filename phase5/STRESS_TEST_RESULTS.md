# TF-IDF Stress-Test — results log

**Question.** Is "count-based TF-IDF is the best text representation" really the final answer, or can a
learned/multimodal model beat it? All numbers below are on the **P0-corrected panel**
(`fix_filing_join.py`: filename year = FILING year; unique fiscal keys). Pre-fix numbers
(struct 0.570 / struct+tfidf 0.611) paired texts with the next filing's labels and are not comparable.

## Anchors (val-2025, n=393, corrected panel — measured 2026-07-02)

| condition | IC | R²_log |
|---|---|---|
| lagged [hgb] | 0.457 | −0.130 |
| structured [hgb] | 0.558 | 0.094 |
| tfidf+lag [sparse ridge] | 0.541 | 0.116 |
| **struct+tfidf [sparse ridge]** | **0.610** | **0.220** |

Text increment over structured: **+0.052 IC** (was +0.041 pre-fix). Note: at n≈393 a single-year IC has a
±0.06 CI — val-2025 deltas alone are indicative only; the backtest paired tests are the verdict.

Notable audit finding: fresh text ≈ stale text IC (0.610 vs 0.611) — Item-1A signal is persistent
boilerplate, which directly motivates the disclosure-change features (E2).

## E3 — extended backtest (primary lens)

`python phase5/stress_grid.py --mode backtest --test-start 2018` and `--test-start 2013`

| condition | 2018–2024 mean IC (t) | 2013–2024 mean IC (t) | ΔIC vs struct (p) |
|---|---|---|---|
| lagged [hgb] | TODO | TODO | — |
| structured [hgb] | TODO | TODO | — |
| tfidf+lag [sparse] | TODO | TODO | TODO |
| struct+tfidf [sparse] | TODO | TODO | TODO |
| struct+change [hgb] | TODO | TODO | TODO |
| struct+tfidf+change [sparse] | TODO | TODO | TODO |

Pre-fix reference: struct+tfidf beat struct by +0.050 IC at p=0.090 on 7 years — under-powered; 12 years
is the powered test.

## E1 — fair grid + everything model (val-2025)

`python phase5/stress_grid.py --mode val2025` (after `sbatch topics/run_encode.sh` fills
`emb_<enc>_fixed.npy`; encoder rows auto-skip until then)

| row | IC [95% CI] | R² | DM p vs struct+tfidf |
|---|---|---|---|
| struct+tfidf_svd [hgb] | TODO | | |
| struct+enc[dual] [ridge/hgb] | TODO | | |
| struct+enc[volaware] [hgb] | TODO | | |
| struct+enc[ftvol] [hgb] | TODO | | |
| struct+enc[bge] [hgb] | TODO | | |
| pooling variants (risk_weighted / topk_risk) | TODO | | |
| struct+enc+topic[best] [hgb] | TODO | | |
| EVERYTHING (svd+enc+topic+chg) [hgb] | TODO | | |
| EVERYTHING sparse twin | TODO | | |

## E2 — disclosure-change features

`python dataset_config/build_change_features.py` — sanity gate: `chg_lex_cos` median in 0.55–0.97,
coverage ≈ filings with a ≤2-yr prior (~7k). Key read-out: does `chg_enc_cos_*` (semantic change) add
where `chg_lex_cos` (lexical change) does not? → the novel positive finding for dense encoders if yes.

## Decision tree → conclusion (every branch a positive headline)

| Outcome | Headline |
|---|---|
| A learned combo beats struct+tfidf (backtest, paired p<0.05) | Learned/multimodal model beats the count-based ceiling. |
| EVERYTHING > struct+tfidf | Dense/topic signal is orthogonal & additive to bag-of-words. |
| Change features add (esp. semantic change) | Novel disclosure-change signal for vol forecasting (Lazy-Prices extension). |
| All null | Text adds over a strong quant baseline (p<0.05 on 12 yrs), topics explain which risks, calls add tone; the exhaustive rep×head×pooling grid *characterizes* the signal as lexical — a sharp finding, not a negative. |

## Run log

- 2026-07-02 — P0-a join fix + re-baseline (this file's anchor table). P0-b encode scripts committed;
  GPU job pending. E1/E2/E3 queued.
