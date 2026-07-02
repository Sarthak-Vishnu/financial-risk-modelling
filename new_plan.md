# New Pipeline Plan — Redesign for Performance (and Marks)

## Context — why we're redesigning

The current pipeline answers a weak question: *"which text method ranks firms by volatility best?"* — and
the answer is that tuned TF-IDF beats every neural encoder (5 independent confirmations). Two root causes:

1. **No real baseline.** The only non-text feature in the model is `lagged_vol_30d` (one number). Volatility
   is driven by a well-known set of *structured* predictors (multi-horizon realized vol, size, leverage,
   liquidity, beta) that we never built. So absolute performance is capped AND there's no strong baseline to
   make a contribution against.
2. **The encoder learned the wrong objective.** DAPT + contrastive trained for *text similarity*, never for
   *volatility*. The embeddings are semantically smooth but not vol-discriminative.

**The reframe (the whole redesign in one line):**
> Stop comparing text-method vs text-method. Build a **strong structured-financial baseline**, then show
> whether **text (10-K + earnings calls) adds incremental, explainable signal on top of it** — with the
> encoder redesigned to be *volatility-aware*.

This is a "definitely succeeds" structure: the structured baseline alone produces strong numbers (realized-vol
persistence across horizons is a powerful predictor), and the research question ("does text add?") is
publishable whether the answer is a modest yes or a clean no — with the BERTopic risk axis providing
explanation TF-IDF cannot.

**Decisions locked with the user:** Balanced hybrid goal · add market + fundamentals + earnings calls ·
keep the encoder work (supervisor brief *mandates* "domain-adaptive pretraining, contrastive learning, and
financial-aware augmentations") but make it volatility-aware.

Target and metric are UNCHANGED: predict `log(fwd_vol_30d)`, evaluate by within-year cross-sectional Spearman
IC (+ R²_log), clean val-2025. Expanding-window backtest 2018–2024 for the leakage-free feature families.

---

## The redesigned architecture (5 stages)

```
[A] Structured feature block   ──┐
[B] Text: 10-K (DAPT+contrastive, NOW vol-aware) + topics ──┤
[C] Text: earnings-call tone ──┤──>  [D] Multimodal fusion model  ──>  [E] Eval: incremental IC over baseline
                                 ┘                                        + topic explanation
```

### [A] Structured-financial feature block — *the biggest lift, no new data*
Engineer from price files already on disk (`all_sp500_prices_2000_2024_delisted_include.csv`,
`crsp_2025_daily.csv`) keyed by `permno`, plus fundamentals via `gvkey`:
- **Realized vol at multiple horizons** before filing: 21d / 63d / 126d / 252d (annualized).
- **Vol-of-vol, return skew/kurtosis**, max drawdown over trailing year.
- **Liquidity/volume**: average dollar volume, Amihud illiquidity, turnover.
- **Market features**: trailing returns (1m/3m/12m), beta, idiosyncratic vol.
- **Fundamentals** (Compustat via gvkey, if collected): log market cap (size), leverage (debt/equity),
  book-to-market, ROA/profitability.
This block IS the new strong baseline ("can standard quant features rank firm vol?").

### [B] 10-K text — keep the mandated encoder work, but make it volatility-aware
Satisfies the supervisor brief while fixing the objective that made it lose:
- **DAPT**: keep as-is (in-domain MLM is fine; it's the base, not the problem).
- **Contrastive = "financial-aware augmentations" reinterpreted as volatility-aware pairs.** Replace/augment
  the similarity views with **supervised-contrastive on volatility buckets**: positives = filings in the same
  within-year forward-vol decile; hard negatives = filings far apart in vol. Optionally keep chrono/sector as
  auxiliary views with lower weight. This is the principled fix — embeddings become vol-discriminative *by
  construction*, not by accident. (Start the encoder from FinBERT or keep DAPT-mpnet; A/B both.)
- **Topics**: keep BERTopic for the **explanation contribution** (which risk themes drive a firm's vol).

### [C] Earnings-call text — the text source where neural NLP can actually beat bag-of-words
Management *tone* in Q&A is genuinely semantic (unlike legalese risk factors), so dense encoders have real
upside here. Features: call-level tone/sentiment embedding, uncertainty/hedging density, Q&A vs prepared-
remarks contrast. **Phased**: 2025–2026 calls already collected (299 files) → prove the signal on val-2025
first; backfill 2006–2024 only if the pilot shows lift (avoids a big collection effort that may not pay off).

### [D] Multimodal fusion model
A single model over `[A structured | B 10-K enc + topics | C call features]`:
- **Gradient-boosted trees** (HistGradientBoosting / LightGBM) — the standard winner for mixed tabular+text,
  handles nonlinearity and missing call data gracefully.
- Also keep the **Ridge hybrid** for a linear, interpretable comparison (reuse `phase5/eval_common.py`).
- Ablations are the core experiment: structured-only → +10-K text → +topics → +calls, reporting incremental
  IC at each step with paired significance tests (reuse `paired_year_test` / `bootstrap_ci`).

### [E] Evaluation & success criteria
- **Headline:** within-year cross-sectional IC, val-2025 clean + 2018–2024 backtest (leakage-free families).
- **Success = the structured+text model's IC beats the structured-only baseline *significantly*** (paired
  test across years), i.e. text demonstrably adds incremental signal. Secondary win: the vol-aware encoder
  closes/reverses the gap to TF-IDF on the text-only comparison.
- Honest negative outcomes recorded, not hidden — but with a strong baseline + explanation, the dissertation
  is markable either way.

---

## Execution order (cheapest, highest-leverage first)

1. **Stage A (structured features) — do first.** No collection; biggest performance jump; creates the
   baseline. New `dataset_config/build_market_features.py` → merge into `feature_table.parquet`. Re-run
   `phase5` to get the structured-only IC. *This step alone likely moves you well past the current 0.545.*
2. **Stage B encoder redesign** — new `contrastive/build_pairs.py` vol-bucket view + supervised-contrastive
   training; re-embed; re-fit topics. One GPU run.
3. **Stage D fusion + ablations** — extend `phase5` to the gradient-boosted multimodal head; run the
   ablation ladder; significance tests.
4. **Stage C earnings-call pilot (val-2025 only)** — prove tone signal before committing to the 2006–2024
   backfill. Backfill + full re-run only if the pilot lifts IC.
5. **Fundamentals** (Compustat via gvkey) folded into A when/if collected.

---

## Critical files

- **New**: `dataset_config/build_market_features.py` (Stage A — realized-vol/liquidity/return features from
  existing price CSVs, keyed by `permno`). Mirror the windowing logic in
  `dataset_config/compute_volatility_labels.py`.
- **Modify**: `dataset_config/build_feature_table.py` — join the new structured columns.
- **Modify**: `contrastive/build_pairs.py` — add a `vol_bucket` positive-pair view (within-year fwd-vol
  decile); `contrastive/train_contrastive.py` — add a supervised-contrastive (SupCon) loss path.
- **Modify/extend**: `phase5/eval_common.py` + `phase5/run_phase5.py` — add a `structured_matrix(df)` block
  and a gradient-boosted fusion head; the ablation ladder (structured → +10-K → +topics → +calls).
- **New (Stage C)**: `dataset_config/build_call_features.py` (call tone/uncertainty features from
  `datasets/calls/`), pilot on 2025 first.
- **Reuse**: topic machinery (`topics/fit_topics.py`), metrics (`eval_metrics`, `per_year_metrics`,
  `paired_year_test`, `bootstrap_ci` in `phase5/eval_common.py`), price/vol windowing in
  `compute_volatility_labels.py`.

## Verification

After Stage A: re-run `phase5/run_phase5.py` / `baselines/run_baselines.py` and confirm structured-only IC
> 0.545 floor (expected: clearly yes). After Stage D: the ablation ladder shows incremental IC at each text
layer with `paired_year_test` p-values; bootstrap CIs on the headline. After Stage C pilot: val-2025 IC with
calls vs without. Every stage compared against (a) AR(1) IC 0.466 and (b) the structured-only baseline.

## Honest risks
- Structured features may explain *so much* vol that text's incremental IC is small — but "text adds a
  significant few points over a strong quant baseline + explains which risks" is still a solid result.
- Earnings-call historical coverage (API Ninjas) for 2006–2024 may be patchy → that's exactly why Stage C is
  piloted on 2025 first before committing collection time.
- Vol-aware contrastive could overfit the vol label → guard with the clean val-2025 monitor and the
  expanding-window backtest.

---

## To deliver
On approval: write this as `new_plan.md` in the repo root, then begin at Stage A (build market features).
