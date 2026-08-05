"""
Round 8 — does the LLM-elicited score beat the hand-crafted Form 4 features?

Three conditions, ONE head, ONE set of folds (the head-fairness rule from the stress test):

  1. structured                  [ridge]     the no-insider baseline
  2. structured + f4_manual      [ridge]     the eight hand-crafted features (Stage E)
  3. structured + f4_llm         [ridge]     the generative model's scores

Condition 3 beating condition 2 would mean hand-crafted feature engineering was the bottleneck.
Condition 3 failing to beat condition 1, as condition 2 already does (Stage E level test: null on
all four arms), would mean the bottleneck is the information content of the Form 4 record itself,
which is the prior stated in study_extended.md VII.5.

Design: 5-fold CV within each filing year, identical fold assignment across every condition so
each comparison is paired. This is the same admissible design call_filing_gate.py uses, for the
same reason — the LLM score exists only on the 2024-2025 rows, so under a train<year split no head
could ever learn it. Metric is the study's headline within-year cross-sectional Spearman IC on
pooled out-of-fold predictions.

STOPPING RULE, fixed before the scores were ever looked at: if condition 3 does not beat condition
2 by more than the half-width of the paired bootstrap 95% CI on the IC difference, the pilot closes
as a measured null and is written up as the bounded answer to QnA Q5. No expansion to the full
panel, no earnings-call lane, no second model family, without a fresh decision.

Also reported, and arguably the more durable output:
  - within-filing score spread across the K samples (the instability measurement)
  - IC of a single draw vs the K-averaged score (how much averaging recovers)
  - if the identified prompts were also scored, the anonymised-minus-identified IC gap, which is
    this study's own measurement of the combined look-ahead + distraction premium and the direct
    analogue of the ~0.06 IC encoder lookahead priced in Part III.3

Run:  python phase5/llm_vs_manual.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_common as E

DATA = ROOT / "datasets"
SCORES = DATA / "form4_llm_scores.parquet"
SCORES_IDENT = DATA / "form4_llm_scores_ident.parquet"

K_FOLDS = 5
SEED = 42
FIELDS = ["volatility_risk", "information_asymmetry", "confidence"]
LLM_COLS = [f"llm_{f}" for f in FIELDS]


def kfold_pred(make_est, X, y, folds):
    pred = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        est = make_est()
        est.fit(X[tr], y[tr])
        pred[te] = est.predict(X[te])
    return pred


def boot_delta_ic(y, pred_a, pred_b, n=2000, seed=0):
    """Paired bootstrap 95% CI for IC(a) - IC(b) over resampled filings (as call_filing_gate.py)."""
    m = np.isfinite(pred_a) & np.isfinite(pred_b)
    yt, a, b = y[m], pred_a[m], pred_b[m]
    rng = np.random.default_rng(seed)
    d = [spearmanr(yt[i], a[i]).statistic - spearmanr(yt[i], b[i]).statistic
         for i in (rng.integers(0, len(yt), len(yt)) for _ in range(n))]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ic(y, pred):
    m = np.isfinite(pred)
    return float(spearmanr(y[m], pred[m]).statistic)


def stability_report(sc):
    print("\n=== 1. Score stability (the measurement VII.5 currently cites rather than shows) ===")
    tot = sc.llm_n_ok.sum() + sc.llm_n_fail.sum()
    print(f"Parse failures: {sc.llm_n_fail.sum():,}/{tot:,} samples "
          f"({sc.llm_n_fail.sum()/max(tot,1):.2%})")
    print(f"{'field':24s} {'across-filing sd':>17s} {'median within-filing sd':>24s} {'ratio':>7s}")
    for f in FIELDS:
        across = sc[f"llm_{f}"].std()
        within = sc[f"llm_{f}_sd"].median()
        print(f"{f:24s} {across:17.2f} {within:24.2f} {within/max(across,1e-9):7.2f}")
    print("ratio >= 1 means a filing's score moves as much between re-runs of the same prompt as "
          "it does between different filings, i.e. the feature is mostly noise.")


def main():
    if not SCORES.exists():
        raise SystemExit(f"{SCORES} not found — run llm_score_form4.py first.")

    panel = E.load_panel()
    p = panel[panel.year.isin([2024, 2025])].reset_index(drop=True)
    Xstr, scols = E.structured_matrix(p)
    if Xstr is None:
        raise SystemExit("market_features.parquet not found.")
    Xf4, f4_cols = E.form4_matrix(p)
    if Xf4 is None:
        raise SystemExit("form4_features.parquet not found.")

    sc = pd.read_parquet(SCORES)
    sc["filing_date"] = pd.to_datetime(sc["filing_date"])
    key = p[["ticker", "filing_date"]].copy()
    key["filing_date"] = pd.to_datetime(key["filing_date"])
    merged = key.merge(sc, on=["ticker", "filing_date"], how="left")
    Xllm = merged[LLM_COLS].to_numpy(dtype=float)

    has = np.isfinite(Xllm).all(axis=1)
    print(f"Panel 2024-2025: {len(p):,} filings | with an LLM score: {int(has.sum()):,} "
          f"({has.mean():.1%})")

    # every condition is evaluated on the SAME rows, so coverage cannot drive a difference.
    # `y` is taken before `p` is rebound, since `has` indexes the unfiltered panel.
    y = p.log_fwd.to_numpy()[has]
    key_all = key.copy()
    p = p[has].reset_index(drop=True)
    Xstr, Xf4, Xllm = Xstr[has], Xf4[has], Xllm[has]
    merged = merged[has].reset_index(drop=True)
    stability_report(merged)

    lag = p[["log_lag"]].to_numpy()
    base = np.hstack([lag, Xstr])
    conds = [("structured            ", base),
             ("structured + f4_manual", np.hstack([base, Xf4])),
             ("structured + f4_llm   ", np.hstack([base, Xllm]))]

    print("\n=== 2. Does the LLM score beat the hand-crafted features? ===")
    print(f"5-fold CV within each year, identical folds, head = imputed ridge, n={len(p):,}")
    rng = np.random.default_rng(SEED)
    folds = np.empty(len(p), dtype=int)
    for yr in sorted(p.year.unique()):                 # fold within year, so each fold spans both
        m = (p.year == yr).to_numpy()                  # years in the same proportion
        folds[m] = rng.permutation(np.arange(m.sum()) % K_FOLDS)

    preds = {}
    for name, X in conds:
        preds[name] = kfold_pred(E.make_imputed_ridge, X, y, folds)
        print(f"  {name}  IC {ic(y, preds[name]):6.3f}   R2_log {E.r2_only(y, preds[name]):7.3f}")

    man, llm = conds[1][0], conds[2][0]
    for a, b in ((man, conds[0][0]), (llm, conds[0][0]), (llm, man)):
        d = ic(y, preds[a]) - ic(y, preds[b])
        lo, hi = boot_delta_ic(y, preds[a], preds[b])
        _, dm = E.paired_dm_test(y, preds[a], preds[b])
        print(f"  dIC {a.strip()} vs {b.strip():22s} {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"DM p {dm:.3f}")

    d_llm_man = ic(y, preds[llm]) - ic(y, preds[man])
    lo, hi = boot_delta_ic(y, preds[llm], preds[man])
    half = (hi - lo) / 2
    verdict = "BEATS" if d_llm_man > half else "MEASURED NULL"
    print(f"\nSTOPPING RULE: dIC(llm vs manual) = {d_llm_man:+.3f}, CI half-width = {half:.3f} "
          f"-> {verdict}")
    if verdict != "BEATS":
        print("  -> pilot closes here. Write up as the bounded answer to QnA Q5; do not expand.")

    print("\n=== 3. How much does averaging K samples buy? ===")
    X1 = np.hstack([base, merged[[f"llm_{f}_s0" for f in FIELDS]].to_numpy(dtype=float)])
    p1 = kfold_pred(E.make_imputed_ridge, X1, y, folds)
    print(f"  single draw       IC {ic(y, p1):6.3f}")
    print(f"  mean of K samples IC {ic(y, preds[llm]):6.3f}   (gain {ic(y, preds[llm])-ic(y, p1):+.3f})")

    if SCORES_IDENT.exists():
        print("\n=== 4. Anonymised vs identified prompts (contamination premium) ===")
        si = pd.read_parquet(SCORES_IDENT)
        si["filing_date"] = pd.to_datetime(si["filing_date"])
        mi = key_all[has].reset_index(drop=True).merge(si, on=["ticker", "filing_date"], how="left")
        Xi = mi[LLM_COLS].to_numpy(dtype=float)
        ok = np.isfinite(Xi).all(axis=1)
        if ok.sum() > 50:
            pi = kfold_pred(E.make_imputed_ridge, np.hstack([base, Xi])[ok], y[ok], folds[ok])
            pa = kfold_pred(E.make_imputed_ridge, np.hstack([base, Xllm])[ok], y[ok], folds[ok])
            gap = ic(y[ok], pi) - ic(y[ok], pa)
            lo2, hi2 = boot_delta_ic(y[ok], pi, pa)
            print(f"  identified  IC {ic(y[ok], pi):6.3f} | anonymised IC {ic(y[ok], pa):6.3f}")
            print(f"  premium from seeing the company's identity: {gap:+.3f} "
                  f"95% CI [{lo2:+.3f}, {hi2:+.3f}]  (n={int(ok.sum())})")
            print("  Compare with the ~0.06 IC encoder lookahead priced in Part III.3. Note the "
                  "sign is not predictable: Glasserman & Lin (2023) find distraction outweighing "
                  "look-ahead in-sample for large firms, so a NEGATIVE premium is a valid result.")
    else:
        print(f"\n(no {SCORES_IDENT.name} — run the identified lane for the contamination premium)")


if __name__ == "__main__":
    main()
