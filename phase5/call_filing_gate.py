"""
Stage C — FILING-LEVEL confirmation gate (pre-filing 2025 calls).

The call-anchored pilot (call_combined_gate.py) showed call tone adds +0.039 IC over the structured
block when predicting post-CALL 30d volatility. This script runs the confirmation that
dataset_collection_discussion/calls_collection_spec_2025.md pre-registered: after collecting, for
each 2025 10-K, the annual/Q4 earnings call that PRECEDES the filing, test whether call tone adds
signal at the FILING level — predicting the same post-filing 30-day forward volatility the whole
study targets.

Design note (why K-fold CV within 2025, not the train<2025 val-2025 ladder): the collected calls
cover only the 2025 filings, so under the ladder's train<2025 / eval-2025 split the call columns
are all-missing in training and no head can learn them. The admissible design is cross-validation
within the 2025 cross-section — identical fold assignment for every condition, so each comparison
is paired — which mirrors the leave-one-quarter-out design of the pilot. The metric is the study's
headline within-year cross-sectional Spearman IC, computed on pooled out-of-fold predictions.

Conditions (same head within every comparison — the head-fairness rule from the stress test):

  structured                (HGB)    vs  structured + tone                (HGB)
  structured              (ridge)    vs  structured + tone              (ridge)
  struct + 10-K tfidf     (ridge)    vs  struct + 10-K tfidf + tone     (ridge)

Each comparison is reported on (a) the FULL 2025 panel (tone NaN where the firm has no matched
call; HGB is NaN-native, the ridge heads median-impute) and (b) the MATCHED subset (filings with a
pre-filing call), so an increment cannot be an artifact of which firms happen to have call
coverage. Significance: bootstrap 95% CI on the pooled delta-IC and a paired DM test on squared
error.

GATE (pre-registered in the spec): if a +tone condition clearly beats its no-tone counterpart,
commit to the 2008-2024 Finstream backfill; otherwise Stage C stops here with the call-anchored
pilot evidence and this filing-level result.

Run (after the pre-filing collection has landed and build_call_features.py has been re-run):
    python dataset_config/build_call_features.py
    python phase5/call_filing_gate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_common as E

K_FOLDS = 5
EVAL_YEAR = 2025
# same tone block as the call-anchored pilot (call_days_before_filing is matching metadata, not a predictor)
TONE = ["call_n_words", "call_uncertainty_dens", "call_negative_dens",
        "call_positive_dens", "call_risk_dens", "call_qa_uncertainty"]


def kfold_pred(make_est, X, y, folds):
    """Pooled out-of-fold predictions; X is an ndarray or a DataFrame (TextNumericRidge)."""
    pred = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        est = make_est()
        Xtr = X.loc[tr] if isinstance(X, pd.DataFrame) else X[tr]
        Xte = X.loc[te] if isinstance(X, pd.DataFrame) else X[te]
        est.fit(Xtr, y[tr])
        pred[te] = est.predict(Xte)
    return pred


def boot_delta_ic(y, pred_a, pred_b, n=2000, seed=0):
    """Paired bootstrap 95% CI for IC(a) - IC(b) over resampled filings."""
    m = np.isfinite(pred_a) & np.isfinite(pred_b)
    yt, a, b = y[m], pred_a[m], pred_b[m]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        deltas.append(spearmanr(yt[idx], a[idx]).statistic
                      - spearmanr(yt[idx], b[idx]).statistic)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def ic_r2(y, pred):
    m = np.isfinite(pred)
    return float(spearmanr(y[m], pred[m]).statistic), E.r2_only(y[m], pred[m])


def run_population(df, y, scols, tone, label, seed=42):
    """All three paired comparisons on one population; identical folds across every condition."""
    n = len(df)
    print(f"\n-- population: {label} (n={n}) --")
    if n < 50:
        print("   too few rows — skipped.")
        return
    folds = np.random.default_rng(seed).permutation(np.arange(n) % K_FOLDS)
    num_base = ["log_lag"] + scols

    def dense(cols, head):
        X = df[cols].to_numpy(dtype=float)
        mk = E.make_hgb if head == "hgb" else E.make_imputed_ridge
        return kfold_pred(mk, X, y, folds)

    def text(numcols):
        return kfold_pred(lambda: E.TextNumericRidge(numeric_cols=numcols), df, y, folds)

    pairs = [
        ("structured             (HGB)", dense(num_base, "hgb"),
         "structured + tone      (HGB)", dense(num_base + tone, "hgb")),
        ("structured           (ridge)", dense(num_base, "ridge"),
         "structured + tone    (ridge)", dense(num_base + tone, "ridge")),
        ("struct + tfidf       (ridge)", text(num_base),
         "struct + tfidf + tone(ridge)", text(num_base + tone)),
    ]

    print(f"   {'condition':30s} {'IC':>7s} {'R2_log':>7s} {'dIC':>7s} {'95% CI dIC':>17s} {'DM p':>6s}")
    for base_name, pb, tone_name, pt in pairs:
        icb, r2b = ic_r2(y, pb)
        ict, r2t = ic_r2(y, pt)
        lo, hi = boot_delta_ic(y, pt, pb)
        _, dm_p = E.paired_dm_test(y, pt, pb)
        print(f"   {base_name:30s} {icb:7.3f} {r2b:7.3f}")
        print(f"   {tone_name:30s} {ict:7.3f} {r2t:7.3f} {ict - icb:+7.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] {dm_p:6.3f}")


def main():
    panel = E.load_panel()
    p = panel[panel.year == EVAL_YEAR].reset_index(drop=True)
    if len(p) < 50:
        raise SystemExit(f"Only {len(p)} filings in {EVAL_YEAR} — check the panel.")
    y = p.log_fwd.to_numpy()

    Xstr, scols = E.structured_matrix(p)
    if Xstr is None:
        raise SystemExit("Stage A structured features not built — run build_market_features.py first.")
    for i, c in enumerate(scols):
        p[c] = Xstr[:, i]

    Xcall, ccols = E.call_matrix(p)
    if Xcall is None:
        raise SystemExit("datasets/call_features.parquet missing — run build_call_features.py first.")
    tone = [c for c in TONE if c in ccols]
    for c in tone:
        p[c] = Xcall[:, ccols.index(c)]

    matched = p["call_uncertainty_dens"].notna().to_numpy()
    print(f"Panel {EVAL_YEAR}: {len(p):,} filings | with pre-filing call: {int(matched.sum())} "
          f"({matched.mean():.0%}) | tone features: {tone}")
    if matched.sum() < 100:
        print("[warn] fewer than 100 matched calls — the pre-filing collection has probably not "
              "landed yet (see dataset_collection_discussion/calls_collection_spec_2025.md).")

    run_population(p, y, scols, tone, "FULL 2025 panel (tone missing where no call)")
    pm = p[matched].reset_index(drop=True)
    run_population(pm, y[matched], scols, tone, "MATCHED subset (pre-filing call only)")

    print("\nGATE: commit to the 2008-2024 Finstream backfill only if a '+ tone' row beats its "
          "counterpart (positive dIC with a CI mostly above zero / DM p small) on BOTH populations. "
          "Otherwise Stage C concludes with the call-anchored pilot + this filing-level result.")


if __name__ == "__main__":
    main()
