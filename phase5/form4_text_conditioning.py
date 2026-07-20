"""
Stage E — does the TEXT increment depend on insider activity?

The absorption reading of the study (Stage C, Part V) says text adds value where the market has
not yet impounded the information into cheaper predictors. Insider-trading intensity/dispersion
proxies information asymmetry, so the pre-registered hypothesis is directional: the text
increment (struct+tfidf [sparse] minus structured [ridge], the study's fair pair) is LARGER in
the high-insider-activity half of each year's cross-section.

Design (leakage-free): the conditioning variables (f4_abn_intensity, f4_disagreement) come from
trailing pre-filing windows, so splitting each TEST year's cross-section at its median is known
at prediction time. Within each half we compute the IC of both models on the same firms, take
the paired per-year delta-IC, and test the high-minus-low difference across years (the study's
usual paired-across-years discipline). Either outcome is reportable: positive ties Stages C/E
into one absorption narrative; null bounds it.

Run (after build_form4_features.py; RISK_TEST_START=2013 for the 12-year window):
    python phase5/form4_text_conditioning.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_common as E

MIN_HALF = 20  # minimum firms per half-year cell for a within-cell IC

CONDITIONERS = ["f4_abn_intensity", "f4_disagreement"]


def half_ic(y, pred, mask):
    if mask.sum() < MIN_HALF:
        return np.nan
    return float(spearmanr(y[mask], pred[mask]).statistic)


def main():
    panel = E.load_panel().sort_values("year").reset_index(drop=True)
    y = panel.log_fwd.to_numpy()
    years = panel.year.to_numpy()

    Xstr, scols = E.structured_matrix(panel)
    if Xstr is None:
        raise SystemExit("market_features.parquet not found — run build_market_features.py first.")
    Xf4, f4_cols = E.form4_matrix(panel)
    if Xf4 is None:
        raise SystemExit("form4_features.parquet not found — run build_form4_features.py first.")

    lag = panel[["log_lag"]].to_numpy()
    struct_block = np.hstack([lag, Xstr])
    tdf = panel.copy()
    for i, c in enumerate(scols):
        tdf[c] = Xstr[:, i]

    # the study's fair text pair: struct+tfidf [sparse] vs structured [ridge]
    print(f"Panel: {len(panel):,} filings | test years {E.TEST_YEARS}")
    print("Backtesting the pair (struct+tfidf [sparse] vs structured [ridge])...", flush=True)
    pred_base = E.rolling_predict(struct_block, y, years, E.make_imputed_ridge)
    pred_text = E.rolling_predict(
        tdf, y, years, lambda: E.TextNumericRidge(numeric_cols=["log_lag"] + scols))

    # sanity anchor: the unconditional paired text increment must reproduce the known result
    py_b = E.per_year_metrics(y, pred_base, years, E.TEST_YEARS)
    py_t = E.per_year_metrics(y, pred_text, years, E.TEST_YEARS)
    d, p = E.paired_year_test(py_t, py_b, key="spear")
    print(f"Unconditional text increment: dIC {d:+.3f} (p={p:.3f})  "
          f"[anchor: +0.016 p=0.098 (7yr) / +0.011 p=0.057 (12yr)]\n")

    for cond in CONDITIONERS:
        cvar = Xf4[:, f4_cols.index(cond)]
        print(f"=== conditioning on {cond} (within-year median split) ===")
        print(f"{'year':>5s} {'n_lo/n_hi':>10s} {'dIC_low':>8s} {'dIC_high':>9s} {'high-low':>9s}")
        diffs, rows = [], []
        for Y in E.TEST_YEARS:
            te = (years == Y) & np.isfinite(pred_base) & np.isfinite(pred_text) & np.isfinite(cvar)
            if te.sum() < 2 * MIN_HALF:
                continue
            med = float(np.median(cvar[te]))
            # strict split: for mass-at-zero conditioners (f4_disagreement's median is 0 in every
            # year) this contrasts "any disagreement" vs "none"; for continuous ones it is an
            # ordinary median split
            hi, lo = te & (cvar > med), te & (cvar <= med)
            d_hi = half_ic(y, pred_text, hi) - half_ic(y, pred_base, hi)
            d_lo = half_ic(y, pred_text, lo) - half_ic(y, pred_base, lo)
            if np.isfinite(d_hi) and np.isfinite(d_lo):
                diffs.append(d_hi - d_lo)
                rows.append((Y, int(lo.sum()), int(hi.sum()), d_lo, d_hi))
        for Y, nl, nh, d_lo, d_hi in rows:
            print(f"{Y:5d} {f'{nl}/{nh}':>10s} {d_lo:8.3f} {d_hi:9.3f} {d_hi - d_lo:9.3f}")
        if len(diffs) >= 2:
            t = ttest_1samp(diffs, 0.0)
            print(f"mean(high-low dIC) = {np.mean(diffs):+.3f}  t = {t.statistic:.2f}  "
                  f"p = {t.pvalue:.3f}  ({len(diffs)} years)")
            print("Hypothesis (pre-registered): positive => text increment concentrates in the "
                  "high-asymmetry half.\n")
        else:
            print("Too few usable years for the paired test.\n")


if __name__ == "__main__":
    main()
