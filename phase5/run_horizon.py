"""
#17 Horizon robustness — the study's fair text pair at the current label horizon, plus the
filing-anchored earnings-call pair (Prof Ma, 2026-08-04).

Answers Prof Ma's Q1/Q2: is the count-model result (and the text increment) consistent across
label horizons? One process per horizon, RISK_HORIZON env selecting the labels (default 30 = the
frozen study labels — that run must reproduce the known anchors before any other horizon is
read). Lanes, all with horizon-matched log_lag:

  lagged                 raw persistence (pred = log_lag)
  structured [ridge]     E.make_imputed_ridge on log_lag + market block
  struct+tfidf [sparse]  E.TextNumericRidge — the study's final model
  structured [hgb]       the Stage C call baseline (same head as its + calls counterpart)
  struct+calls [hgb]     structured + earnings-call tone
  struct+tfidf+calls     the full model + tone

The call lanes extend Stage C's filing-anchored null across the window spectrum: that null is what
the Stage C verdict rests on, so it has to be shown not to be a 30-day artifact. Construction is
copied from run_fusion.py (tone columns only, call_days_before_filing is matching metadata rather
than a predictor) so the H=30 run reproduces the published Stage C rows exactly.

Reported per horizon: 2018-2024 expanding-window backtest (per-year ICs, mean IC, t, paired dIC + p
against the same-head counterpart) and clean val-2025 (IC, R2_log).

Run:  RISK_HORIZON=60 python phase5/run_horizon.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_common as E


def val2025(panel, X, y, make_est):
    tr = (panel.year < 2025).to_numpy()
    ev = (panel.year == 2025).to_numpy()
    if ev.sum() < 20:
        return None
    import pandas as pd
    Xtr = X[tr] if not isinstance(X, pd.DataFrame) else X.loc[tr]
    Xev = X[ev] if not isinstance(X, pd.DataFrame) else X.loc[ev]
    est = make_est(); est.fit(Xtr, y[tr])
    yp = est.predict(Xev); yt = y[ev]
    return {"ic": float(spearmanr(yt, yp).statistic), "r2": E.r2_only(yt, yp), "n": int(ev.sum())}


def main():
    panel = E.load_panel().sort_values("year").reset_index(drop=True)
    y = panel.log_fwd.to_numpy()
    years = panel.year.to_numpy()
    log_lag = panel.log_lag.to_numpy()

    Xstr, scols = E.structured_matrix(panel)
    if Xstr is None:
        raise SystemExit("market_features.parquet not found — run build_market_features.py first.")
    struct_block = np.hstack([panel[["log_lag"]].to_numpy(), Xstr])
    tdf = panel.copy()
    for i, c in enumerate(scols):
        tdf[c] = Xstr[:, i]
    make_text = lambda: E.TextNumericRidge(numeric_cols=["log_lag"] + scols)

    # ---- Stage C call lanes: tone columns only, same construction as run_fusion.py ----
    Xcall, ccols_all = E.call_matrix(panel)
    tone_cols = [c for c in (ccols_all or []) if c != "call_days_before_filing"]
    Xtone = (Xcall[:, [ccols_all.index(c) for c in tone_cols]]
             if Xcall is not None and tone_cols else None)
    has_calls = Xtone is not None and bool(np.isfinite(Xtone).any())

    print(f"=== horizon {E.HORIZON}d | panel {len(panel):,} filings | "
          f"test years {E.TEST_YEARS} ===")
    if has_calls:
        call_block = np.hstack([struct_block, Xtone])
        tdf_call = tdf.copy()
        for i, c in enumerate(tone_cols):
            tdf_call[c] = Xtone[:, i]
        make_text_call = lambda: E.TextNumericRidge(numeric_cols=["log_lag"] + scols + tone_cols)
        print(f"calls: {int(np.isfinite(Xtone).any(axis=1).sum()):,}/{len(panel):,} filings have "
              f"a pre-filing call | tone cols: {tone_cols}")
    else:
        print("calls: call_features.parquet absent or empty -> call lanes skipped")

    # ---------- backtest 2018-2024 ----------
    print("Backtesting (structured [ridge], struct+tfidf [sparse])...", flush=True)
    pred_base = E.rolling_predict(struct_block, y, years, E.make_imputed_ridge)
    pred_text = E.rolling_predict(tdf, y, years, make_text)

    py_lag = E.per_year_metrics(y, log_lag, years, E.TEST_YEARS)
    py_b = E.per_year_metrics(y, pred_base, years, E.TEST_YEARS)
    py_t = E.per_year_metrics(y, pred_text, years, E.TEST_YEARS)
    dic, p = E.paired_year_test(py_t, py_b, key="spear")

    print(f"\n{'year':>5s} {'n':>5s} {'lagged':>7s} {'struct':>7s} {'s+tfidf':>8s} {'dIC':>7s}")
    for Y in sorted(set(py_lag) & set(py_b) & set(py_t)):
        print(f"{Y:5d} {py_b[Y]['n']:5d} {py_lag[Y]['spear']:7.3f} {py_b[Y]['spear']:7.3f} "
              f"{py_t[Y]['spear']:8.3f} {py_t[Y]['spear'] - py_b[Y]['spear']:7.3f}")
    al, ab, at = (E.aggregate_cs(x) for x in (py_lag, py_b, py_t))
    print(f"{'mean':>5s} {'':>5s} {al['mean_ic']:7.3f} {ab['mean_ic']:7.3f} {at['mean_ic']:8.3f}"
          f"   (IC_t: lagged {al['ic_t']:.1f} / struct {ab['ic_t']:.1f} / s+tfidf {at['ic_t']:.1f})")
    print(f"paired text increment (s+tfidf vs struct, same-universe years): "
          f"dIC {dic:+.3f}  p = {p:.3f}")

    # ---------- Stage C: does the filing-anchored call null hold at this window? ----------
    call_rows, d_sc, p_sc, d_stc, p_stc = [], float("nan"), float("nan"), float("nan"), float("nan")
    if has_calls:
        print("\nCall lanes (each paired against its SAME-HEAD counterpart)...", flush=True)
        pred_hgb = E.rolling_predict(struct_block, y, years, E.make_hgb)
        pred_hgb_call = E.rolling_predict(call_block, y, years, E.make_hgb)
        pred_text_call = E.rolling_predict(tdf_call, y, years, make_text_call)

        py_h = E.per_year_metrics(y, pred_hgb, years, E.TEST_YEARS)
        py_hc = E.per_year_metrics(y, pred_hgb_call, years, E.TEST_YEARS)
        py_tc = E.per_year_metrics(y, pred_text_call, years, E.TEST_YEARS)
        d_sc, p_sc = E.paired_year_test(py_hc, py_h, key="spear")
        d_stc, p_stc = E.paired_year_test(py_tc, py_t, key="spear")

        ah, ahc, atc = (E.aggregate_cs(x) for x in (py_h, py_hc, py_tc))
        call_rows = [("struct+calls [hgb] vs structured [hgb]",
                      ahc["mean_ic"], ah["mean_ic"], d_sc, p_sc),
                     ("struct+tfidf+calls vs struct+tfidf",
                      atc["mean_ic"], at["mean_ic"], d_stc, p_stc)]
        print(f"  {'pair (same head)':40s} {'IC':>7s} {'vs':>7s} {'dIC':>7s} {'p':>6s}")
        for name, ic_c, ic_b, d, pp in call_rows:
            print(f"  {name:40s} {ic_c:7.3f} {ic_b:7.3f} {d:+7.3f} {pp:6.3f}")

    # ---------- clean val-2025 ----------
    print("\nval-2025 (train < 2025):")
    ev = (panel.year == 2025).to_numpy()
    r_lag = ({"ic": float(spearmanr(y[ev], log_lag[ev]).statistic),
              "r2": E.r2_only(y[ev], log_lag[ev]), "n": int(ev.sum())} if ev.sum() >= 20 else None)
    rows = [("lagged", r_lag),
            ("structured [ridge]", val2025(panel, struct_block, y, E.make_imputed_ridge)),
            ("struct+tfidf [sparse]", val2025(panel, tdf, y, make_text))]
    if has_calls:
        rows += [("structured [hgb]", val2025(panel, struct_block, y, E.make_hgb)),
                 ("struct+calls [hgb]", val2025(panel, call_block, y, E.make_hgb)),
                 ("struct+tfidf+calls", val2025(panel, tdf_call, y, make_text_call))]
    for name, r in rows:
        if r:
            print(f"  {name:22s} IC {r['ic']:6.3f}  R2_log {r['r2']:7.3f}  n={r['n']}")

    v_b, v_t = rows[1][1], rows[2][1]
    print(f"\nSUMMARY H={E.HORIZON:<3d} backtest: lagged {al['mean_ic']:.3f} | "
          f"struct {ab['mean_ic']:.3f} | s+tfidf {at['mean_ic']:.3f} | dIC {dic:+.3f} p={p:.3f}"
          + (f" || val2025: struct {v_b['ic']:.3f} | s+tfidf {v_t['ic']:.3f}"
             if v_b and v_t else ""))
    if has_calls:
        print(f"SUMMARY_CALLS H={E.HORIZON:<3d} struct+calls dIC {d_sc:+.3f} p={p_sc:.3f} | "
              f"struct+tfidf+calls dIC {d_stc:+.3f} p={p_stc:.3f}")


if __name__ == "__main__":
    main()
