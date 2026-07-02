"""
Stage D — multimodal fusion + the ablation ladder (the core experiment of the redesign).

The research question is now: does TEXT add incremental, significant signal ON TOP OF a strong
structured-financial baseline? We build the ladder and report within-year cross-sectional IC:

  lagged              [log lagged vol]                       reference (~0.466)
  tfidf_lag           [TF-IDF + log lag]   (ridge)           the old floor (~0.545)
  structured          [log lag + market features] (HGB)      NEW strong baseline
  struct + tfidf      [TF-IDF + structured] (imputed ridge)  does lexical text add over structured?
  struct + enc        [structured + encoder emb] (HGB)       does the dense encoder add?
  struct + enc + topic [+ BERTopic loadings] (HGB)           does the topic axis add?

Two evaluations, because of representation leakage:
  (1) BACKTEST 2018-2024 cross-sectional IC  — leakage-free families only (lagged / structured /
      tfidf), with a paired across-years test vs the structured baseline.
  (2) CLEAN val-2025 (train<2025, eval 2025) — ALL conditions incl. encoder/topics, which were
      trained on all years so are only leakage-free on the held-out 2025 split.

Run (after build_market_features.py):
    python phase5/run_fusion.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_common as E

KEYS = ["ticker", "fiscal_year"]


def dense_blocks(panel):
    """Assemble aligned dense feature blocks (ndarray per row of `panel`)."""
    lag = panel[["log_lag"]].to_numpy()
    Xstr, scols = E.structured_matrix(panel)
    blocks = {"lag": lag}
    if Xstr is not None:
        blocks["structured"] = np.hstack([lag, Xstr])
        print(f"structured features: {len(scols)} cols -> {scols}")
    # encoder (mean-pooled) + topics for each encoder whose embeddings exist
    encs = {}
    for enc in ["volaware", "dual", "three_lora", "sbert", "ftvol"]:
        penc = E.mean_pooled_filings(enc, pool="mean")
        if penc is None:
            continue
        ecols = [c for c in penc.columns if c.startswith("e") and c[1:].isdigit()]
        m = panel.merge(penc, on=KEYS, how="left")
        Eenc = m[ecols].to_numpy()
        topic = E.topic_filings(enc)
        Tt = None
        if topic is not None:
            tcols = [c for c in topic.columns if c.startswith("t") and c[1:].isdigit()]
            mt = panel.merge(topic, on=KEYS, how="left")
            Tt = mt[tcols].to_numpy()
        encs[enc] = (Eenc, Tt)
    return blocks, encs, (Xstr is not None)


def cs_backtest(panel, X, y, years, lag, make_est):
    pred = E.rolling_predict(X, y, years, make_est)
    py = E.per_year_metrics(y, pred, years, E.TEST_YEARS, lag=lag)
    return pred, py, E.aggregate_cs(py)


def val2025(panel, X, y, make_est):
    from scipy.stats import spearmanr
    tr = (panel.year < 2025).to_numpy()
    ev = (panel.year == 2025).to_numpy()
    if ev.sum() < 20:
        return None
    Xtr = X[tr] if not isinstance(X, pd.DataFrame) else X.loc[tr]
    Xev = X[ev] if not isinstance(X, pd.DataFrame) else X.loc[ev]
    est = make_est(); est.fit(Xtr, y[tr])
    yp = est.predict(Xev); yt = y[ev]
    return {"ic": float(spearmanr(yt, yp).statistic), "r2": E.r2_only(yt, yp), "n": int(ev.sum())}


def main():
    panel = E.load_panel().sort_values("year").reset_index(drop=True)
    y = panel.log_fwd.to_numpy()
    years = panel.year.to_numpy()
    lag = panel.log_lag.to_numpy()
    print(f"Panel: {len(panel):,} filings | test years {E.TEST_YEARS}\n")

    blocks, encs, has_struct = dense_blocks(panel)
    if not has_struct:
        print("market_features.parquet not found — run build_market_features.py first "
              "(showing lagged/text-only conditions).")

    # ---------- (1) leakage-free backtest: cross-sectional IC ----------
    print("\n=== (1) BACKTEST 2018-2024 — cross-sectional IC (leakage-free families) ===")
    print(f"{'condition':22s} {'mean_IC':>8s} {'IC_t':>6s} {'CS_R2':>7s} {'dIC vs struct':>13s} {'p':>7s}")
    bt = {}
    # dense HGB conditions
    dense_conds = [("lagged", blocks["lag"], E.make_hgb)]
    if has_struct:
        dense_conds.append(("structured", blocks["structured"], E.make_hgb))
    for name, X, mk in dense_conds:
        _, py, agg = cs_backtest(panel, X, y, years, lag, mk)
        bt[name] = py
        print(f"{name:22s} {agg['mean_ic']:8.3f} {agg['ic_t']:6.2f} {agg['cs_r2']:7.3f}")

    # TF-IDF lanes (leakage-free): tfidf+lag, and tfidf+structured
    tdf = panel.copy()
    tfidf_lag = lambda: E.TextNumericRidge(numeric_cols=["log_lag"])
    _, py, agg = cs_backtest(panel, tdf, y, years, lag, tfidf_lag)
    bt["tfidf_lag"] = py
    base = bt.get("structured")
    dimp = E.paired_year_test(py, base, key="spear") if base else (float("nan"), float("nan"))
    print(f"{'tfidf_lag':22s} {agg['mean_ic']:8.3f} {agg['ic_t']:6.2f} {agg['cs_r2']:7.3f} "
          f"{dimp[0]:13.3f} {dimp[1]:7.3f}")

    if has_struct:
        # attach structured cols to the text DataFrame for TextNumericRidge (it median-imputes)
        Xstr, scols = E.structured_matrix(panel)
        tdf2 = panel.copy()
        for i, c in enumerate(scols):
            tdf2[c] = Xstr[:, i]
        tfidf_str = lambda: E.TextNumericRidge(numeric_cols=["log_lag"] + scols)
        _, py, agg = cs_backtest(panel, tdf2, y, years, lag, tfidf_str)
        dimp = E.paired_year_test(py, base, key="spear")
        print(f"{'struct + tfidf':22s} {agg['mean_ic']:8.3f} {agg['ic_t']:6.2f} {agg['cs_r2']:7.3f} "
              f"{dimp[0]:13.3f} {dimp[1]:7.3f}")

    # ---------- (2) clean val-2025: all conditions incl. encoder/topics ----------
    print("\n=== (2) CLEAN val-2025 — IC / R2 (encoder & topic conditions are valid here) ===")
    print(f"{'condition':26s} {'IC':>7s} {'R2_log':>7s}  n")

    def show(name, X, mk):
        r = val2025(panel, X, y, mk)
        if r:
            print(f"{name:26s} {r['ic']:7.3f} {r['r2']:7.3f}  {r['n']}")

    show("lagged", blocks["lag"], E.make_hgb)
    show("tfidf_lag", panel, tfidf_lag)
    if has_struct:
        show("structured", blocks["structured"], E.make_hgb)
        show("struct + tfidf", tdf2, tfidf_str)
        for enc, (Eenc, Tt) in encs.items():
            se = np.hstack([blocks["structured"], Eenc])
            show(f"struct + enc[{enc}]", se, E.make_hgb)
            if Tt is not None:
                set_ = np.hstack([blocks["structured"], Eenc, Tt])
                show(f"struct + enc + topic[{enc}]", set_, E.make_hgb)
    else:
        for enc, (Eenc, Tt) in encs.items():
            show(f"enc[{enc}] + lag", np.hstack([blocks["lag"], Eenc]), E.make_hgb)

    print("\nWin = structured beats tfidf_lag (~0.545) on backtest IC; and text rows beat 'structured' "
          "on val-2025 IC. Encoder/topic conditions are only leakage-free on val-2025.")


if __name__ == "__main__":
    main()
