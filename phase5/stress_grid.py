"""
E1 + E3 — the TF-IDF stress-test driver on the P0-corrected data.

Two modes:

  --mode val2025 (default)
      Fair representation x head grid on the clean val-2025 split (train<2025, eval 2025):
      every dense representation (TF-IDF->SVD-256, each encoder with an emb_<e>_fixed cache,
      topic exposures, pooling variants) gets BOTH an imputed-Ridge and an HGB head on top of
      the fixed [log_lag | structured] base; the sparse TF-IDF lane gets an in-fold alpha grid
      (the old fixed alpha=10 was never tuned). Adds the never-run "everything" model
      (struct + TF-IDF-SVD + best encoder [+ topics] [+ change]) with a sparse-Ridge twin.
      Each row: IC, R2_log, bootstrap CI on IC, Diebold-Mariano p vs the struct+TF-IDF reference.

  --mode backtest --test-start {2018|2013}
      Expanding-window ladder over the leakage-free families (lagged / structured / tfidf+lag /
      struct+tfidf / [+change once built]), with per-year ICs and a paired across-years test vs
      the structured baseline. 2013 start = 12 test years (the primary significance lens; the
      single-year val-2025 IC has a +-0.06 CI at n~400, so val deltas alone prove nothing).

Anchors (P0-corrected panel, re-baselined 2026-07-02): struct[hgb] 0.558, tfidf+lag 0.508,
struct+tfidf 0.610. The PRE-fix numbers (0.570/0.545/0.611) are not comparable — they paired
texts with the next filing's labels (see dataset_config/fix_filing_join.py).

tfidf+lag was 0.541 until 2026-08-12. Both figures are correct for the corrected join and
differ only in the sparse-ridge penalty: 0.541 is the fixed alpha=10 result, 0.508 is what this
harness produces when alpha is tuned on 2024 over SPARSE_ALPHAS and selects 30. The anchor was
therefore checking tuned output against a fixed-alpha reference and firing DRIFT on every run.
Both are recorded side by side ("tfidf+lag 0.508 tuned vs 0.541 at fixed
alpha=10"). 0.508 is the figure this code path actually computes, so it is the one anchored.

Output: phase5/out/stress_grid_<mode>*.json + printed tables.
Run:    python phase5/stress_grid.py --mode val2025
        python phase5/stress_grid.py --mode backtest --test-start 2013
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase5"))
import eval_common as E  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

OUT_DIR = ROOT / "phase5" / "out"
ENCODERS = ["dual", "sbert", "volaware", "three", "three_lora", "ftvol", "bge"]
SPARSE_ALPHAS = (3.0, 10.0, 30.0)
ANCHORS = {"structured [hgb]": 0.558, "struct+tfidf [sparse]": 0.610, "tfidf+lag [sparse]": 0.508}


def ic_of(yt, yp):
    return float(spearmanr(yt, yp).statistic)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val2025", "backtest"], default="val2025")
    p.add_argument("--test-start", type=int, default=2018, dest="test_start")
    p.add_argument("--pool-encoders", type=int, default=2,
                   help="how many top encoders also get risk_weighted/topk_risk pooling rows")
    p.add_argument("--enc", type=str, default="",
                   help="backtest mode: comma list of encoder lanes to add (e.g. ftvol2018). "
                        "ONLY admissible for encoders whose training never saw the test years' "
                        "labels — the vol-supervised ftvol/volaware checkpoints trained on "
                        "pre-2025 labels do NOT qualify for pre-2025 test windows.")
    return p.parse_args()


# --------------------------------------------------------------------------------- shared
def build_blocks(panel, backtest=False):
    Xstr, scols = E.structured_matrix(panel)
    if Xstr is None:
        raise SystemExit("market_features.parquet missing — run build_market_features.py first")
    lag = panel[["log_lag"]].to_numpy()
    base = np.hstack([lag, Xstr])
    # backtest lanes must not see change columns from label-trained encoders (test-year leakage)
    chg, ccols = E.change_matrix(panel, exclude_label_trained=backtest)
    return lag, base, Xstr, scols, chg, ccols


def merge_enc(panel, enc, pool="mean"):
    pe = E.mean_pooled_filings(enc, pool=pool)
    if pe is None:
        return None, []
    ecols = [c for c in pe.columns if c.startswith("e") and c[1:].isdigit()]
    m = panel.merge(pe, on=["ticker", "fiscal_year"], how="left")
    return m[ecols].to_numpy(), ecols


def merge_topics(panel, enc):
    tv = E.topic_filings(enc)
    if tv is None:
        return None, []
    tcols = [c for c in tv.columns if c.startswith("t") and c[1:].isdigit()]
    m = panel.merge(tv, on=["ticker", "fiscal_year"], how="left")
    return m[tcols].to_numpy(), tcols


def fit_sparse(panel, y, numeric_cols, tr, ev, tune=True):
    """TextNumericRidge with a small alpha grid tuned on the last train year (2024)."""
    yrs = panel.year.to_numpy()
    best_a = 10.0
    if tune:
        itr, iev = tr & (yrs < 2024), tr & (yrs == 2024)
        if iev.sum() >= 50:
            scores = {}
            for a in SPARSE_ALPHAS:
                m = E.TextNumericRidge(numeric_cols=numeric_cols, alpha=a)
                m.fit(panel.loc[itr], y[itr])
                scores[a] = ic_of(y[iev], m.predict(panel.loc[iev]))
            best_a = max(scores, key=scores.get)
    m = E.TextNumericRidge(numeric_cols=numeric_cols, alpha=best_a)
    m.fit(panel.loc[tr], y[tr])
    return m.predict(panel.loc[ev]), best_a


# --------------------------------------------------------------------------------- val2025
def run_val2025(args):
    panel = E.load_panel().sort_values("year").reset_index(drop=True)
    y = panel.log_fwd.to_numpy()
    yrs = panel.year.to_numpy()
    tr, ev = yrs < 2025, yrs == 2025
    print(f"panel {len(panel):,} | train {tr.sum():,} | val-2025 {ev.sum():,} | fixed={E.USE_FIXED}")

    lag, base, Xstr, scols, chg, ccols = build_blocks(panel)
    yt = y[ev]
    rows, preds = [], {}

    def add(name, yp, extra=None):
        r = {"name": name, "ic": ic_of(yt, yp), "r2": E.r2_only(yt, yp),
             "ic_ci": E.bootstrap_ci(yt, yp, ic_of)}
        if "struct+tfidf [sparse]" in preds:
            r["dm_p_vs_ref"] = E.paired_dm_test(yt, yp, preds["struct+tfidf [sparse]"])[1]
        if extra:
            r.update(extra)
        rows.append(r)
        preds[name] = yp
        ci = r["ic_ci"]
        print(f"{name:38s} IC {r['ic']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  R2 {r['r2']:6.3f}"
              + (f"  DMp {r.get('dm_p_vs_ref', float('nan')):.3f}" if "dm_p_vs_ref" in r else ""))

    def dense(X, mk):
        m = mk(); m.fit(X[tr], y[tr]); return m.predict(X[ev])

    # ---- reference + baselines -----------------------------------------------------------
    tdfs = panel.copy()
    for i, c in enumerate(scols):
        tdfs[c] = Xstr[:, i]
    yp, a = fit_sparse(tdfs, y, ["log_lag"] + scols, tr, ev)
    add("struct+tfidf [sparse]", yp, {"alpha": a})
    add("lagged [hgb]", dense(lag, E.make_hgb))
    add("structured [ridge]", dense(base, E.make_imputed_ridge))
    add("structured [hgb]", dense(base, E.make_hgb))
    yp, a = fit_sparse(panel, y, ["log_lag"], tr, ev)
    add("tfidf+lag [sparse]", yp, {"alpha": a})

    # ---- TF-IDF -> SVD-256 (the missing TF-IDF x HGB lane) --------------------------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                          max_features=20000, sublinear_tf=True)
    Ztr = vec.fit_transform(panel.loc[tr, "text"])
    svd = TruncatedSVD(n_components=256, random_state=42).fit(Ztr)
    Zsvd = np.full((len(panel), 256), np.nan)
    Zsvd[tr] = svd.transform(Ztr)
    Zsvd[ev] = svd.transform(vec.transform(panel.loc[ev, "text"]))
    Xsvd = np.hstack([base, Zsvd])
    add("struct+tfidf_svd [ridge]", dense(Xsvd, E.make_imputed_ridge))
    add("struct+tfidf_svd [hgb]", dense(Xsvd, E.make_hgb))

    # ---- encoders x heads ------------------------------------------------------------------
    enc_data, enc_hgb_ic = {}, {}
    for enc in ENCODERS:
        Eenc, ecols = merge_enc(panel, enc)
        if Eenc is None:
            continue
        enc_data[enc] = (Eenc, ecols)
        X = np.hstack([base, Eenc])
        add(f"struct+enc[{enc}] [ridge]", dense(X, E.make_imputed_ridge))
        yp = dense(X, E.make_hgb)
        add(f"struct+enc[{enc}] [hgb]", yp)
        enc_hgb_ic[enc] = ic_of(yt, yp)
        Tt, tcols = merge_topics(panel, enc)
        if Tt is not None:
            add(f"struct+topic[{enc}] [hgb]", dense(np.hstack([base, Tt]), E.make_hgb))
            add(f"struct+enc+topic[{enc}] [hgb]", dense(np.hstack([base, Eenc, Tt]), E.make_hgb))

    # pooling variants for the strongest encoders
    for enc in sorted(enc_hgb_ic, key=enc_hgb_ic.get, reverse=True)[: args.pool_encoders]:
        for pool in ("risk_weighted", "topk_risk"):
            Ep, _ = merge_enc(panel, enc, pool=pool)
            if Ep is not None:
                add(f"struct+enc[{enc},{pool}] [hgb]", dense(np.hstack([base, Ep]), E.make_hgb))

    # ---- change features --------------------------------------------------------------------
    if chg is not None:
        add("struct+change [hgb]", dense(np.hstack([base, chg]), E.make_hgb))
        tdfc = panel.copy()
        for i, c in enumerate(ccols):
            tdfc[c] = chg[:, i]
        for i, c in enumerate(scols):
            tdfc[c] = Xstr[:, i]
        yp, a = fit_sparse(tdfc, y, ["log_lag"] + scols + ccols, tr, ev)
        add("struct+tfidf+change [sparse]", yp, {"alpha": a})
    else:
        print("(change features not built — run dataset_config/build_change_features.py)")

    # ---- the everything model ---------------------------------------------------------------
    if enc_hgb_ic:
        best = max(enc_hgb_ic, key=enc_hgb_ic.get)
        Eb, ecols = enc_data[best]
        parts, tag = [base, Zsvd, Eb], f"svd+enc[{best}]"
        Tt, _ = merge_topics(panel, best)
        if Tt is not None:
            parts.append(Tt); tag += "+topic"
        if chg is not None:
            parts.append(chg); tag += "+chg"
        add(f"EVERYTHING {tag} [hgb]", dense(np.hstack(parts), E.make_hgb))
        # sparse twin: full TF-IDF + everything dense as numeric cols
        tdfe = panel.copy()
        num = ["log_lag"] + scols
        for i, c in enumerate(scols):
            tdfe[c] = Xstr[:, i]
        for i, c in enumerate(ecols):
            tdfe[c] = Eb[:, i]
        num += ecols
        if chg is not None:
            for i, c in enumerate(ccols):
                tdfe[c] = chg[:, i]
            num += ccols
        yp, a = fit_sparse(tdfe, y, num, tr, ev)
        add(f"EVERYTHING tfidf+enc[{best}]{'+chg' if chg is not None else ''} [sparse]", yp, {"alpha": a})
    else:
        print("(no encoder caches with the current suffix — submit topics/run_encode.sh first)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"stress_grid_val2025{E.CACHE_SUFFIX}.json"
    json.dump({"anchors_expected": ANCHORS, "fixed_data": E.USE_FIXED, "rows": rows},
              open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")

    print("\nAnchor check (corrected-panel anchors; >0.02 drift = wiring problem):")
    for name, ref in ANCHORS.items():
        got = next((r["ic"] for r in rows if r["name"] == name), None)
        if got is not None:
            flag = "OK " if abs(got - ref) <= 0.02 else "DRIFT"
            print(f"  [{flag}] {name}: got {got:.3f} vs anchor {ref:.3f}")


# --------------------------------------------------------------------------------- backtest
def run_backtest(args):
    panel = E.load_panel().sort_values("year").reset_index(drop=True)
    y = panel.log_fwd.to_numpy()
    yrs = panel.year.to_numpy()
    lagv = panel.log_lag.to_numpy()
    test_years = list(range(args.test_start, 2025))
    print(f"panel {len(panel):,} | test years {test_years[0]}..{test_years[-1]} "
          f"({len(test_years)}) | fixed={E.USE_FIXED}")

    lag, base, Xstr, scols, chg, ccols = build_blocks(panel, backtest=True)
    print(f"change cols (label-trained-encoder cols excluded): {ccols}")
    tdf2 = panel.copy()
    for i, c in enumerate(scols):
        tdf2[c] = Xstr[:, i]

    conds = {
        "lagged [hgb]": (lag, E.make_hgb),
        "structured [hgb]": (base, E.make_hgb),
        "structured [ridge]": (base, E.make_imputed_ridge),
        "tfidf+lag [sparse]": (panel, lambda: E.TextNumericRidge(numeric_cols=["log_lag"])),
        "struct+tfidf [sparse]": (tdf2, lambda: E.TextNumericRidge(numeric_cols=["log_lag"] + scols)),
    }
    if chg is not None:
        conds["struct+change [hgb]"] = (np.hstack([base, chg]), E.make_hgb)
        tdfc = tdf2.copy()
        for i, c in enumerate(ccols):
            tdfc[c] = chg[:, i]
        conds["struct+tfidf+change [sparse]"] = (
            tdfc, lambda: E.TextNumericRidge(numeric_cols=["log_lag"] + scols + ccols))

    # encoder lanes (frozen embeddings; admissible iff the encoder never saw test-year labels)
    for enc in [e.strip() for e in args.enc.split(",") if e.strip()]:
        Eenc, _ = merge_enc(panel, enc)
        if Eenc is None:
            print(f"(no emb_{enc}{E.CACHE_SUFFIX}.npy cache — skipping encoder lane)")
            continue
        Xe = np.hstack([base, Eenc])
        conds[f"struct+enc[{enc}] [ridge]"] = (Xe, E.make_imputed_ridge)
        conds[f"struct+enc[{enc}] [hgb]"] = (Xe, E.make_hgb)

    per, aggs = {}, {}
    for name, (X, mk) in conds.items():
        pred = E.rolling_predict(X, y, yrs, mk, test_years=test_years)
        per[name] = E.per_year_metrics(y, pred, yrs, test_years, lag=lagv)
        aggs[name] = E.aggregate_cs(per[name])
        a = aggs[name]
        print(f"{name:30s} mean_IC {a['mean_ic']:.3f}  IC_t {a['ic_t']:5.2f}  CS_R2 {a['cs_r2']:6.3f}")

    print("\nPaired across-years tests vs structured [hgb] (IC):")
    tests = {}
    for name in conds:
        if name == "structured [hgb]":
            continue
        d, p = E.paired_year_test(per[name], per["structured [hgb]"], key="spear")
        tests[name] = {"d_ic": d, "p": p}
        print(f"  {name:30s} dIC {d:+.3f}  p {p:.4f}")

    print("\nPaired across-years tests vs struct+tfidf [sparse] (IC) — the count-model reference:")
    tests_ref = {}
    for name in conds:
        if name == "struct+tfidf [sparse]":
            continue
        d, p = E.paired_year_test(per[name], per["struct+tfidf [sparse]"], key="spear")
        tests_ref[name] = {"d_ic": d, "p": p}
        print(f"  {name:30s} dIC {d:+.3f}  p {p:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"stress_grid_backtest_{args.test_start}{E.CACHE_SUFFIX}.json"
    json.dump({"test_years": test_years, "fixed_data": E.USE_FIXED,
               "per_year": per, "aggregate": aggs, "paired_vs_structured": tests,
               "paired_vs_struct_tfidf": tests_ref},
              open(out, "w"), indent=2, default=float)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    args = parse_args()
    (run_val2025 if args.mode == "val2025" else run_backtest)(args)
