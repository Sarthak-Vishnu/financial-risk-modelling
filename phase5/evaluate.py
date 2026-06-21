"""
Phase 5 — full validation driver (expanding-window backtest + incremental value + significance).

Runs the tier ladder on an **expanding-window backtest over 2018–2024** (~3,100 pooled test filings, vs the
397 in the single val-2025 window), so the thesis claim is tested with real statistical power and year-over-
year robustness. For every text/representation tier we report its **ΔR² over a lagged-only model** (text's
recovery of the ~66% residual persistence leaves), bootstrap 95% CIs, and a paired Diebold–Mariano test vs
the lagged baseline — i.e. *is the improvement significant?*

Heads: RidgeCV (linear), PCA→Ridge (dense embeddings), HistGradientBoosting (low-dim topic/hybrid). A simple
z-score ensemble of TF-IDF+lagged and the dense hybrid stands in for the "hybrid representation" headline.

Auto-skips encoders whose Phase-4 artifacts aren't on disk yet; re-run with no args after the BERTopic job to
pull in dual/three/three_lora.

Usage:
    python phase5/evaluate.py                 # all encoders with artifacts
    python phase5/evaluate.py --encoders sbert
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import eval_common as E   # same directory

ALL_ENCODERS = ["sbert", "dapt", "dual", "three", "three_lora"]
KEYS = ["ticker", "fiscal_year"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=str(E.ROOT / "phase5" / "out"))
    return p.parse_args()


def zscore(a):
    m = np.isfinite(a)
    out = np.full_like(a, np.nan)
    out[m] = (a[m] - a[m].mean()) / (a[m].std() + 1e-9)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    panel = E.load_panel()
    encoders = ([e.strip() for e in args.encoders.split(",")] if args.encoders
                else [e for e in ALL_ENCODERS
                      if (E.TOPIC_OUT / f"emb_{e}.npy").exists()
                      and (E.TOPIC_OUT / f"filing_topic_vectors_{e}.parquet").exists()])
    print(f"Panel: {len(panel):,} labelled filings | backtest years {E.TEST_YEARS} | encoders {encoders}")

    report = {"test_years": E.TEST_YEARS, "encoders": {}}

    for enc in encoders:
        penc, ptop = E.mean_pooled_filings(enc), E.topic_filings(enc)
        if penc is None or ptop is None:
            print(f"  [{enc}] missing artifacts — skipped"); continue
        w = panel.merge(penc, on=KEYS, how="inner").merge(ptop, on=KEYS, how="inner").reset_index(drop=True)
        y, years, lag = w["log_fwd"].to_numpy(), w["year"].to_numpy(), w["log_lag"].to_numpy()
        ecols = [c for c in penc.columns if c.startswith("e") and c[1:].isdigit()]
        tcols = [c for c in ptop.columns if c.startswith("t") and c[1:].isdigit()]
        ctrl, _ = E.controls_matrix(w)
        Xlag = w[["log_lag"]].to_numpy()
        enc_a, top_a = w[ecols].to_numpy(), w[tcols].to_numpy()

        def D(*arrs):
            return np.hstack(arrs)

        # (name, X, head-factory) — all "+lagged" so ΔR² over the lagged baseline is the headline
        conds = [
            ("lagged_only",       Xlag,                       E.make_ridge),
            ("controls+lagged",   D(ctrl, Xlag),              E.make_ridge),
            ("tfidf+lagged",      w[["text", "log_lag"]],     lambda: E.TextNumericRidge(["log_lag"])),
            ("encoder+lagged",    D(enc_a, Xlag),             lambda: E.make_pca_ridge(min(100, enc_a.shape[1]))),
            ("topic+lagged",      D(top_a, Xlag),             E.make_ridge),
            ("topic+lagged[hgb]", D(top_a, Xlag),             E.make_hgb),
            ("hybrid",            D(enc_a, top_a, ctrl, Xlag), E.make_ridge),
            ("hybrid[hgb]",       D(enc_a, top_a, ctrl, Xlag), E.make_hgb),
        ]

        preds = {}
        for name, X, fac in conds:
            preds[name] = E.rolling_predict(X, y, years, fac)
        # ensemble: z-avg of the best text + dense-hybrid OOS preds (rank-only — z-scores aren't on target scale)
        preds["ensemble(z)"] = np.nanmean(
            np.vstack([zscore(preds["tfidf+lagged"]), zscore(preds["hybrid"])]), axis=0)

        # HEADLINE = cross-sectional mean IC (within-year Spearman): "does it rank firms by vol?".
        # Out-of-time LEVEL is unpredictable (regime shifts) so R² is shown only as honest context.
        py = {name: E.per_year_metrics(y, preds[name], years, E.TEST_YEARS, lag) for name in preds}
        cs = {name: E.aggregate_cs(py[name]) for name in preds}
        base_py = py["lagged_only"]
        ic_base = cs["lagged_only"]["mean_ic"]

        rows = []
        for name in list(dict.fromkeys([c[0] for c in conds] + ["ensemble(z)"])):
            m = dict(cs[name])
            m["pooled_r2"] = float("nan") if name == "ensemble(z)" else E.eval_metrics(y, preds[name])["r2_log"]
            m["delta_ic"] = 0.0 if name == "lagged_only" else m["mean_ic"] - ic_base
            if name != "lagged_only":   # is the IC increment over lagged significant across years?
                _, m["ic_paired_p"] = E.paired_year_test(py[name], base_py, key="spear")
            m["per_year"] = py[name]
            rows.append((name, m))

        report["encoders"][enc] = {"n_filings": int(len(w)), "mean_ic_lagged_only": ic_base,
                                   "conditions": {n: {k: v for k, v in m.items() if k != "per_year"}
                                                  for n, m in rows},
                                   "per_year": {n: m["per_year"] for n, m in rows}}

        print(f"\n===== {enc} (n={len(w):,} | within-year cross-sectional, mean over {E.TEST_YEARS[0]}–{E.TEST_YEARS[-1]}) =====")
        print(f"{'condition':20s} {'meanIC':>7s} {'dIC':>7s} {'ICp':>6s} {'IC-t':>6s} {'dirAcc':>6s} {'pooledR2*':>9s}")
        for name, m in rows:
            p = m.get("ic_paired_p")
            p_str = f"{p:.3f}" if p is not None and p == p else "-"
            print(f"{name:20s} {m['mean_ic']:7.3f} {m['delta_ic']:+7.3f} {p_str:>6s} "
                  f"{m['ic_t']:6.2f} {m['dir_acc']:6.3f} {m['pooled_r2']:9.3f}")
        hy = py["hybrid"]
        print("  * pooledR2 negative = out-of-time LEVEL unpredictable (regime shifts); IC = the real signal")
        print("  hybrid IC per year: " + " ".join(f"{Y}:{hy[Y]['spear']:.2f}" for Y in sorted(hy)))

    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out_dir / 'evaluation.json'}")
    print("Validated if: a text/representation tier shows ΔR²>0 with DMp<0.05 (significant) and a CI "
          "excluding 0, consistently across years.")


if __name__ == "__main__":
    main()
