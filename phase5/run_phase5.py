"""
Phase 5 — Tier-3 (learned representations) + hybrid scoring for 30-day forward-volatility prediction.

Drops the domain encoders and BERTopic topic-exposure vectors into the SAME harness as the Tier-1/Tier-2
baselines (`baselines/run_baselines.py`) — same filing set, same temporal split, same log-target, same
`metrics()` — so every tier is directly comparable and the numbers slot straight into the report's
Phase-5 ablation table. This is the real thesis go/no-go: do the learned representations beat the
TF-IDF+lagged floor (R²_log≈0.18 / Spearman≈0.60 on val 2025)?

Per encoder, three representation conditions, each through a RidgeCV AND a small MLP head (report §3.1.5):
  Tier-3a  encoder (mean-pooled) : average the encoder's paragraph embeddings to one 768-d filing vector
  Tier-3b  topic exposure        : the BERTopic per-filing t0..t{K-1} fractions
  Tier-3c  hybrid                : [mean-pooled encoder | topic vector | log_lagged]   (condition-7)

Artifacts (produced in Phase 4):
  topics/out/emb_<enc>.npy                      doc embeddings, row-aligned 1:1 with topic_docs.jsonl
  topics/data/topic_docs.jsonl                  (ticker, fiscal_year, split) per doc — the pooling key
  topics/out/filing_topic_vectors_<enc>.parquet per-filing t0..t{K-1} + keys
A condition is silently skipped for an encoder whose artifact isn't on disk yet (so this runs on sbert
today and fills in dual/three/three_lora as the BERTopic job lands).

Leakage-safe: StandardScaler / RidgeCV / MLP are all fit on TRAIN filings only, applied to the eval split.

Outputs:
  phase5/out/phase5_results.json   (per encoder × condition × head metrics + the Tier-1/2 floor echoed in)
  printed comparison table

Usage:
  python phase5/run_phase5.py                          # all available encoders, eval on val 2025
  python phase5/run_phase5.py --encoders sbert         # dry-run one encoder
  python phase5/run_phase5.py --eval_split test        # once 2026 labels are computed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines.run_baselines import load_filing_frame, metrics  # reuse: same rows / split / metrics

TOPIC_DOCS = ROOT / "topics" / "data" / "topic_docs.jsonl"
TOPIC_OUT = ROOT / "topics" / "out"
BASELINE_JSON = ROOT / "baselines" / "out" / "baseline_results.json"
OUT_DIR = ROOT / "phase5" / "out"

ALL_ENCODERS = ["sbert", "dapt", "dual", "three", "three_lora"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", type=str, default=None,
                   help="comma list; default = every encoder with at least one artifact on disk")
    p.add_argument("--eval_split", choices=["val", "test"], default="val")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def mean_pooled_filings(enc):
    """Average emb_<enc>.npy paragraph rows to one vector per (ticker, fiscal_year). None if no cache."""
    cache = TOPIC_OUT / f"emb_{enc}.npy"
    if not cache.exists():
        return None
    emb = np.load(cache)
    keys = pd.read_json(TOPIC_DOCS, lines=True)[["ticker", "fiscal_year"]]
    if len(keys) != emb.shape[0]:
        raise SystemExit(f"[{enc}] emb rows {emb.shape[0]} != docs {len(keys)} — re-run Phase 4 encode")
    groups = keys.groupby(["ticker", "fiscal_year"]).indices  # (ticker, fy) -> row indices
    rows, vecs = [], []
    for k, idx in groups.items():
        rows.append(k)
        vecs.append(emb[idx].mean(axis=0))
    out = pd.DataFrame(vecs, columns=[f"e{i}" for i in range(emb.shape[1])])
    out[["ticker", "fiscal_year"]] = pd.DataFrame(rows, columns=["ticker", "fiscal_year"])
    return out


def topic_filings(enc):
    """Per-filing topic-exposure vectors (t0..t{K-1}) keyed by (ticker, fiscal_year). None if absent."""
    pq = TOPIC_OUT / f"filing_topic_vectors_{enc}.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    tcols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]
    return df[["ticker", "fiscal_year"] + tcols].copy()


def fit_score(Xtr, ytr, Xev, yev, seed):
    """Fit RidgeCV + a small MLP on train (scaled), return {head: metrics} on eval."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xev_s = sc.transform(Xtr), sc.transform(Xev)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr_s, ytr)
    # small + strongly-regularized head: 768 dense dims on ~7k filings overfits hard otherwise
    mlp = MLPRegressor(hidden_layer_sizes=(64,), alpha=1.0, max_iter=1000,
                       early_stopping=True, n_iter_no_change=20,
                       random_state=seed).fit(Xtr_s, ytr)
    return {"ridge": metrics(yev, ridge.predict(Xev_s)),
            "mlp": metrics(yev, mlp.predict(Xev_s))}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    base = load_filing_frame()  # canonical filing set (same rows as the baselines)
    tr_mask = base["split"] == "train"
    ev_mask = base["split"] == args.eval_split
    print(f"Filings: train={int(tr_mask.sum())} | {args.eval_split}={int(ev_mask.sum())} "
          f"(same set as baselines)")
    if ev_mask.sum() == 0:
        raise SystemExit(f"No scorable rows in '{args.eval_split}' (2026 labels likely not computed).")

    encoders = ([e.strip() for e in args.encoders.split(",")] if args.encoders
                else [e for e in ALL_ENCODERS
                      if (TOPIC_OUT / f"emb_{e}.npy").exists()
                      or (TOPIC_OUT / f"filing_topic_vectors_{e}.parquet").exists()])
    print(f"Encoders with artifacts: {encoders}\n")

    results = {}
    for enc in encoders:
        pooled = mean_pooled_filings(enc)         # Tier-3a feature source
        topics = topic_filings(enc)               # Tier-3b feature source
        conditions = {}

        # feature columns are e<int> (encoder dims) / t<int> (topics); guard against 'ticker' etc.
        def ecols(d): return [c for c in d.columns if c.startswith("e") and c[1:].isdigit()]
        def tcols(d): return [c for c in d.columns if c.startswith("t") and c[1:].isdigit()]

        # assemble each condition's feature frame, merged onto the canonical filing set
        feats = {}
        if pooled is not None:
            feats["tier3a_encoder"] = (pooled, ecols(pooled))
        if topics is not None:
            feats["tier3b_topic"] = (topics, tcols(topics))
        if pooled is not None and topics is not None:
            hyb = pooled.merge(topics, on=["ticker", "fiscal_year"], how="inner")
            feats["tier3c_hybrid"] = (hyb, ecols(hyb) + tcols(hyb))

        for name, (fdf, cols) in feats.items():
            m = base.merge(fdf, on=["ticker", "fiscal_year"], how="inner")
            tr = m[m["split"] == "train"]; ev = m[m["split"] == args.eval_split]
            X = cols + (["log_lag"] if name == "tier3c_hybrid" else [])  # hybrid adds lagged anchor
            scores = fit_score(tr[X].to_numpy(), tr["log_fwd"].to_numpy(),
                               ev[X].to_numpy(), ev["log_fwd"].to_numpy(), args.seed)
            conditions[name] = {"n_train": int(len(tr)), "n_eval": int(len(ev)),
                                "n_features": len(X), **scores}
            print(f"  [{enc}] {name:14s} ridge R2={scores['ridge']['r2_log']:.4f} "
                  f"rho={scores['ridge']['spearman']:.4f} | "
                  f"mlp R2={scores['mlp']['r2_log']:.4f} rho={scores['mlp']['spearman']:.4f}")
        if not feats:
            print(f"  [{enc}] no artifacts on disk yet — skipped")
        results[enc] = conditions

    floor = json.loads(BASELINE_JSON.read_text())["results"] if BASELINE_JSON.exists() else {}
    payload = {"eval_split": args.eval_split, "target": "log(fwd_vol_30d)",
               "baseline_floor": floor, "results": results}
    (out_dir / "phase5_results.json").write_text(json.dumps(payload, indent=2))

    # comparison table: floor first, then learned tiers (best head per condition)
    print(f"\n=== Phase 5 on '{args.eval_split}', target=log(fwd_vol_30d) "
          f"(higher R²/Spearman, lower RMSE = better) ===")
    print(f"{'tier / encoder':30s} {'head':6s} {'R2_log':>8s} {'Spearman':>9s} {'RMSE_log':>9s}")
    for name, m in floor.items():
        print(f"{name:30s} {'-':6s} {m['r2_log']:8.4f} {m['spearman']:9.4f} {m['rmse_log']:9.4f}")
    for enc, conds in results.items():
        for cname, c in conds.items():
            for head in ("ridge", "mlp"):
                print(f"{enc+' / '+cname:30s} {head:6s} {c[head]['r2_log']:8.4f} "
                      f"{c[head]['spearman']:9.4f} {c[head]['rmse_log']:9.4f}")
    print(f"\nWrote {out_dir / 'phase5_results.json'}")
    print("Beat tier2plus_tfidf_lagged (R²_log≈0.18 / Spearman≈0.60) to justify the learned pipeline.")


if __name__ == "__main__":
    main()
