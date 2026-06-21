"""
Phase 5 — Tier-1 / Tier-2 baselines for 30-day forward-volatility prediction.

Establishes the *floor* the learned representations (encoders + BERTopic vectors) must beat.
All tiers are evaluated on the SAME filings, the SAME split, and the SAME target transform so
the numbers drop straight into the unified Phase-5 comparison later.

Tiers
  Tier-1a  naive persistence : y_hat = lagged_vol_30d            (random walk; 0 params)
  Tier-1b  AR(1)             : RidgeCV  log(fwd) ~ log(lagged)   (fitted persistence)
  Tier-2   TF-IDF            : RidgeCV  log(fwd) ~ tfidf(text)   (risk text only)
  Tier-2+  TF-IDF + lagged   : RidgeCV  log(fwd) ~ tfidf | log(lagged)   (does text add signal?)

Target / split
  Target  : fwd_vol_30d, modelled in LOG space (volatility is ~log-normal; one transform for
            every tier keeps R²/RMSE comparable across Phase 5).
  Split   : train = filing_date < 2025-01-01 ; eval = --eval_split (default 'val' = 2025).
            NOTE: the 2026 'test' fwd_vol labels are not computable yet (forward window not closed,
            1/346 populated) — score 'val' now, re-run with --eval_split test once labels exist.

Data
  datasets/feature_table.parquet  -> lagged_vol_30d, fwd_vol_30d, filing_date, split keys
  topics/data/topic_docs.jsonl    -> per-filing risk text (paragraphs joined), built in Phase 4

Outputs
  baselines/out/baseline_results.json   (per-model metrics + run metadata)
  printed comparison table

Usage
  python baselines/run_baselines.py                      # eval on val (2025)
  python baselines/run_baselines.py --eval_split test    # once 2026 labels are computed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"
TOPIC_DOCS = ROOT / "topics" / "data" / "topic_docs.jsonl"
OUT_DIR = ROOT / "baselines" / "out"

TRAIN_END = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2026-01-01")
EPS = 1e-6  # floor before log (volatility is strictly positive)


def split_of(d: pd.Timestamp) -> str:
    if d < TRAIN_END:
        return "train"
    if d < VAL_END:
        return "val"
    return "test"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_split", choices=["val", "test"], default="val",
                   help="held-out split to score (train is always <2025)")
    p.add_argument("--max_features", type=int, default=20000)
    p.add_argument("--min_df", type=int, default=5)
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def load_filing_frame():
    """One row per filing: target + lagged predictor + joined risk text, with the temporal split."""
    ft = pd.read_parquet(FEATURE_TABLE)
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["split"] = ft["filing_date"].map(split_of)
    ft = ft[["ticker", "cik", "fiscal_year", "filing_date", "split",
             "lagged_vol_30d", "fwd_vol_30d"]].copy()

    # per-filing risk text from the Phase-4 doc set (paragraphs -> one document)
    docs = pd.read_json(TOPIC_DOCS, lines=True)
    text = (docs.groupby(["ticker", "fiscal_year"])
                .agg(text=("text", lambda s: " ".join(s)), n_paras=("text", "size"))
                .reset_index())

    df = ft.merge(text, on=["ticker", "fiscal_year"], how="inner")
    # keep only filings that are fully usable by EVERY tier (fair head-to-head)
    df = df[df["fwd_vol_30d"].notna() & df["lagged_vol_30d"].notna()].copy()
    df["log_fwd"] = np.log(df["fwd_vol_30d"].clip(lower=EPS))
    df["log_lag"] = np.log(df["lagged_vol_30d"].clip(lower=EPS))
    return df


def metrics(y_true_log, y_pred_log):
    """Primary metrics in log space; RMSE also reported back on the original vol scale."""
    from scipy.stats import spearmanr
    err = y_pred_log - y_true_log
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_log - y_true_log.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(y_true_log, y_pred_log).statistic)
    rmse_orig = float(np.sqrt(np.mean((np.exp(y_pred_log) - np.exp(y_true_log)) ** 2)))
    return {"rmse_log": rmse, "mae_log": mae, "r2_log": r2,
            "spearman": rho, "rmse_origscale": rmse_orig}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeCV
    from scipy.sparse import hstack, csr_matrix

    df = load_filing_frame()
    tr = df[df["split"] == "train"]
    ev = df[df["split"] == args.eval_split]
    print(f"Filings: train={len(tr)} | {args.eval_split}={len(ev)} "
          f"(usable = text + lagged + fwd all present)")
    if len(ev) == 0:
        raise SystemExit(f"No scorable rows in '{args.eval_split}' — "
                         f"2026 fwd_vol labels likely not computed yet.")

    y_tr, y_ev = tr["log_fwd"].to_numpy(), ev["log_fwd"].to_numpy()
    alphas = np.logspace(-2, 3, 20)
    results = {}

    # Tier-1a — naive persistence: predict log(fwd) = log(lagged)
    results["tier1a_naive_persistence"] = metrics(y_ev, ev["log_lag"].to_numpy())

    # Tier-1b — AR(1): RidgeCV on the single lagged feature
    ar1 = RidgeCV(alphas=alphas).fit(tr[["log_lag"]], y_tr)
    results["tier1b_ar1"] = metrics(y_ev, ar1.predict(ev[["log_lag"]]))

    # Tier-2 — TF-IDF risk text only
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=args.min_df, max_features=args.max_features,
                          sublinear_tf=True)
    Xtr_txt = vec.fit_transform(tr["text"])
    Xev_txt = vec.transform(ev["text"])
    tfidf = RidgeCV(alphas=alphas).fit(Xtr_txt, y_tr)
    results["tier2_tfidf"] = metrics(y_ev, tfidf.predict(Xev_txt))

    # Tier-2+ — TF-IDF + lagged vol (does text add signal beyond persistence?)
    Xtr = hstack([Xtr_txt, csr_matrix(tr[["log_lag"]].to_numpy())]).tocsr()
    Xev = hstack([Xev_txt, csr_matrix(ev[["log_lag"]].to_numpy())]).tocsr()
    tfidf_lag = RidgeCV(alphas=alphas).fit(Xtr, y_tr)
    results["tier2plus_tfidf_lagged"] = metrics(y_ev, tfidf_lag.predict(Xev))

    payload = {
        "eval_split": args.eval_split,
        "n_train": int(len(tr)), "n_eval": int(len(ev)),
        "target": "log(fwd_vol_30d)",
        "vectorizer": {"max_features": args.max_features, "min_df": args.min_df,
                       "ngram_range": [1, 2], "sublinear_tf": True},
        "results": results,
    }
    (out_dir / "baseline_results.json").write_text(json.dumps(payload, indent=2))

    # comparison table (lower RMSE / higher R² & Spearman = better)
    print(f"\n=== Baselines on '{args.eval_split}' (n={len(ev)}), target=log(fwd_vol_30d) ===")
    print(f"{'model':28s} {'RMSE_log':>9s} {'MAE_log':>8s} {'R2_log':>8s} {'Spearman':>9s} {'RMSE_orig':>10s}")
    for name, m in results.items():
        print(f"{name:28s} {m['rmse_log']:9.4f} {m['mae_log']:8.4f} "
              f"{m['r2_log']:8.4f} {m['spearman']:9.4f} {m['rmse_origscale']:10.4f}")
    print(f"\nWrote {out_dir / 'baseline_results.json'}")
    print("Floor for Phase 5: encoders + topic vectors must beat tier2plus on R2_log / Spearman.")


if __name__ == "__main__":
    main()
