"""
Stage C — the COMBINED gate (does calls help on top of the FULL model, not just lag?).

The first pilot only tested call tone vs persistence. This tests the real question: at each call date we
build the SAME 20 structured features (multi-horizon vol, liquidity, momentum, beta, ... reused from
build_market_features.features_for), then ask whether adding call TONE and/or full transcript TF-IDF
improves prediction of post-call 30d forward vol — nonlinearly (HGB) and linearly (ridge) — over the
structured baseline alone. Leave-one-quarter-out, leakage-free. Still small n (~152), single regime, so
this is a decision gate, not proof.

Run:  python phase5/call_combined_gate.py
"""

import math
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_config"))
from build_market_features import load_prices, market_index, features_for
from build_call_features import call_records, _extract, _date_of, _ticker_of

ANNUALIZE = math.sqrt(252)
WINDOW = 30
TONE = ["call_uncertainty_dens", "call_negative_dens", "call_positive_dens",
        "call_risk_dens", "call_qa_uncertainty", "call_n_words"]


def fwd_vol(series, d):
    after = series.index[series.index > d]
    return float(series.loc[after[:WINDOW]].std()) * ANNUALIZE if len(after) >= WINDOW else np.nan


def main():
    calls = call_records().dropna(subset=["ticker", "call_date"])
    calls["ticker"] = calls["ticker"].str.upper().str.strip()
    # transcript text per (ticker, date)
    texts = {}
    for fp in (ROOT / "datasets" / "calls").rglob("*.json"):
        try:
            o = json.load(open(fp))
        except Exception:
            continue
        o = o[0] if isinstance(o, list) else o
        full, _ = _extract(o)
        t, d = _ticker_of(o, fp.name), _date_of(o, fp.name)
        if t and pd.notna(d):
            texts[(t, pd.Timestamp(d).normalize())] = full

    px = load_prices()
    mkt = market_index(px)
    ret_by = {t: g.set_index("date")["log_ret"].sort_index() for t, g in px.groupby("ticker")}
    dv_by = {t: g.set_index("date")["dollar_vol"].sort_index() for t, g in px.groupby("ticker")}

    rows = []
    for r in calls.itertuples():
        s = ret_by.get(r.ticker)
        if s is None:
            continue
        fwd = fwd_vol(s, r.call_date)
        if not np.isfinite(fwd):
            continue
        feat = features_for(s, dv_by.get(r.ticker), mkt, r.call_date)  # full structured block @ call date
        if not feat:
            continue
        feat["logfwd"] = math.log(max(fwd, 1e-6))
        feat["q"] = pd.Timestamp(r.call_date).to_period("Q").__str__()
        feat["text"] = texts.get((r.ticker, pd.Timestamp(r.call_date).normalize()), "")
        for c in TONE:
            feat[c] = getattr(r, c)
        rows.append(feat)

    df = pd.DataFrame(rows)
    scols = [c for c in df.columns if c not in (["logfwd", "q", "text"] + TONE)]
    has_text = df["text"].str.len() > 200
    print(f"Usable calls: {len(df):,} | {df['q'].nunique()} quarters | structured cols: {len(scols)} | "
          f"with text: {int(has_text.sum())}\n")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix

    def hgb(): return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                                    l2_regularization=1.0, random_state=42)

    def loqo(cols, kind="hgb"):
        pred = np.full(len(df), np.nan)
        for q in df["q"].unique():
            tr, te = (df["q"] != q).to_numpy(), (df["q"] == q).to_numpy()
            if tr.sum() < 30 or te.sum() < 8:
                continue
            if kind == "hgb":
                est = hgb(); est.fit(df.loc[tr, cols], df.loc[tr, "logfwd"])
                pred[te] = est.predict(df.loc[te, cols])
            else:  # ridge on imputed+scaled dense
                imp = SimpleImputer(strategy="median").fit(df.loc[tr, cols])
                sc = StandardScaler().fit(imp.transform(df.loc[tr, cols]))
                Xtr, Xte = sc.transform(imp.transform(df.loc[tr, cols])), sc.transform(imp.transform(df.loc[te, cols]))
                m = Ridge(alpha=10.0).fit(Xtr, df.loc[tr, "logfwd"]); pred[te] = m.predict(Xte)
        ok = np.isfinite(pred)
        return spearmanr(df.loc[ok, "logfwd"], pred[ok]).statistic, int(ok.sum())

    def loqo_text(struct_cols):
        """structured (dense) + transcript TF-IDF, ridge, leave-one-quarter-out."""
        pred = np.full(len(df), np.nan)
        for q in df["q"].unique():
            tr = ((df["q"] != q) & has_text).to_numpy()
            te = ((df["q"] == q) & has_text).to_numpy()
            if tr.sum() < 30 or te.sum() < 8:
                continue
            imp = SimpleImputer(strategy="median").fit(df.loc[tr, struct_cols])
            sc = StandardScaler().fit(imp.transform(df.loc[tr, struct_cols]))
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                  max_features=8000, sublinear_tf=True)
            Ttr, Tte = vec.fit_transform(df.loc[tr, "text"]), vec.transform(df.loc[te, "text"])
            Xtr = hstack([csr_matrix(sc.transform(imp.transform(df.loc[tr, struct_cols]))), Ttr]).tocsr()
            Xte = hstack([csr_matrix(sc.transform(imp.transform(df.loc[te, struct_cols]))), Tte]).tocsr()
            m = Ridge(alpha=10.0, solver="lsqr", max_iter=1000).fit(Xtr, df.loc[tr, "logfwd"])
            pred[te] = m.predict(Xte)
        ok = np.isfinite(pred)
        return spearmanr(df.loc[ok, "logfwd"], pred[ok]).statistic, int(ok.sum())

    base, nb = loqo(scols, "hgb")
    tone, nt = loqo(scols + TONE, "hgb")
    base_r, _ = loqo(scols, "ridge")
    text_r, ntx = loqo_text(scols)
    print("Leave-one-quarter-out pooled IC (predicting post-call 30d vol):")
    print(f"  structured only         (HGB)   IC = {base:.3f}   (n={nb})")
    print(f"  structured + tone       (HGB)   IC = {tone:.3f}   (n={nt})   delta {tone - base:+.3f}")
    print(f"  structured only        (ridge)  IC = {base_r:.3f}")
    print(f"  structured + call-text (ridge)  IC = {text_r:.3f}   (n={ntx})   delta {text_r - base_r:+.3f}")
    print(f"\nGATE: calls worth re-collecting if structured+tone or structured+call-text clearly beats "
          f"structured-only. Small n ({len(df)}), single regime — read deltas as suggestive.")


if __name__ == "__main__":
    main()
