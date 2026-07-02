"""
Stage C — call-anchored pilot (decision gate, no new data).

The 2025 call collection postdates the 10-Ks, so calls can't be matched to filings leakage-free. But we
CAN ask the underlying question directly: does an earnings call's TONE predict the volatility that follows
THE CALL? For each call we compute 30-trading-day forward realised vol AFTER the call date (leakage-free,
from the daily prices already on disk) and test whether tone (uncertainty / negativity / risk density)
ranks it.

This is a yes/no gate: if tone has signal here, the proper pre-filing re-collection is worth the time;
if not, we drop calls with a documented reason. Single-window, small n (~299 calls) — supplementary, not
the main filing-level ladder.

Run:  python phase5/call_anchored_pilot.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_config"))
from build_market_features import load_prices            # reuse price loader
from build_call_features import call_records, _extract, _date_of, _ticker_of  # reuse call parser + tone
import json

ANNUALIZE = math.sqrt(252)
WINDOW = 30
TONE = ["call_uncertainty_dens", "call_negative_dens", "call_positive_dens",
        "call_risk_dens", "call_qa_uncertainty", "call_n_words"]


def vols(series, d):
    before = series.index[series.index < d]
    after = series.index[series.index > d]
    lag = float(series.loc[before[-WINDOW:]].std()) * ANNUALIZE if len(before) >= WINDOW else np.nan
    fwd = float(series.loc[after[:WINDOW]].std()) * ANNUALIZE if len(after) >= WINDOW else np.nan
    return lag, fwd


def main():
    calls = call_records().dropna(subset=["ticker", "call_date"])
    calls["ticker"] = calls["ticker"].str.upper().str.strip()
    px = load_prices()
    ret_by = {t: g.set_index("date")["log_ret"].sort_index() for t, g in px.groupby("ticker")}

    rec = []
    for r in calls.itertuples():
        s = ret_by.get(r.ticker)
        if s is None:
            continue
        lag, fwd = vols(s, r.call_date)
        if not np.isfinite(fwd):
            continue
        d = {"ticker": r.ticker, "call_date": r.call_date, "lag": lag, "fwd": fwd}
        for c in TONE:
            d[c] = getattr(r, c)
        rec.append(d)
    df = pd.DataFrame(rec)
    df["logfwd"] = np.log(df["fwd"].clip(lower=1e-6))
    df["loglag"] = np.log(df["lag"].clip(lower=1e-6))
    df["q"] = df["call_date"].dt.to_period("Q").astype(str)
    print(f"Usable calls (with a 30d forward window): {len(df):,} | "
          f"{df.ticker.nunique()} tickers | quarters {sorted(df.q.unique())}\n")

    # baseline: does last-30d vol predict next-30d vol after a call?
    base = spearmanr(df["loglag"], df["logfwd"], nan_policy="omit").statistic
    print(f"BASELINE  persistence (loglag -> logfwd) IC = {base:.3f}\n")

    # residual of logfwd after removing persistence -> tone must explain THIS to add value
    ok = df["loglag"].notna() & df["logfwd"].notna()
    b = np.polyfit(df.loc[ok, "loglag"], df.loc[ok, "logfwd"], 1)
    df["resid"] = df["logfwd"] - (b[0] * df["loglag"] + b[1])

    print(f"{'tone feature':24s} {'IC vs fwd':>10s} {'IC vs resid':>12s} {'within-Q IC':>12s}")
    for c in TONE:
        m = df[c].notna()
        ic = spearmanr(df.loc[m, c], df.loc[m, "logfwd"]).statistic
        icr = spearmanr(df.loc[m, c], df.loc[m, "resid"]).statistic
        wq = [spearmanr(g[c], g["logfwd"]).statistic
              for _, g in df.groupby("q") if g[c].notna().sum() >= 15]
        wqm = float(np.nanmean(wq)) if wq else float("nan")
        print(f"{c:24s} {ic:10.3f} {icr:12.3f} {wqm:12.3f}")

    # multivariate: tone-only and tone+lag, leave-one-quarter-out, pooled IC
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    def loqo_ic(cols):
        pred = np.full(len(df), np.nan)
        for q in df.q.unique():
            tr, te = df.q != q, df.q == q
            if tr.sum() < 30 or te.sum() < 10:
                continue
            est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV())
            est.fit(df.loc[tr, cols], df.loc[tr, "logfwd"])
            pred[te.to_numpy()] = est.predict(df.loc[te, cols])
        m = np.isfinite(pred)
        return spearmanr(df.loc[m, "logfwd"], pred[m]).statistic, int(m.sum())

    ic_tone, n1 = loqo_ic(TONE)
    ic_both, n2 = loqo_ic(TONE + ["loglag"])
    ic_lag, _ = loqo_ic(["loglag"])
    print(f"\nLeave-one-quarter-out pooled IC (hand-crafted tone scalars):")
    print(f"  lag only      IC = {ic_lag:.3f}")
    print(f"  tone only     IC = {ic_tone:.3f}   (n={n1})")
    print(f"  tone + lag    IC = {ic_both:.3f}   (n={n2})")

    # --- fair test: FULL call transcript TF-IDF (mirrors what made 10-K text win) -----------------
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
    df["text"] = [texts.get((r.ticker, pd.Timestamp(r.call_date).normalize()), "") for r in df.itertuples()]
    has_text = df["text"].str.len() > 200

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack, csr_matrix

    def loqo_text_ic(with_lag):
        pred = np.full(len(df), np.nan)
        for q in df.q.unique():
            tr = (df.q != q) & has_text
            te = (df.q == q) & has_text
            if tr.sum() < 30 or te.sum() < 8:
                continue
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                  max_features=8000, sublinear_tf=True)
            Xtr, Xte = vec.fit_transform(df.loc[tr, "text"]), vec.transform(df.loc[te, "text"])
            if with_lag:
                Xtr = hstack([Xtr, csr_matrix(df.loc[tr, ["loglag"]].fillna(df["loglag"].median()).to_numpy())]).tocsr()
                Xte = hstack([Xte, csr_matrix(df.loc[te, ["loglag"]].fillna(df["loglag"].median()).to_numpy())]).tocsr()
            m = Ridge(alpha=10.0, solver="lsqr", max_iter=1000).fit(Xtr, df.loc[tr, "logfwd"])
            pred[te.to_numpy()] = m.predict(Xte)
        ok = np.isfinite(pred)
        return spearmanr(df.loc[ok, "logfwd"], pred[ok]).statistic, int(ok.sum())

    ic_txt, n3 = loqo_text_ic(False)
    ic_txtlag, n4 = loqo_text_ic(True)
    print(f"\nLeave-one-quarter-out pooled IC (FULL transcript TF-IDF, n_text={int(has_text.sum())}):")
    print(f"  call-text only      IC = {ic_txt:.3f}   (n={n3})")
    print(f"  call-text + lag     IC = {ic_txtlag:.3f}   (n={n4})")

    print(f"\nGATE — calls are worth re-collecting only if text/tone beats persistence on the residual:")
    print(f"  lag-only {ic_lag:.3f} | tone+lag {ic_both:.3f} | calltext+lag {ic_txtlag:.3f}")


if __name__ == "__main__":
    main()
