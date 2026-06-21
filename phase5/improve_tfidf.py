"""
Lever 2: push the winning TF-IDF+lagged model higher. Tests improved text-feature variants on clean
val-2025, all with log_lag included. Higher Spearman (IC) / R²_log = better.

Variants:
  base        : word (1,2)-grams, 20k feats          (current floor)
  trigrams    : word (1,3)-grams, 50k feats
  char        : base + char_wb (3,5)-grams           (catches risk sub-words / morphology)
  lm          : base + Loughran-McDonald-style risk/uncertainty density features
  fselect     : base, then univariate f_regression top-k feature selection (denoise)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines.run_baselines import load_filing_frame, metrics
import eval_common as E

ALPHAS = np.logspace(-2, 3, 20)
# a few LM-style categories (small, illustrative; swap for the full LM master dictionary if available)
LM = {
    "risk": E.RISK_TERMS,
    "uncertain": set("uncertain uncertainty uncertainties may might could possibly probable approximate "
                     "contingent depend depends pending unpredictable indefinite assume".split()),
    "litigious": set("litigation lawsuit lawsuits claim claims plaintiff defendant court judicial "
                     "settlement damages liable allegation subpoena".split()),
    "negative": set("loss losses decline declines adverse adversely fail failure decrease weak "
                    "deteriorate impairment default shortfall".split()),
}


def lm_features(texts):
    feats = np.zeros((len(texts), len(LM)), dtype=float)
    for i, t in enumerate(texts):
        toks = [w.strip(".,();:'\"") for w in t.lower().split()]
        n = len(toks) or 1
        for j, (_, vocab) in enumerate(LM.items()):
            feats[i, j] = sum(w in vocab for w in toks) / n
    return feats


def ridge_eval(Xtr, ytr, Xev, yev):
    m = metrics(yev, E.fast_ridge_predict(Xtr, ytr, Xev))
    return m["spearman"], m["r2_log"]


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    from scipy.sparse import hstack, csr_matrix

    df = load_filing_frame()
    tr, ev = df[df.split == "train"], df[df.split == "val"]
    ytr, yev = tr["log_fwd"].to_numpy(), ev["log_fwd"].to_numpy()
    lag_tr = csr_matrix(tr[["log_lag"]].to_numpy()); lag_ev = csr_matrix(ev[["log_lag"]].to_numpy())
    print(f"val-2025 n={len(ev)}. All variants include log_lag.\n")
    print(f"{'variant':12s} {'IC(rho)':>8s} {'R2_log':>8s}")

    def word_vec(ng, mf):
        return TfidfVectorizer(stop_words="english", ngram_range=ng, min_df=5,
                               max_features=mf, sublinear_tf=True)

    # base
    v = word_vec((1, 2), 20000); Atr, Aev = v.fit_transform(tr.text), v.transform(ev.text)
    base_tr = hstack([Atr, lag_tr]).tocsr(); base_ev = hstack([Aev, lag_ev]).tocsr()
    print("%-12s %8.3f %8.3f" % (("base",) + ridge_eval(base_tr, ytr, base_ev, yev)))

    # trigrams
    v = word_vec((1, 3), 50000); Atr, Aev = v.fit_transform(tr.text), v.transform(ev.text)
    print("%-12s %8.3f %8.3f" % (("trigrams",) + ridge_eval(
        hstack([Atr, lag_tr]).tocsr(), ytr, hstack([Aev, lag_ev]).tocsr(), yev)))

    # char n-grams added to base
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=40000, sublinear_tf=True)
    Ctr, Cev = cv.fit_transform(tr.text), cv.transform(ev.text)
    v = word_vec((1, 2), 20000); Wtr, Wev = v.fit_transform(tr.text), v.transform(ev.text)
    print("%-12s %8.3f %8.3f" % (("char",) + ridge_eval(
        hstack([Wtr, Ctr, lag_tr]).tocsr(), ytr, hstack([Wev, Cev, lag_ev]).tocsr(), yev)))

    # LM density features added to base
    sc = StandardScaler().fit(lm_features(tr.text.tolist()))
    Ltr = csr_matrix(sc.transform(lm_features(tr.text.tolist())))
    Lev = csr_matrix(sc.transform(lm_features(ev.text.tolist())))
    print("%-12s %8.3f %8.3f" % (("lm",) + ridge_eval(
        hstack([Wtr, Ltr, lag_tr]).tocsr(), ytr, hstack([Wev, Lev, lag_ev]).tocsr(), yev)))

    # supervised feature selection on base words
    k = min(5000, Wtr.shape[1])
    sk = SelectKBest(f_regression, k=k).fit(Wtr, ytr)
    print("%-12s %8.3f %8.3f" % (("fselect",) + ridge_eval(
        hstack([sk.transform(Wtr), lag_tr]).tocsr(), ytr,
        hstack([sk.transform(Wev), lag_ev]).tocsr(), yev)))


if __name__ == "__main__":
    main()
