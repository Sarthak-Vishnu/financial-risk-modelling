"""
Path-1 test: does adding the dense encoder + topic vectors ON TOP OF TF-IDF beat TF-IDF alone?

So far "hybrid" never included TF-IDF, so we kept asking "encoder vs TF-IDF" (TF-IDF won). The real
thesis question is whether the learned representations add anything *orthogonal* to bag-of-words. Here the
TF-IDF block is IDENTICAL in both models, so the delta is purely the encoder+topic contribution:

    floor :  TF-IDF + log_lagged
    full  :  TF-IDF + log_lagged + mean-pooled encoder + topic exposure

Clean val-2025 split (encoders genuinely out-of-sample). Higher Spearman / R²_log = better.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines.run_baselines import load_filing_frame, metrics
import eval_common as E

KEYS = ["ticker", "fiscal_year"]


def fit_eval(tr, ev, num_cols, enc_cols=None, top_cols=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from scipy.sparse import hstack, csr_matrix

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                          max_features=20000, sublinear_tf=True)
    Xtr_txt = vec.fit_transform(tr["text"]); Xev_txt = vec.transform(ev["text"])
    cols = list(num_cols) + (enc_cols or []) + (top_cols or [])
    sc = StandardScaler().fit(tr[cols].to_numpy())
    Xtr = hstack([Xtr_txt, csr_matrix(sc.transform(tr[cols].to_numpy()))]).tocsr()
    Xev = hstack([Xev_txt, csr_matrix(sc.transform(ev[cols].to_numpy()))]).tocsr()
    model = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr, tr["log_fwd"].to_numpy())
    m = metrics(ev["log_fwd"].to_numpy(), model.predict(Xev))
    return m["r2_log"], m["spearman"]


def main():
    base = load_filing_frame()
    encoders = [e for e in ["sbert", "dapt", "dual", "three", "three_lora"]
                if (E.TOPIC_OUT / f"emb_{e}.npy").exists()]
    print(f"Clean val-2025 test. Floor = TF-IDF+lagged.\n")
    print(f"{'encoder':12s} {'floor R2':>9s} {'full R2':>9s} {'dR2':>7s} | "
          f"{'floor rho':>9s} {'full rho':>9s} {'drho':>7s}")

    for enc in encoders:
        penc = E.mean_pooled_filings(enc); ptop = E.topic_filings(enc)
        w = base.merge(penc, on=KEYS, how="inner")
        ecols = [c for c in penc.columns if c.startswith("e") and c[1:].isdigit()]
        tcols = []
        if ptop is not None:
            w = w.merge(ptop, on=KEYS, how="inner")
            tcols = [c for c in ptop.columns if c.startswith("t") and c[1:].isdigit()]
        tr = w[w["split"] == "train"]; ev = w[w["split"] == "val"]
        # floor uses the SAME rows as full (so the comparison is apples-to-apples)
        f_r2, f_rho = fit_eval(tr, ev, ["log_lag"])
        h_r2, h_rho = fit_eval(tr, ev, ["log_lag"], ecols, tcols)
        print(f"{enc:12s} {f_r2:9.3f} {h_r2:9.3f} {h_r2-f_r2:+7.3f} | "
              f"{f_rho:9.3f} {h_rho:9.3f} {h_rho-f_rho:+7.3f}")

    print("\nWin = full beats floor (dR2>0 / drho>0): the encoder+topic add signal orthogonal to TF-IDF.")


if __name__ == "__main__":
    main()
