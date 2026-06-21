"""
Lever 1: does a sharper paragraph->filing pooling beat plain averaging for the encoder?

Clean val-2025. TF-IDF is fit ONCE (identical across pooling methods); only the encoder block changes,
so this runs fast. For each encoder x pool:
  enc-only : RidgeCV on [pooled encoder | log_lag]           -> does the encoder rank firms?
  full     : Ridge on [TF-IDF | log_lag | encoder | topic]   -> does it beat the TF-IDF+lagged floor?
Higher Spearman (IC) / R²_log = better.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval_common as E

KEYS = ["ticker", "fiscal_year"]
POOLS = ["mean", "max", "risk_weighted", "topk_risk"]


def split_col(df):
    return np.where(df.year < 2025, "train", np.where(df.year == 2025, "val", "test"))


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix

    panel = E.load_panel()
    panel["split"] = split_col(panel)
    tr, ev = panel[panel.split == "train"], panel[panel.split == "val"]
    ytr, yev = tr.log_fwd.to_numpy(), ev.log_fwd.to_numpy()

    # TF-IDF fit ONCE — identical across all pooling variants
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                          max_features=20000, sublinear_tf=True)
    Xtr_txt, Xev_txt = vec.fit_transform(tr.text), vec.transform(ev.text)
    lag_tr, lag_ev = csr_matrix(tr[["log_lag"]].to_numpy()), csr_matrix(ev[["log_lag"]].to_numpy())

    f = E.eval_metrics(yev, E.fast_ridge_predict(hstack([Xtr_txt, lag_tr]).tocsr(), ytr,
                                                 hstack([Xev_txt, lag_ev]).tocsr()))
    print(f"FLOOR  TF-IDF+lagged   IC={f['spearman']:.3f}  R2={f['r2_log']:.3f}  (val-2025, n={len(ev)})\n")
    print(f"{'encoder':11s} {'pool':13s} {'enc-only IC':>11s} {'enc R2':>7s} | {'full IC':>7s} {'full R2':>7s}")

    encoders = [e for e in ["sbert", "dual", "three_lora"] if (E.TOPIC_OUT / f"emb_{e}.npy").exists()]
    for enc in encoders:
        topic = E.topic_filings(enc)
        tcols = [c for c in topic.columns if c.startswith("t") and c[1:].isdigit()] if topic is not None else []
        for pool in POOLS:
            penc = E.mean_pooled_filings(enc, pool=pool)
            ecols = [c for c in penc.columns if c.startswith("e") and c[1:].isdigit()]
            w = panel.merge(penc, on=KEYS, how="left")
            if topic is not None:
                w = w.merge(topic, on=KEYS, how="left")
            trw, evw = w[w.split == "train"], w[w.split == "val"]
            dcols = ["log_lag"] + ecols + tcols
            sc = StandardScaler().fit(trw[dcols].to_numpy())
            Dtr, Dev = sc.transform(trw[dcols].to_numpy()), sc.transform(evw[dcols].to_numpy())

            # enc-only (dense, fast RidgeCV) on [encoder | lag]
            enc_idx = [dcols.index(c) for c in ecols] + [dcols.index("log_lag")]
            est = E.make_ridge().fit(Dtr[:, enc_idx], ytr)
            em = E.eval_metrics(yev, est.predict(Dev[:, enc_idx]))

            # full = TF-IDF (reused) | standardized dense block
            full_tr = hstack([Xtr_txt, csr_matrix(Dtr)]).tocsr()
            full_ev = hstack([Xev_txt, csr_matrix(Dev)]).tocsr()
            hm = E.eval_metrics(yev, E.fast_ridge_predict(full_tr, ytr, full_ev))
            print(f"{enc:11s} {pool:13s} {em['spearman']:11.3f} {em['r2_log']:7.3f} | "
                  f"{hm['spearman']:7.3f} {hm['r2_log']:7.3f}")
        print()
    print(f"Beat the floor: full IC > {f['spearman']:.3f} and/or full R2 > {f['r2_log']:.3f}.")


if __name__ == "__main__":
    main()
