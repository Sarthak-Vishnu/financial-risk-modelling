"""
E2 — disclosure-change features (Lazy Prices: Cohen, Malloy & Nguyen, JF 2020).

For every filing that has the same firm's previous filing (fiscal-year gap <= 2), compute
year-over-year change measures of the Item-1A risk section, decomposed into:

  lexical   chg_lex_cos       cosine of per-pair sublinear term-frequency vectors (per-pair
                              vocabulary -> trivially leakage-free; IDF is degenerate on 2 docs)
            chg_jaccard       token-set Jaccard similarity
  semantic  chg_enc_cos_<e>   cosine of the filings' mean-pooled encoder vectors, for every
                              encoder with a topics/out/emb_<e><suffix>.npy cache
  thematic  chg_topic_jsd     Jensen-Shannon divergence of normalized topic-exposure vectors
  shape     chg_len_ratio     log paragraph-count ratio (current / previous)
            chg_new_para_frac fraction of current paragraphs with max cosine < 0.9 vs all
                              previous-filing paragraphs (paragraph-level sbert embeddings)

All features are backward-looking -> leakage-free -> valid in the expanding-window backtest.
Novelty angle: if semantic change adds where lexical change does not, dense encoders carry a
signal bag-of-words cannot express.

Output: datasets/change_features<suffix>.parquet keyed (ticker, fiscal_year), where <suffix>
follows eval_common.CACHE_SUFFIX (P0-a corrected data by default). Loaded downstream via
eval_common.change_matrix().

Run:  python dataset_config/build_change_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase5"))
import eval_common as E  # noqa: E402

MAX_GAP = 2          # pair with previous filing at most this many fiscal years back
NEW_PARA_TH = 0.9    # a current paragraph is "new" if its best match in prev filing is below this


def sublinear_tf_cosine(a: str, b: str):
    """Per-pair vocabulary sublinear-TF cosine + token-set Jaccard."""
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        X = CountVectorizer().fit_transform([a, b]).toarray().astype(float)
    except ValueError:      # empty vocabulary
        return np.nan, np.nan
    tf = np.where(X > 0, 1.0 + np.log(X, where=X > 0, out=np.zeros_like(X)), 0.0)
    na, nb = np.linalg.norm(tf[0]), np.linalg.norm(tf[1])
    cos = float(tf[0] @ tf[1] / (na * nb)) if na > 0 and nb > 0 else np.nan
    sa, sb = X[0] > 0, X[1] > 0
    jac = float((sa & sb).sum() / max((sa | sb).sum(), 1))
    return cos, jac


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 0, None); q = np.clip(q, 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log(x[mask] / y[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    print(f"USE_FIXED={E.USE_FIXED}  suffix='{E.CACHE_SUFFIX}'")
    docs = E._docs_df()  # one row per paragraph: ticker, fiscal_year, text
    filing_text = (docs.groupby(["ticker", "fiscal_year"])
                       .agg(text=("text", " ".join), n_paras=("text", "size")).reset_index())
    filing_text = filing_text.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    print(f"filings with text: {len(filing_text):,}")

    # consecutive-filing pairs per ticker (gap <= MAX_GAP)
    ft = filing_text
    ft["prev_fy"] = ft.groupby("ticker")["fiscal_year"].shift(1)
    ft["prev_ix"] = ft.groupby("ticker").cumcount() - 1
    pairs = ft[ft.prev_fy.notna() & ((ft.fiscal_year - ft.prev_fy) <= MAX_GAP)].copy()
    # absolute index of the previous filing row within filing_text
    grp_start = ft.index.to_series().groupby(ft.ticker).transform("min")
    pairs["prev_abs"] = (grp_start.loc[pairs.index] + pairs["prev_ix"]).astype(int)
    print(f"pairs (gap<= {MAX_GAP}): {len(pairs):,}")

    # ---- per-filing pooled encoder vectors (semantic change) --------------------------------
    enc_pooled = {}
    for enc in ["dual", "sbert", "volaware", "three_lora", "ftvol", "bge"]:
        pe = E.mean_pooled_filings(enc)
        if pe is None:
            continue
        ecols = [c for c in pe.columns if c.startswith("e") and c[1:].isdigit()]
        M = pe[ecols].to_numpy()
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        enc_pooled[enc] = dict(zip(zip(pe.ticker, pe.fiscal_year), M))
        print(f"  semantic change from enc[{enc}]")

    # ---- per-filing topic exposure (thematic change) ----------------------------------------
    topic_map = None
    for enc in ["dual", "sbert"]:
        tf_ = E.topic_filings(enc)
        if tf_ is not None:
            tcols = [c for c in tf_.columns if c.startswith("t") and c[1:].isdigit()]
            topic_map = dict(zip(zip(tf_.ticker, tf_.fiscal_year), tf_[tcols].to_numpy()))
            print(f"  thematic change from topic[{enc}]")
            break

    # ---- paragraph-level embeddings for chg_new_para_frac -----------------------------------
    para_groups, para_emb = None, None
    cache = E.TOPIC_OUT / f"emb_sbert{E.CACHE_SUFFIX}.npy"
    if cache.exists():
        emb = np.load(cache, mmap_mode="r")
        if emb.shape[0] == len(docs):
            para_emb = emb
            para_groups = docs.groupby(["ticker", "fiscal_year"]).indices
            print("  chg_new_para_frac from paragraph sbert embeddings")
    if para_emb is None:
        print("  [warn] no aligned paragraph embeddings -> chg_new_para_frac will be NaN")

    rows = []
    for _, r in pairs.iterrows():
        cur_key = (r.ticker, int(r.fiscal_year))
        prev = filing_text.iloc[int(r.prev_abs)]
        prev_key = (prev.ticker, int(prev.fiscal_year))
        assert prev_key[0] == r.ticker and prev_key[1] == int(r.prev_fy)

        lex, jac = sublinear_tf_cosine(r.text, prev.text)
        rec = {"ticker": r.ticker, "fiscal_year": int(r.fiscal_year),
               "chg_lex_cos": lex, "chg_jaccard": jac,
               "chg_len_ratio": float(np.log(r.n_paras / max(prev.n_paras, 1)))}

        for enc, m in enc_pooled.items():
            va, vb = m.get(cur_key), m.get(prev_key)
            rec[f"chg_enc_cos_{enc}"] = float(va @ vb) if va is not None and vb is not None else np.nan

        if topic_map is not None:
            ta, tb = topic_map.get(cur_key), topic_map.get(prev_key)
            rec["chg_topic_jsd"] = js_divergence(ta, tb) if ta is not None and tb is not None else np.nan

        if para_emb is not None:
            ia = para_groups.get(cur_key); ib = para_groups.get(prev_key)
            if ia is not None and ib is not None and len(ia) and len(ib):
                S = np.asarray(para_emb[np.asarray(ia)]) @ np.asarray(para_emb[np.asarray(ib)]).T
                rec["chg_new_para_frac"] = float((S.max(axis=1) < NEW_PARA_TH).mean())
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = E.CHANGE_FEATURES
    out.to_parquet(out_path, index=False)

    print(f"\nSaved {len(out):,} rows -> {out_path}")
    print("\nSanity (Lazy-Prices stylized fact: YoY similarity is HIGH, median ~0.7-0.9):")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(out[[c for c in out.columns if c.startswith("chg_")]]
              .describe().loc[["count", "mean", "50%", "min", "max"]].T.to_string())
    med = out["chg_lex_cos"].median()
    if not (0.55 <= med <= 0.97):
        print(f"[GATE] chg_lex_cos median {med:.3f} outside the expected 0.55-0.97 band — inspect before use.")
    else:
        print(f"[gate ok] chg_lex_cos median {med:.3f}")


if __name__ == "__main__":
    main()
