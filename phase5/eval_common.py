"""
Phase 5 — shared evaluation machinery for the performance-&-validation upgrade.

Why this module exists: the single-window val-2025 eval (n≈397) is too noisy to validate the thesis.
Here we provide an **expanding-window backtest** over 2018–2024 (~3,100 test filings), **incremental-value**
framing (ΔR² of text over a lagged-only / controls model — the actual claim), **non-text controls**, stronger
heads, and **statistical significance** (bootstrap CIs + a paired Diebold–Mariano test). Reused by
`phase5/evaluate.py`. Leakage-safe: every vectorizer / scaler / PCA / estimator is fit on the train years
only, inside each fold.

Target = log(fwd_vol_30d). Persistence (log lagged vol) explains ~34% of its variance, so the interesting
quantity is how much of the remaining ~66% residual the text representations recover — reported as ΔR².
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os

# P0-a (fix_filing_join.py): the corrected filing<->text join re-keys fiscal_year and re-assigns
# pickles by FILING year. When the corrected artifacts exist they are the default; every cache
# aligned to them carries a "_fixed" suffix (embeddings/topic vectors built against the old join
# are silently misaligned, so they must never be mixed). Set RISK_LEGACY_DATA=1 to reproduce
# pre-fix numbers.
_FT_FIXED = ROOT / "datasets" / "feature_table_fixed.parquet"
_TD_FIXED = ROOT / "topics" / "data" / "topic_docs_fixed.jsonl"
USE_FIXED = (_FT_FIXED.exists() and _TD_FIXED.exists()
             and os.environ.get("RISK_LEGACY_DATA") != "1")
CACHE_SUFFIX = "_fixed" if USE_FIXED else ""

FEATURE_TABLE = _FT_FIXED if USE_FIXED else ROOT / "datasets" / "feature_table.parquet"
MARKET_FEATURES = ROOT / "datasets" / "market_features.parquet"
FUNDAMENTAL_FEATURES = ROOT / "datasets" / "fundamental_features.parquet"
CALL_FEATURES = ROOT / "datasets" / "call_features.parquet"
TOPIC_DOCS = _TD_FIXED if USE_FIXED else ROOT / "topics" / "data" / "topic_docs.jsonl"
TOPIC_OUT = ROOT / "topics" / "out"

EPS = 1e-6
ALPHAS = np.logspace(-2, 3, 20)
TEST_YEARS = list(range(int(os.environ.get("RISK_TEST_START", "2018")), 2025))
# expanding-window test years (default 2018..2024; RISK_TEST_START=2013 -> the 12-year window)
HORIZON = int(os.environ.get("RISK_HORIZON", "30"))
# label horizon in trading days (default 30 = the study's frozen labels; 20/60/90 swap in the
# horizon labels from compute_horizon_labels.py — the 30d columns are never modified)
HORIZON_LABELS = ROOT / "datasets" / "volatility_labels_horizons.parquet"
MIN_TRAIN_YEAR = 2007                  # earliest labelled year with full coverage


# ----------------------------------------------------------------------------- data
def load_panel():
    """One labelled row per filing: target + lagged + controls + risk text, keyed (ticker, fiscal_year)."""
    ft = pd.read_parquet(FEATURE_TABLE)
    ft["filing_date"] = pd.to_datetime(ft["filing_date"])
    ft["year"] = ft["filing_date"].dt.year
    ft["sic2"] = ft["sic"].astype(str).str[:2]
    ft = ft[["ticker", "cik", "fiscal_year", "year", "filing_date", "sic2",
             "lagged_vol_30d", "fwd_vol_30d"]].copy()

    docs = _docs_df()   # cached parse (shared with mean_pooled_filings; avoids re-reading the 193k jsonl)
    text = (docs.groupby(["ticker", "fiscal_year"])
                .agg(text=("text", lambda s: " ".join(s)), n_paras=("text", "size"))
                .reset_index())

    df = ft.merge(text, on=["ticker", "fiscal_year"], how="inner")
    df = df[df["fwd_vol_30d"].notna() & df["lagged_vol_30d"].notna()].copy()
    df["log_fwd"] = np.log(df["fwd_vol_30d"].clip(lower=EPS))
    df["log_lag"] = np.log(df["lagged_vol_30d"].clip(lower=EPS))
    if HORIZON != 30:
        # horizon panel = the 30d panel re-labelled at H (subset where the H-day windows exist),
        # so cross-horizon comparisons run on near-identical universes
        if not HORIZON_LABELS.exists():
            raise SystemExit(f"{HORIZON_LABELS} not found — run compute_horizon_labels.py first.")
        hl = pd.read_parquet(HORIZON_LABELS)
        hl["filing_date"] = pd.to_datetime(hl["filing_date"])
        fc, lc = f"fwd_vol_{HORIZON}d", f"lagged_vol_{HORIZON}d"
        df = df.merge(hl[["ticker", "filing_date", fc, lc]], on=["ticker", "filing_date"],
                      how="left")
        df = df[df[fc].notna() & df[lc].notna()].copy()
        df["log_fwd"] = np.log(df[fc].clip(lower=EPS))
        df["log_lag"] = np.log(df[lc].clip(lower=EPS))
        print(f"[horizon {HORIZON}d] panel: {len(df):,} filings")
    return df.reset_index(drop=True)


# --- paragraph-doc cache (parsed once; repeated re-parsing of the 193k-line jsonl was the slow path) ---
_DOCS = None
_RISK = None
# small risk/uncertainty lexicon (Loughran–McDonald-flavoured) to score how "risky" a paragraph reads
RISK_TERMS = set((
    "risk risks risky uncertain uncertainty uncertainties adverse adversely litigation litigations "
    "lawsuit lawsuits loss losses liability liabilities default defaults impairment impairments "
    "decline declines decrease decreased disruption disruptions fail failed failure failures breach "
    "breaches penalty penalties downturn recession volatility volatile competition competitive "
    "regulatory regulation regulations negative negatively material adverse harm harmed damages "
    "delinquency bankruptcy insolvency restructuring writedown shortfall deterioration").split())


def _docs_df():
    """Parsed paragraph docs (ticker, fiscal_year, text), cached. Fast — no risk scoring."""
    global _DOCS
    if _DOCS is None:
        _DOCS = pd.read_json(TOPIC_DOCS, lines=True)[["ticker", "fiscal_year", "text"]]
    return _DOCS


def _risk_scores():
    """Per-paragraph risk-term density, cached. Vectorized via CountVectorizer (fast); lazy."""
    global _RISK
    if _RISK is None:
        from sklearn.feature_extraction.text import CountVectorizer
        df = _docs_df()
        cv = CountVectorizer(vocabulary=sorted(RISK_TERMS))
        counts = np.asarray(cv.transform(df["text"]).sum(axis=1)).ravel()
        lengths = df["text"].str.count(r"\S+").to_numpy() + 1.0
        _RISK = counts / lengths
    return _RISK


def mean_pooled_filings(enc, pool="mean", topk=5):
    """Pool emb_<enc>.npy paragraph rows to one vector per (ticker, fiscal_year). None if no cache.

    pool: 'mean' (default), 'max', 'risk_weighted' (weight paragraphs by risk-term density), or
    'topk_risk' (mean of the top-k riskiest paragraphs). Averaging blurs the signal when only a few
    paragraphs drive volatility — the alternatives test whether a sharper aggregation recovers it."""
    cache = TOPIC_OUT / f"emb_{enc}{CACHE_SUFFIX}.npy"
    if not cache.exists():
        return None
    emb = np.load(cache)
    docs = _docs_df()
    if len(docs) != emb.shape[0]:
        print(f"[warn] emb_{enc}{CACHE_SUFFIX}.npy rows {emb.shape[0]} != docs {len(docs)} "
              f"(stale cache — re-encode) -> skipping encoder '{enc}'")
        return None
    risk = _risk_scores() if pool in ("risk_weighted", "topk_risk") else None
    groups = docs.groupby(["ticker", "fiscal_year"]).indices
    rows, vecs = [], []
    for k, idx in groups.items():
        idx = np.asarray(idx)
        Eg = emb[idx]
        if pool == "mean":
            v = Eg.mean(axis=0)
        elif pool == "max":
            v = Eg.max(axis=0)
        elif pool == "risk_weighted":
            w = risk[idx] + 1e-3
            v = (Eg * w[:, None]).sum(axis=0) / w.sum()
        elif pool == "topk_risk":
            sel = idx[np.argsort(risk[idx])[-min(topk, len(idx)):]] if len(idx) > topk else idx
            v = emb[sel].mean(axis=0)
        else:
            raise ValueError(f"unknown pool '{pool}'")
        rows.append(k); vecs.append(v)
    out = pd.DataFrame(vecs, columns=[f"e{i}" for i in range(emb.shape[1])])
    out[["ticker", "fiscal_year"]] = pd.DataFrame(rows, columns=["ticker", "fiscal_year"])
    return out


def topic_filings(enc):
    """Per-filing topic-exposure (t0..t{K-1}) + outlier_frac, keyed (ticker, fiscal_year). None if absent."""
    pq = TOPIC_OUT / f"filing_topic_vectors_{enc}{CACHE_SUFFIX}.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    tcols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]
    keep = ["ticker", "fiscal_year"] + tcols + (["outlier_frac"] if "outlier_frac" in df.columns else [])
    return df[keep].copy()


def structured_filings():
    """Per-filing structured features keyed (ticker, filing_date): market (Stage A) + Compustat
    fundamentals, outer-joined on the shared key. None until at least one of the two is built."""
    frames = []
    for path in (MARKET_FEATURES, FUNDAMENTAL_FEATURES):
        if path.exists():
            d = pd.read_parquet(path)
            d["filing_date"] = pd.to_datetime(d["filing_date"])
            frames.append(d)
    if not frames:
        return None
    df = frames[0]
    for d in frames[1:]:
        df = df.merge(d, on=["ticker", "filing_date"], how="outer")
    return df


def structured_matrix(df, feats=None):
    """Merge market features onto panel rows by (ticker, filing_date). Returns (ndarray, colnames).
    NaNs are preserved (HGB handles them natively; the imputed-ridge head fills them per fold).
    `df` must carry ticker + filing_date (load_panel does). Returns (None, []) if Stage A not built."""
    mf = structured_filings()
    if mf is None:
        return None, []
    cols = feats or [c for c in mf.columns if c not in ("ticker", "filing_date")]
    key = df[["ticker", "filing_date"]].copy()
    key["filing_date"] = pd.to_datetime(key["filing_date"])
    merged = key.merge(mf[["ticker", "filing_date"] + cols], on=["ticker", "filing_date"], how="left")
    return merged[cols].to_numpy(dtype=float), cols


def call_matrix(df):
    """Merge earnings-call tone features onto panel rows by (ticker, filing_date). (None, []) if not built.
    Calls cover 2025-2026 only -> use this on the val-2025 split (pilot), not the full backtest."""
    if not CALL_FEATURES.exists():
        return None, []
    mf = pd.read_parquet(CALL_FEATURES)
    mf["filing_date"] = pd.to_datetime(mf["filing_date"])
    cols = [c for c in mf.columns if c.startswith("call_")]
    key = df[["ticker", "filing_date"]].copy()
    key["filing_date"] = pd.to_datetime(key["filing_date"])
    merged = key.merge(mf, on=["ticker", "filing_date"], how="left")
    return merged[cols].to_numpy(dtype=float), cols


FORM4_FEATURES = ROOT / "datasets" / "form4_features.parquet"


def form4_matrix(df):
    """Merge Form 4 insider features onto panel rows by (ticker, filing_date). (None, []) if not
    built. Trailing pre-filing windows (build_form4_features.py) -> leakage-free, full 2006+
    coverage -> valid in both backtest windows."""
    if not FORM4_FEATURES.exists():
        return None, []
    mf = pd.read_parquet(FORM4_FEATURES)
    mf["filing_date"] = pd.to_datetime(mf["filing_date"])
    cols = [c for c in mf.columns if c.startswith("f4_")]
    key = df[["ticker", "filing_date"]].copy()
    key["filing_date"] = pd.to_datetime(key["filing_date"])
    merged = key.merge(mf, on=["ticker", "filing_date"], how="left")
    return merged[cols].to_numpy(dtype=float), cols


CHANGE_FEATURES = ROOT / "datasets" / f"change_features{CACHE_SUFFIX}.parquet"


LABEL_TRAINED_ENCODERS = ("volaware", "ftvol")  # trained on pre-2025 vol labels (regression / decile pairing)


def change_matrix(df, exclude_label_trained=False):
    """Merge disclosure-change features (build_change_features.py) onto panel rows by
    (ticker, fiscal_year). Backward-looking (this filing vs the firm's previous one) ->
    leakage-free, valid in the backtest. Returns (None, []) if not built.

    exclude_label_trained: drop chg_enc_cos_* columns computed from encoders that were trained
    with pre-2025 volatility labels — in a pre-2025 backtest those similarities carry test-year
    label information through the encoder, so backtest lanes must not see them. (ftvol2018-style
    era-cutoff encoders keep their suffix out of LABEL_TRAINED_ENCODERS and remain admissible.)"""
    if not CHANGE_FEATURES.exists():
        return None, []
    cf = pd.read_parquet(CHANGE_FEATURES)
    cols = [c for c in cf.columns if c.startswith("chg_")]
    if exclude_label_trained:
        bad = tuple(f"chg_enc_cos_{e}" for e in LABEL_TRAINED_ENCODERS)
        cols = [c for c in cols if not c.startswith(bad)]
    merged = df[["ticker", "fiscal_year"]].merge(cf[["ticker", "fiscal_year"] + cols],
                                                 on=["ticker", "fiscal_year"], how="left")
    return merged[cols].to_numpy(dtype=float), cols


def make_imputed_ridge():
    """Ridge for dense numeric blocks that may contain NaN: median-impute (fit on train) -> scale -> RidgeCV."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))


def controls_matrix(df):
    """Non-text controls: SIC-2 sector one-hots + log document length. Returns (ndarray, colnames)."""
    sect = pd.get_dummies(df["sic2"], prefix="sic", dtype=float)
    nlen = np.log1p(df["n_paras"].to_numpy()).reshape(-1, 1)
    X = np.hstack([sect.to_numpy(), nlen])
    return X, list(sect.columns) + ["log_n_paras"]


# ----------------------------------------------------------------------------- heads
def make_ridge():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def fast_ridge_predict(Xtr, ytr, Xev, alpha=10.0):
    """Ridge that stays sparse (solver=sparse_cg) — avoids RidgeCV's GCV densification of big TF-IDF
    matrices. Fixed alpha, so absolute numbers shift slightly vs RidgeCV, but relative comparisons
    across variants (held constant) are valid and the run is ~10x faster."""
    from sklearn.linear_model import Ridge
    from scipy.sparse import issparse
    solver = "lsqr" if issparse(Xtr) else "auto"   # lsqr converges fast & robustly on TF-IDF
    return Ridge(alpha=alpha, solver=solver, max_iter=500).fit(Xtr, ytr).predict(Xev)


def make_pca_ridge(k=100):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=42),
                         RidgeCV(alphas=ALPHAS))


def make_hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                         max_iter=400, l2_regularization=1.0,
                                         early_stopping=True, random_state=42)


class TextNumericRidge:
    """TF-IDF over risk text, optionally hstacked with median-imputed + standardized numeric cols,
    then a fixed-alpha sparse Ridge (lsqr). Fits vectorizer/imputer/scaler on the training fold
    (leakage-safe). Uses Ridge not RidgeCV: RidgeCV's GCV densifies the big sparse TF-IDF and stalls."""

    def __init__(self, numeric_cols=None, max_features=20000, min_df=5, alpha=10.0):
        self.numeric_cols = numeric_cols or []
        self.max_features = max_features
        self.min_df = min_df
        self.alpha = alpha

    def _numeric(self, X, fit):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import csr_matrix
        A = X[self.numeric_cols].to_numpy(dtype=float)
        if fit:
            self.imp_ = SimpleImputer(strategy="median").fit(A)
            self.sc_ = StandardScaler().fit(self.imp_.transform(A))
        return csr_matrix(self.sc_.transform(self.imp_.transform(A)))

    def fit(self, X, y):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from scipy.sparse import hstack
        self.vec_ = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                    min_df=self.min_df, max_features=self.max_features,
                                    sublinear_tf=True)
        Xt = self.vec_.fit_transform(X["text"])
        if self.numeric_cols:
            Xt = hstack([Xt, self._numeric(X, fit=True)]).tocsr()
        # fixed-alpha sparse Ridge (lsqr) — RidgeCV's GCV densifies the TF-IDF block and stalls
        self.model_ = Ridge(alpha=self.alpha, solver="lsqr", max_iter=1000).fit(Xt, y)
        return self

    def predict(self, X):
        from scipy.sparse import hstack
        Xt = self.vec_.transform(X["text"])
        if self.numeric_cols:
            Xt = hstack([Xt, self._numeric(X, fit=False)]).tocsr()
        return self.model_.predict(Xt)


# ----------------------------------------------------------------------------- backtest
def rolling_predict(X, y, years, make_estimator, test_years=TEST_YEARS, min_train_year=MIN_TRAIN_YEAR):
    """Expanding-window out-of-sample predictions. X is an ndarray or a DataFrame (boolean-mask sliceable).
    For each test year Y: fit a fresh estimator on filings in [min_train_year, Y), predict Y. Returns a
    full-length pred array (NaN where never tested)."""
    y = np.asarray(y, dtype=float)
    pred = np.full(len(y), np.nan)
    years = np.asarray(years)
    for Y in test_years:
        tr = (years >= min_train_year) & (years < Y)
        te = (years == Y)
        if tr.sum() < 100 or te.sum() == 0:
            continue
        est = make_estimator()
        Xtr = X[tr] if not isinstance(X, pd.DataFrame) else X.loc[tr]
        Xte = X[te] if not isinstance(X, pd.DataFrame) else X.loc[te]
        est.fit(Xtr, y[tr])
        pred[te] = est.predict(Xte)
    return pred


# ----------------------------------------------------------------------------- metrics + significance
def eval_metrics(y_true, y_pred, lag=None):
    from scipy.stats import spearmanr
    m = np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    err = yp - yt
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    out = {"n": int(m.sum()),
           "rmse_log": float(np.sqrt(np.mean(err ** 2))),
           "r2_log": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
           "spearman": float(spearmanr(yt, yp).statistic)}
    if lag is not None:  # directional accuracy: did vol rise/fall vs lagged, predicted correctly?
        lg = np.asarray(lag)[m]
        out["dir_acc"] = float(np.mean(np.sign(yt - lg) == np.sign(yp - lg)))
    return out


def bootstrap_ci(y_true, y_pred, fn, n=1000, seed=0):
    """Percentile CI for a scalar metric fn(y_true,y_pred) over resampled test filings."""
    m = np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        stats.append(fn(yt[idx], yp[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_dm_test(y_true, pred_a, pred_b):
    """Paired Diebold–Mariano-style test on squared-error loss (a vs b). Negative mean_diff => a better.
    Returns (mean_loss_diff, two-sided p). p<0.05 => the two models' accuracy differs significantly."""
    from scipy.stats import norm
    m = np.isfinite(pred_a) & np.isfinite(pred_b)
    d = (pred_a[m] - y_true[m]) ** 2 - (pred_b[m] - y_true[m]) ** 2
    mean = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(len(d)))
    if se == 0:
        return mean, float("nan")
    p = 2 * (1 - norm.cdf(abs(mean / se)))
    return mean, float(p)


def r2_only(yt, yp):
    ss_res = np.sum((yp - yt) ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# --- cross-sectional (within-year) evaluation ---------------------------------
# Out-of-time *level* prediction is dominated by macro vol regime shifts (e.g. 2020). The markable
# question is cross-sectional: WITHIN a year, does the representation rank which firms are more volatile?
# So we compute metrics within each test year, then aggregate across years (mean IC = mean within-year
# Spearman, the standard finance signal metric; mean within-year R² = "CS-R²").
def per_year_metrics(y, pred, years, test_years, lag=None):
    from scipy.stats import spearmanr
    rows = {}
    for Y in test_years:
        m = (years == Y) & np.isfinite(pred)
        if m.sum() < 20:
            continue
        yt, yp = y[m], pred[m]
        d = {"r2": r2_only(yt, yp), "spear": float(spearmanr(yt, yp).statistic), "n": int(m.sum())}
        if lag is not None:
            lg = np.asarray(lag)[m]
            d["dir"] = float(np.mean(np.sign(yt - lg) == np.sign(yp - lg)))
        rows[int(Y)] = d
    return rows


def aggregate_cs(per_year):
    r2 = [v["r2"] for v in per_year.values()]
    sp = [v["spear"] for v in per_year.values()]
    da = [v["dir"] for v in per_year.values() if "dir" in v]
    out = {"cs_r2": float(np.mean(r2)) if r2 else float("nan"),
           "mean_ic": float(np.mean(sp)) if sp else float("nan"),
           "dir_acc": float(np.mean(da)) if da else float("nan")}
    if len(sp) >= 2:  # t-stat of the IC series across years (is the signal reliably non-zero?)
        out["ic_t"] = float(np.mean(sp) / (np.std(sp, ddof=1) / np.sqrt(len(sp)) + 1e-12))
    else:
        out["ic_t"] = float("nan")
    return out


def paired_year_test(per_cond, per_base, key="r2"):
    """Paired test across years: is the per-year metric of cond > base? Returns (mean_diff, two-sided p)."""
    from scipy.stats import ttest_1samp
    ys = sorted(set(per_cond) & set(per_base))
    diff = [per_cond[y][key] - per_base[y][key] for y in ys]
    if len(diff) < 2:
        return (float(diff[0]) if diff else float("nan")), float("nan")
    t = ttest_1samp(diff, 0.0)
    return float(np.mean(diff)), float(t.pvalue)
