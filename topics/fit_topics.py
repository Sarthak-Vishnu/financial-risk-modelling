"""
Phase 4 — Increment 2: fit BERTopic per encoder, score topic quality, emit per-filing exposure vectors.

For each encoder backend (sbert + the domain encoders), this:
  1. encodes every risk-factor paragraph (GPU) and caches the embeddings (.npy);
  2. fits BERTopic (UMAP -> HDBSCAN -> c-TF-IDF) on TRAIN docs only, then transforms val + test
     (leakage-safe: no held-out text shapes the topic space — report §3.1.1);
  3. saves the model + a human-readable topic table (top c-TF-IDF terms) for manual review;
  4. computes topic coherence C_v (gensim) and topic diversity (Dieng 2020) on the train-fit topics;
  5. aggregates per-document topic assignments into a per-FILING topic-exposure vector (fraction of the
     filing's paragraphs in each topic) -> parquet, the Phase 5 / feature-table input.

Encoder embeddings are passed to BERTopic directly (`embedding_model=None`), so the same precomputed
vectors drive both clustering and (later) the dense-embedding ablation conditions.

Outputs (topics/out/ and topics/models/):
  emb_<encoder>.npy                       (cached doc embeddings)
  models/<encoder>/                       (saved BERTopic model)
  topics_<encoder>.csv                    (topic id, size, top-10 terms, representative doc)
  filing_topic_vectors_<encoder>.parquet  (cik, fiscal_year, filing_date, split, t0..t{K-1})
  topic_quality.json                      (appended: encoder -> {n_topics, c_v, diversity})

Usage:
    python topics/fit_topics.py --encoders sbert,dapt,dual,three,three_lora
    python topics/fit_topics.py --encoders three_lora --nr_topics 50 --min_cluster_size 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "topics" / "data" / "topic_docs.jsonl"
OUT_DIR = ROOT / "topics" / "out"
MODEL_DIR = ROOT / "topics" / "models"

ENCODER_PATHS = {
    "sbert": "sentence-transformers/all-mpnet-base-v2",
    "dapt": str(ROOT / "dapt_checkpoints_para" / "best"),
    "dual": str(ROOT / "contrastive_checkpoints" / "dual"),
    "three": str(ROOT / "contrastive_checkpoints" / "three"),
    "three_lora": str(ROOT / "contrastive_checkpoints" / "three_lora"),
}
TOP_N = 10  # words per topic for coherence / diversity / display


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", type=str, default="sbert,dapt,dual,three,three_lora")
    p.add_argument("--docs", type=str, default=str(DOCS))
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--model_dir", type=str, default=str(MODEL_DIR))
    p.add_argument("--min_cluster_size", type=int, default=200,
                   help="HDBSCAN min_cluster_size (larger -> fewer, broader topics)")
    p.add_argument("--nr_topics", type=int, default=50,
                   help="Reduce to this many topics (target the report's 20-80 band); 0 = no reduction")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_docs(path):
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame(rows)
    return df


def get_embeddings(encoder, paths, texts, out_dir, batch_size):
    """Encode once and cache to .npy; reuse on re-runs."""
    cache = Path(out_dir) / f"emb_{encoder}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(texts):
            print(f"  [{encoder}] using cached embeddings {emb.shape}")
            return emb
        print(f"  [{encoder}] cache size mismatch ({emb.shape[0]} != {len(texts)}); re-encoding")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(paths[encoder])
    print(f"  [{encoder}] encoding {len(texts):,} docs...")
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


def build_topic_model(min_cluster_size, seed):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=10)
    return BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer, calculate_probabilities=False, verbose=True)


def coherence_and_diversity(topic_model, train_texts):
    """C_v coherence (gensim) + topic diversity over top-N words, on non-outlier topics."""
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    topic_ids = [t for t in topic_model.get_topics() if t != -1]
    topic_words = [[w for w, _ in topic_model.get_topic(t)[:TOP_N]] for t in topic_ids]
    topic_words = [tw for tw in topic_words if len(tw) >= 2]
    if len(topic_words) < 2:
        return float("nan"), float("nan")

    # tokenize with the SAME analyzer used for c-TF-IDF so bigram topic words align
    analyzer = topic_model.vectorizer_model.build_analyzer()
    tokens = [analyzer(doc) for doc in train_texts]
    dictionary = Dictionary(tokens)
    cm = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary,
                        coherence="c_v", topn=TOP_N)
    c_v = cm.get_coherence()

    flat = [w for tw in topic_words for w in tw]
    diversity = len(set(flat)) / len(flat) if flat else float("nan")
    return float(c_v), float(diversity)


def filing_vectors(df, topics):
    """Per-filing topic exposure = fraction of the filing's (non-outlier) paragraphs in each topic."""
    df = df.copy()
    df["topic"] = topics
    pos = sorted(t for t in set(topics) if t != -1)
    col = {t: i for i, t in enumerate(pos)}
    K = len(pos)

    keys = ["cik", "fiscal_year", "filing_date", "split", "ticker"]
    rows = []
    for key, grp in df.groupby(keys):
        vec = np.zeros(K, dtype=np.float32)
        for t in grp["topic"]:
            if t != -1:
                vec[col[t]] += 1
        s = vec.sum()
        if s > 0:
            vec /= s
        rec = dict(zip(keys, key))
        rec["n_paras"] = len(grp)
        rec["outlier_frac"] = float((grp["topic"] == -1).mean())
        for i in range(K):
            rec[f"t{i}"] = float(vec[i])
        rows.append(rec)
    return pd.DataFrame(rows), pos


def main():
    args = parse_args()
    encoders = [e.strip() for e in args.encoders.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_dir); model_root.mkdir(parents=True, exist_ok=True)

    df = load_docs(args.docs)
    texts = df["text"].tolist()
    is_train = (df["split"] == "train").to_numpy()
    print(f"Docs: {len(df):,} (train {is_train.sum():,}) | encoders: {encoders}")

    quality_path = out_dir / "topic_quality.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}

    for enc in encoders:
        print(f"\n===== {enc} =====", flush=True)
        emb = get_embeddings(enc, ENCODER_PATHS, texts, out_dir, args.batch_size)
        train_texts = [t for t, m in zip(texts, is_train) if m]
        train_emb = emb[is_train]

        topic_model = build_topic_model(args.min_cluster_size, args.seed)
        print(f"  [{enc}] fitting BERTopic on {len(train_texts):,} train docs...", flush=True)
        topic_model.fit(train_texts, embeddings=train_emb)
        if args.nr_topics and args.nr_topics > 0:
            topic_model.reduce_topics(train_texts, nr_topics=args.nr_topics)

        # assign every doc a topic: train via its fitted labels, val/test via transform
        topics = np.empty(len(df), dtype=int)
        topics[is_train] = topic_model.topics_
        if (~is_train).any():
            other_t, _ = topic_model.transform([t for t, m in zip(texts, is_train) if not m],
                                               embeddings=emb[~is_train])
            topics[~is_train] = other_t

        n_topics = len([t for t in topic_model.get_topics() if t != -1])
        c_v, diversity = coherence_and_diversity(topic_model, train_texts)
        print(f"  [{enc}] topics={n_topics} C_v={c_v:.4f} diversity={diversity:.4f}", flush=True)

        # save model + topic table
        mdir = model_root / enc
        topic_model.save(str(mdir), serialization="safetensors",
                         save_ctfidf=True, save_embedding_model=False)
        info = topic_model.get_topic_info()
        info.to_csv(out_dir / f"topics_{enc}.csv", index=False)

        # per-filing exposure vectors
        fv, pos = filing_vectors(df, topics)
        fv.to_parquet(out_dir / f"filing_topic_vectors_{enc}.parquet", index=False)
        print(f"  [{enc}] filing vectors: {fv.shape[0]:,} filings x {len(pos)} topics", flush=True)

        quality[enc] = {"n_topics": n_topics, "c_v": c_v, "diversity": diversity,
                        "min_cluster_size": args.min_cluster_size, "nr_topics": args.nr_topics}
        quality_path.write_text(json.dumps(quality, indent=2))

    print(f"\nDone. Quality -> {quality_path}")
    for enc, q in quality.items():
        print(f"  {enc:12s} topics={q['n_topics']:3d} C_v={q['c_v']:.4f} diversity={q['diversity']:.4f}")


if __name__ == "__main__":
    main()
