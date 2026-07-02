"""
P0-b / E4 — encode-only pass over the paragraph corpus, with FULL-TEXT windowing.

Why this exists (audit 2026-07-02): the Phase-4 encode truncated every paragraph at the model's
max sequence length while 36.6% of paragraphs are longer — ~65% of corpus words never reached any
encoder, but TF-IDF ingests all of them. Every encoder-vs-TF-IDF comparison so far was therefore
full-text bag-of-words vs one-third-text encoders. Here long paragraphs are split into overlapping
token windows, each window is encoded, and windows are mean-pooled back to one L2-normalized vector
per paragraph (same row contract as fit_topics.get_embeddings).

Also serves E4: regenerates embeddings for `ftvol` (checkpoint exists, embeddings never written)
and `bge` (BAAI/bge-base-en-v1.5 — run `hf download BAAI/bge-base-en-v1.5` on an internet node
first; the compute node is offline).

Default input is the P0-a corrected corpus (topic_docs_fixed.jsonl) and output carries the matching
`_fixed` suffix that phase5/eval_common.py expects: topics/out/emb_<enc>_fixed.npy.

Run (GPU):  python topics/encode_paragraphs.py --encoders dual,sbert,volaware,ftvol,bge
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from topics.fit_topics import ENCODER_PATHS  # noqa: E402

DOCS_FIXED = ROOT / "topics" / "data" / "topic_docs_fixed.jsonl"
OUT_DIR = ROOT / "topics" / "out"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", type=str, default="dual,sbert,volaware,ftvol,bge")
    p.add_argument("--docs", type=str, default=str(DOCS_FIXED))
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--suffix", type=str, default="_fixed",
                   help="cache suffix; must match eval_common.CACHE_SUFFIX for the docs used")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=256, help="window length (wordpieces)")
    p.add_argument("--stride", type=int, default=192, help="window step; overlap = max_tokens-stride")
    p.add_argument("--no_windowed", action="store_true", help="plain truncating encode (legacy behaviour)")
    return p.parse_args()


def windows_of(ids: list[int], max_tokens: int, stride: int) -> list[list[int]]:
    if len(ids) <= max_tokens:
        return [ids]
    out = []
    for start in range(0, len(ids), stride):
        w = ids[start:start + max_tokens]
        if len(w) < max_tokens // 4 and out:  # tail sliver already mostly covered by overlap
            break
        out.append(w)
        if start + max_tokens >= len(ids):
            break
    return out


def encode_one(enc: str, texts: list[str], args) -> np.ndarray | None:
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(ENCODER_PATHS[enc])
    except Exception as e:  # e.g. bge not downloaded on the offline node
        print(f"  [{enc}] SKIP — could not load model: {e}")
        return None
    model.max_seq_length = args.max_tokens

    if args.no_windowed:
        return model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)

    tok = model.tokenizer
    print(f"  [{enc}] tokenizing {len(texts):,} paragraphs...")
    all_ids = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]

    win_texts, owner = [], []
    for i, ids in enumerate(all_ids):
        for w in windows_of(ids, args.max_tokens, args.stride):
            win_texts.append(tok.decode(w, skip_special_tokens=True))
            owner.append(i)
    owner = np.asarray(owner)
    n_long = int((np.bincount(owner, minlength=len(texts)) > 1).sum())
    print(f"  [{enc}] {len(win_texts):,} windows for {len(texts):,} paragraphs "
          f"({n_long:,} paragraphs windowed)")

    wemb = model.encode(win_texts, batch_size=args.batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    emb = np.zeros((len(texts), wemb.shape[1]), dtype=np.float64)
    np.add.at(emb, owner, wemb)
    emb /= np.bincount(owner, minlength=len(texts))[:, None]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb.astype(np.float32)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = [json.loads(l)["text"] for l in open(args.docs)]
    print(f"Docs: {len(texts):,} from {args.docs}")

    for enc in args.encoders.split(","):
        enc = enc.strip()
        if enc not in ENCODER_PATHS:
            print(f"  [{enc}] SKIP — unknown encoder (see fit_topics.ENCODER_PATHS)")
            continue
        cache = out_dir / f"emb_{enc}{args.suffix}.npy"
        if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(texts):
            print(f"  [{enc}] cache OK ({cache.name}) — skipping")
            continue
        try:
            emb = encode_one(enc, texts, args)
        except Exception as e:  # one encoder's failure must not abort the remaining ones
            print(f"  [{enc}] FAILED mid-encode ({type(e).__name__}: {e}) — continuing with next encoder")
            continue
        if emb is None:
            continue
        np.save(cache, emb)
        print(f"  [{enc}] saved {emb.shape} -> {cache}")

    print("Done.")


if __name__ == "__main__":
    main()
