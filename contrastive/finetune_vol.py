"""
Lever 3 — supervised fine-tuning of the encoder ON the volatility task.

Every encoder so far was trained for *similarity* (MLM, contrastive), never for *volatility*. That
mismatch is why dense embeddings only match (never beat) TF-IDF for cross-sectional vol ranking. Here we
fine-tune the encoder end-to-end with a **volatility regression objective**: each risk paragraph is labelled
with its filing's log forward-vol, and a linear head on the pooled embedding is trained with MSE. The hope
is the embeddings become *vol-discriminative*, not just semantically smooth.

Strictly forward-looking: train on filings < 2025, monitor on 2025 (filing-level cross-sectional IC).

Outputs (default --out_dir contrastive_checkpoints/ftvol):
  the fine-tuned SentenceTransformer encoder (re-embeddable by the Phase-5 harness)
  head.pt, train_summary.json
  topics/out/emb_ftvol.npy  — per-paragraph embeddings (topic_docs order) for evaluate.py / improve_pooling.py

Usage (GPU):
  python contrastive/finetune_vol.py --base_model dapt_checkpoints_para/best --epochs 3 --fp16
  python contrastive/finetune_vol.py --base_model contrastive_checkpoints/three_lora --lora --fp16
Then evaluate:
  python phase5/improve_pooling.py            # add 'ftvol' to its encoder list, or
  python phase5/run_phase5.py --encoders ftvol
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from sentence_transformers import SentenceTransformer, models

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOPIC_DOCS = ROOT / "topics" / "data" / "topic_docs.jsonl"
FEATURE_TABLE = ROOT / "datasets" / "feature_table.parquet"
DEFAULT_BASE = ROOT / "dapt_checkpoints_para" / "best"
MAX_LEN = 256
EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default=str(DEFAULT_BASE))
    p.add_argument("--out_dir", type=str, default=str(ROOT / "contrastive_checkpoints" / "ftvol"))
    p.add_argument("--emb_out", type=str, default=str(ROOT / "topics" / "out" / "emb_ftvol.npy"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data():
    """Per-paragraph (text, filing log_fwd_vol, year, filing_id) in topic_docs order."""
    ft = pd.read_parquet(FEATURE_TABLE)
    lab = {(r.ticker, int(r.fiscal_year)): np.log(max(r.fwd_vol_30d, EPS))
           for r in ft.itertuples() if pd.notna(r.fwd_vol_30d)}
    docs = pd.read_json(TOPIC_DOCS, lines=True)
    docs["y"] = [lab.get((t, int(fy)), np.nan) for t, fy in zip(docs["ticker"], docs["fiscal_year"])]
    docs["fid"] = docs["ticker"] + "_" + docs["fiscal_year"].astype(str)
    docs["year"] = pd.to_datetime(docs["filing_date"]).dt.year
    return docs


def build_model(base, lora, lora_r, device):
    word = models.Transformer(base, max_seq_length=MAX_LEN)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word, pool], device=device)  # no Normalize: keep magnitude for regression
    peft_handle = None
    if lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1, bias="none",
                         target_modules=["q", "v"])
        peft_handle = get_peft_model(word.auto_model, cfg)
        peft_handle.print_trainable_parameters()
    return model, peft_handle


def embed(model, texts, device, bs=128):
    out = []
    for s in range(0, len(texts), bs):
        feats = model.tokenize(texts[s:s + bs])
        feats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
        out.append(model(feats)["sentence_embedding"])
    return torch.cat(out, 0)


def filing_ic(emb, fids, y_by_fid):
    """Mean-pool paragraph embeddings -> filing, score head later; here return grouping for IC calc."""
    df = pd.DataFrame({"fid": fids})
    groups = df.groupby("fid").indices
    return groups


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | base: {args.base_model} | lora: {args.lora}")

    docs = load_data()
    tr = docs[(docs.year < 2025) & docs.y.notna()].reset_index(drop=True)
    va = docs[(docs.year == 2025) & docs.y.notna()].reset_index(drop=True)
    print(f"Train paragraphs: {len(tr):,} | val paragraphs: {len(va):,} "
          f"({va.fid.nunique()} val filings)")

    ymean, ystd = tr.y.mean(), tr.y.std()
    tr_y = ((tr.y - ymean) / ystd).to_numpy().astype("float32")
    tr_text = tr.text.tolist()

    model, peft_handle = build_model(args.base_model, args.lora, args.lora_r, device)
    head = nn.Linear(model.get_sentence_embedding_dimension(), 1).to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if (args.fp16 and device == "cuda") else None

    # val filing-level targets + grouping (paragraph order fixed)
    va_groups = va.groupby("fid").indices
    va_y = np.array([va.loc[ix[0], "y"] for ix in va_groups.values()])

    from scipy.stats import spearmanr
    rng = np.random.default_rng(args.seed)
    best_ic, best_state = -1.0, None
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        order = rng.permutation(len(tr_text))
        tot, n = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            texts = [tr_text[i] for i in idx]
            yb = torch.tensor(tr_y[idx], device=device)
            feats = model.tokenize(texts)
            feats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
            ctx = torch.amp.autocast("cuda") if scaler else nullcontext()
            with ctx:
                emb = model(feats)["sentence_embedding"]
                pred = head(emb).squeeze(-1)
                loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item() * len(idx); n += len(idx)

        # val: filing-level cross-sectional IC
        model.eval(); head.eval()
        with torch.no_grad():
            emb_va = embed(model, va.text.tolist(), device)
            fil_pred = []
            for ix in va_groups.values():
                v = emb_va[list(ix)].mean(0, keepdim=True)
                fil_pred.append(head(v).item())
        ic = spearmanr(va_y, np.array(fil_pred)).statistic
        print(f"epoch {epoch} | train mse {tot/max(n,1):.4f} | val filing IC {ic:.4f}", flush=True)
        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(head.state_dict(), out_dir / "head.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    if peft_handle is not None:
        peft_handle.merge_and_unload()
    model.save(str(out_dir))
    json.dump({"base": args.base_model, "best_val_filing_ic": float(best_ic),
               "epochs": args.epochs, "lora": args.lora, "y_mean": float(ymean), "y_std": float(ystd)},
              open(out_dir / "train_summary.json", "w"), indent=2)

    # emit per-paragraph embeddings (topic_docs order) for the Phase-5 harness
    print("Embedding all paragraphs for evaluation...", flush=True)
    model.eval()
    with torch.no_grad():
        allemb = embed(model, docs.text.tolist(), device).cpu().numpy().astype("float32")
    np.save(args.emb_out, allemb)
    print(f"\nBest val filing IC {best_ic:.4f} -> saved {out_dir}\n  emb -> {args.emb_out} {allemb.shape}")
    print("Now: python phase5/run_phase5.py --encoders ftvol   (compare IC vs TF-IDF floor 0.545)")


if __name__ == "__main__":
    main()
