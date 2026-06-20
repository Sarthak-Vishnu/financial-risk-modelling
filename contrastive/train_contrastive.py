"""
Phase 3 — Increment 2: three-view contrastive fine-tuning.

Fine-tunes the post-DAPT encoder (`dapt_checkpoints_para/best`) with a SimCSE-style
in-batch-negatives objective over the pairs built in Increment 1. Implemented as a custom
PyTorch loop (not the stock SentenceTransformer Trainer) because the loss needs per-example
(sic2, fiscal_year, view) metadata for:

  - Label-aware false-negative masking (Khosla 2020, Literature_agent_phase3.md Q3): for each
    anchor, in-batch items sharing its (sic2, fiscal_year) are masked out of the negative set —
    they are neither the positive nor a valid negative. Default MNRL would push them apart.
  - Per-view weighting (Q1): soft sector positives are down-weighted (lambda_sector < 1).

Encoder = models.Transformer(max_seq_length=256) + mean Pooling + L2 Normalize.
Loss = weighted InfoNCE, cos-sim scaled by 20.0 (τ=0.05, SimCSE default).

Usage:
    # dual-view (Chiu replication)
    python contrastive/train_contrastive.py --views lexical,chrono \
        --out_dir contrastive_checkpoints/dual
    # three-view
    python contrastive/train_contrastive.py --views lexical,chrono,sector \
        --lambda_sector 0.5 --out_dir contrastive_checkpoints/three
    # three-view + LoRA
    python contrastive/train_contrastive.py --views lexical,chrono,sector \
        --lambda_sector 0.5 --lora --out_dir contrastive_checkpoints/three_lora
"""

import argparse
import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models

ROOT = Path(__file__).resolve().parent.parent
PAIRS_DIR = ROOT / "contrastive" / "pairs"
BASE_MODEL = ROOT / "dapt_checkpoints_para" / "best"
MAX_LEN = 256
SCALE = 20.0  # cos-sim scale ⇒ τ = 1/20 = 0.05 (SimCSE / MNRL default)
ALL_VIEWS = ["lexical", "chrono", "sector"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--views", type=str, default="lexical,chrono,sector",
                   help="Comma-separated subset of lexical,chrono,sector")
    p.add_argument("--base_model", type=str, default=str(BASE_MODEL))
    p.add_argument("--pairs_dir", type=str, default=str(PAIRS_DIR))
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pairs_per_view", type=int, default=10000)
    p.add_argument("--val_per_view", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lambda_sector", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_targets", type=str, default="q,v")
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_examples(pairs_dir, views, per_view, val_per_view, rng):
    """Return (train, val) example lists, subsampled and balanced per view."""
    train, val = [], []
    for view in views:
        rows = []
        path = Path(pairs_dir) / f"pairs_{view}.jsonl"
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        rng.shuffle(rows)
        val += rows[:val_per_view]
        train += rows[val_per_view : val_per_view + per_view]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_model(base_model, lora, lora_r, lora_targets, device):
    """Returns (model, peft_handle). peft_handle is None unless --lora.

    get_peft_model injects LoRA layers in-place into word.auto_model, so we do NOT
    reassign the attribute (that does not stick on models.Transformer); we keep the
    PeftModel handle only for merge_and_unload() at save time.
    """
    word = models.Transformer(base_model, max_seq_length=MAX_LEN)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    norm = models.Normalize()
    model = SentenceTransformer(modules=[word, pool, norm], device=device)
    peft_handle = None
    if lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1, bias="none",
            target_modules=[t.strip() for t in lora_targets.split(",")],
        )
        peft_handle = get_peft_model(word.auto_model, cfg)  # in-place LoRA injection
        peft_handle.print_trainable_parameters()
    return model, peft_handle


def encode(model, texts, device):
    feats = model.tokenize(texts)
    feats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
    return model(feats)["sentence_embedding"]


def key_ids(keys):
    """Map (sic2, fy) tuples to contiguous int ids for fast masking."""
    uniq = {}
    return [uniq.setdefault(k, len(uniq)) for k in keys]


def masked_infonce(emb_a, emb_p, a_keys, p_keys, weights, device):
    """Weighted InfoNCE with label-aware false-negative masking.

    For anchor i: positive is p_i; mask p_j (j!=i) whose (sic2,fy) == anchor i's.
    """
    B = emb_a.size(0)
    sim = (emb_a @ emb_p.t()) * SCALE
    a_id = torch.tensor(a_keys, device=device)
    p_id = torch.tensor(p_keys, device=device)
    eye = torch.eye(B, dtype=torch.bool, device=device)
    false_neg = (a_id[:, None] == p_id[None, :]) & (~eye)
    sim = sim.masked_fill(false_neg, float("-inf"))
    labels = torch.arange(B, device=device)
    per = F.cross_entropy(sim, labels, reduction="none")
    w = torch.tensor(weights, device=device, dtype=per.dtype)
    loss = (per * w).sum() / w.sum()
    acc = (sim.argmax(dim=1) == labels).float().mean()
    return loss, acc


def run_epoch(model, examples, batch_size, lambda_map, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    for s in range(0, len(examples), batch_size):
        batch = examples[s : s + batch_size]
        if len(batch) < 2:
            continue
        a_texts = [b["anchor_text"] for b in batch]
        p_texts = [b["positive_text"] for b in batch]
        # anchor + positive keys must share ONE id space so masking compares correctly
        all_keys = [(b["anchor_sic2"], b["anchor_fy"]) for b in batch] + \
                   [(b["positive_sic2"], b["positive_fy"]) for b in batch]
        ids = key_ids(all_keys)
        a_keys, p_keys = ids[: len(batch)], ids[len(batch):]
        weights = [lambda_map[b["view"]] for b in batch]

        ctx = torch.amp.autocast("cuda") if scaler else nullcontext()
        with torch.set_grad_enabled(train):
            with ctx:
                emb_a = encode(model, a_texts, device)
                emb_p = encode(model, p_texts, device)
                loss, acc = masked_infonce(emb_a, emb_p, a_keys, p_keys, weights, device)
        if train:
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(batch)
        total_acc += acc.item() * len(batch)
        n += len(batch)
    return total_loss / max(n, 1), total_acc / max(n, 1)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    views = [v.strip() for v in args.views.split(",")]
    lambda_map = {"lexical": 1.0, "chrono": 1.0, "sector": args.lambda_sector}

    print(f"Device: {device} | views: {views} | lora: {args.lora}")
    train, val = load_examples(args.pairs_dir, views, args.pairs_per_view, args.val_per_view, rng)
    print(f"Train examples: {len(train):,} | val: {len(val):,}")

    model, peft_handle = build_model(args.base_model, args.lora, args.lora_r, args.lora_targets, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if (args.fp16 and device == "cuda") else None

    best_acc, best_epoch, bad, best_state = -1.0, -1, 0, None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train)
        tr_loss, tr_acc = run_epoch(model, train, args.batch_size, lambda_map, device, optimizer, scaler)
        with torch.no_grad():
            va_loss, va_acc = run_epoch(model, val, args.batch_size, lambda_map, device)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f}", flush=True)

        if va_acc > best_acc:
            best_acc, best_epoch, bad = va_acc, epoch, 0
            # keep best weights (incl. LoRA adapters) on CPU; merge/save once at the end
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if peft_handle is not None:
        # bake adapters into base weights in-place so the saved dir is a plain encoder
        peft_handle.merge_and_unload()
    model.save(str(out_dir))

    summary = {"views": views, "lora": args.lora, "best_val_acc": best_acc,
               "best_epoch": best_epoch, "lambda_sector": args.lambda_sector,
               "pairs_per_view": args.pairs_per_view}
    with open(out_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest val acc {best_acc:.3f} @ epoch {best_epoch} -> saved {out_dir}")


if __name__ == "__main__":
    main()
