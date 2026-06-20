"""
Phase 3 — Increment 3: FinMTEB intrinsic evaluation.

Scores SentenceTransformer-style encoders on the FinMTEB English STS + Retrieval
sub-tasks (Tang & Yang 2025). Primary metrics: Spearman ρ (STS), NDCG@10 (Retrieval).

Runs in the SEPARATE `finmteb` conda env (not `diss`) so the pinned FinMTEB stack does
not clobber the training env. The `finance_mteb` package lives in the cloned repo, added
to sys.path below.

Models evaluated (default): our contrastive encoders (dual / three / three_lora) plus
baselines SBERT (all-mpnet-base-v2, unadapted) and DAPT-only — all mean-pooling
SentenceTransformers, passed directly to MTEB.

Fin-E5 (FinanceMTEB/FinE5, 7B, last-token pooling + instructions) is NOT scored here —
use the repo's own runner:
    python ~/FinMTEB/eval_FinanceMTEB.py --model_name_or_path FinanceMTEB/FinE5 --pooling_method last

Usage (in the finmteb env):
    conda activate finmteb
    python contrastive/eval_finmteb.py --tasks sts                 # quick STS-only
    python contrastive/eval_finmteb.py --tasks all                 # STS + Retrieval
    python contrastive/eval_finmteb.py --models "sbert=sentence-transformers/all-mpnet-base-v2" --tasks FinSTS
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/FinMTEB"))  # local finance_mteb package
from finance_mteb import MTEB
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
ENG_STS = ["FinSTS", "FINAL"]
ENG_RETRIEVAL = [
    "FiQA2018Retrieval", "Apple10KRetrieval", "FinQARetrieval", "FinanceBenchRetrieval",
    "HC3Retrieval", "TATQARetrieval", "TheGoldmanEnRetrieval",
    "TradeTheEventEncyclopediaRetrieval", "USNewsRetrieval", "TradeTheEventNewsRetrieval",
]

DEFAULT_MODELS = {
    "sbert": "sentence-transformers/all-mpnet-base-v2",
    "dapt": str(ROOT / "dapt_checkpoints_para" / "best"),
    "dual": str(ROOT / "contrastive_checkpoints" / "dual"),
    "three": str(ROOT / "contrastive_checkpoints" / "three"),
    "three_lora": str(ROOT / "contrastive_checkpoints" / "three_lora"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=None,
                   help='Comma list of name=path. Default: sbert,dapt,dual,three,three_lora')
    p.add_argument("--tasks", type=str, default="all",
                   help='"sts" | "retrieval" | "all" | comma-separated task names')
    p.add_argument("--output_dir", type=str, default=str(ROOT / "eval_results" / "finmteb"))
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def resolve_models(spec):
    if not spec:
        return DEFAULT_MODELS
    out = {}
    for item in spec.split(","):
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def resolve_tasks(spec):
    key = spec.strip().lower()
    if key == "sts":
        return ENG_STS
    if key == "retrieval":
        return ENG_RETRIEVAL
    if key == "all":
        return ENG_STS + ENG_RETRIEVAL
    return [t.strip() for t in spec.split(",")]


def find_main_score(obj):
    """Recursively pull the first 'main_score' from an mteb result JSON."""
    if isinstance(obj, dict):
        if "main_score" in obj and isinstance(obj["main_score"], (int, float)):
            return obj["main_score"]
        for v in obj.values():
            r = find_main_score(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_main_score(v)
            if r is not None:
                return r
    return None


def collect_scores(model_out_dir, tasks):
    """Read per-task result JSONs and return {task: main_score}."""
    scores = {}
    for task in tasks:
        hits = list(Path(model_out_dir).rglob(f"{task}.json"))
        if not hits:
            scores[task] = None
            continue
        with open(hits[0]) as f:
            scores[task] = find_main_score(json.load(f))
    return scores


def main():
    args = parse_args()
    models = resolve_models(args.models)
    tasks = resolve_tasks(args.tasks)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Models: {list(models)}")
    print(f"Tasks ({len(tasks)}): {tasks}")

    results = {}
    for name, path in models.items():
        print(f"\n===== {name}  ({path}) =====")
        model = SentenceTransformer(path)
        model_out = out_root / name
        # Do NOT force eval_splits: STS tasks use "test", but every FinMTEB
        # Retrieval task declares eval_splits=["train"]. Passing eval_splits=None
        # lets run() fall back to each task's own metadata split (MTEB.py:322).
        MTEB(tasks=tasks).run(model, output_folder=str(model_out),
                              overwrite_results=False,
                              batch_size=args.batch_size)
        results[name] = collect_scores(model_out, tasks)

    # summary table: rows = models, cols = tasks + mean
    print("\n=== FinMTEB main-score summary ===")
    header = ["model"] + tasks + ["MEAN"]
    print("\t".join(header))
    summary = {}
    for name, sc in results.items():
        vals = [sc.get(t) for t in tasks]
        present = [v for v in vals if isinstance(v, (int, float))]
        mean = sum(present) / len(present) if present else None
        summary[name] = {"scores": sc, "mean": mean}
        row = [name] + [f"{v:.4f}" if isinstance(v, (int, float)) else "—" for v in vals]
        row += [f"{mean:.4f}" if mean is not None else "—"]
        print("\t".join(row))

    with open(out_root / "summary.json", "w") as f:
        json.dump({"tasks": tasks, "models": summary}, f, indent=2)
    print(f"\nSaved summary -> {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
