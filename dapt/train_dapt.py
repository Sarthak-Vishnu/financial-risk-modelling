"""
Phase 2 — Step 2: Continued MLM pretraining (DAPT) of all-mpnet-base-v2.

Trains on dapt_data/train.jsonl, evaluates on dapt_data/val.jsonl.
Best checkpoint selected on val loss (perplexity). Logs to W&B if
WANDB_API_KEY is set, otherwise logs to local files only.

Usage:
    python dapt/train_dapt.py [--epochs N] [--batch_size N] [--lr LR]

Checkpoints saved to: dapt_checkpoints/
Best checkpoint:      dapt_checkpoints/best/
"""

import argparse
import math
import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dapt_data"
CHECKPOINT_DIR = ROOT / "dapt_checkpoints"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 512


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--early_stopping_patience", type=int, default=2)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Truncate train set to N samples (smoke test)")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Truncate val set to N samples (smoke test)")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                   help="Directory containing train.jsonl / val.jsonl")
    p.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR),
                   help="Directory to write checkpoints and best/ to")
    return p.parse_args()


def tokenize_fn(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    report_to = "wandb" if use_wandb else "none"
    if use_wandb:
        import wandb
        wandb.init(project="financial-risk-dapt", name="dapt-mpnet-base-v2")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    print("Loading dataset...")
    raw = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    if args.max_train_samples is not None:
        raw["train"] = raw["train"].select(range(min(args.max_train_samples, len(raw["train"]))))
    if args.max_eval_samples is not None:
        raw["validation"] = raw["validation"].select(range(min(args.max_eval_samples, len(raw["validation"]))))

    print("Tokenizing...")
    tokenized = raw.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = math.ceil(len(tokenized["train"]) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to=report_to,
        save_total_limit=3,        # keep last 3 checkpoints + best
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Starting DAPT training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save best model to a fixed 'best/' directory for easy reference
    best_dir = checkpoint_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Report final val perplexity
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"\nBest checkpoint val perplexity: {perplexity:.4f}")
    print(f"Saved to: {best_dir}")

    if use_wandb:
        import wandb
        wandb.log({"best_val_perplexity": perplexity})
        wandb.finish()


if __name__ == "__main__":
    main()
