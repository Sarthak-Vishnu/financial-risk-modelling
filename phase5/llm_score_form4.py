"""
Round 8 — score the anonymised Form 4 windows with an open-weights generative model.

One prompt per filing (build_form4_llm_prompts.py), K independent samples each. The K samples are
the point as much as the score is: their spread is this study's own measurement of the score
instability that the literature asserts (Sclar et al. 2024,
Ouyang et al. 2024). An assertion backed by a number on our own data is worth more than one backed
only by citations.

Model: Qwen2.5-7B-Instruct by default. Chosen for a documented late-2023 training cutoff, which is
what makes the 2024+2025 evaluation slice genuinely post-cutoff. Change the model and that argument
has to be re-made, not assumed — RISK_LLM_MODEL exists for ablation, not for convenience.

Plain transformers with batched generation rather than vLLM: ~865 prompts x 5 samples does not
justify the install risk on a shared cluster, and transformers is already in the environment.

Determinism note: the run is seeded, but seeding does NOT make this reproducible in the way the
rest of the pipeline is. Sampling at temperature > 0 is intended here — measuring dispersion is
the point — and Ouyang et al. (2024) find that even temperature 0 fails to guarantee identical
output. Treat the saved scores as data, not as a reproducible function of the inputs.

Output: datasets/form4_llm_scores{_ident}.parquet — one row per filing, with per-sample scores,
their mean and standard deviation, and the parse failure count.
Run:  python phase5/llm_score_form4.py          (needs a GPU; see run_llm_form4.sbatch)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"

ANON = os.environ.get("RISK_LLM_ANON", "1") != "0"
SUF = "" if ANON else "_ident"
PROMPTS = DATA / f"form4_llm_prompts{SUF}.parquet"
OUT_PATH = DATA / f"form4_llm_scores{SUF}.parquet"

MODEL = os.environ.get("RISK_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
K = int(os.environ.get("RISK_LLM_K", "5"))            # samples per filing
TEMP = float(os.environ.get("RISK_LLM_TEMP", "0.7"))
BATCH = int(os.environ.get("RISK_LLM_BATCH", "8"))    # prompts per forward pass (x K sequences)
MAX_NEW = 96
SEED = 42

FIELDS = ["volatility_risk", "information_asymmetry", "confidence"]
SYSTEM = ("You are a careful financial analyst. You answer only with the JSON object you are "
          "asked for, with no commentary before or after it.")


def parse_scores(text):
    """Pull the JSON object out of a completion. Returns dict of field -> float in [0,100], or None.

    Models wrap JSON in prose or code fences often enough that a bare json.loads is not sufficient,
    so fall back to per-field regex before giving up. Every failure is counted and reported: a high
    failure rate is itself a finding about the route's viability."""
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            d = {f: float(o[f]) for f in FIELDS if f in o and o[f] is not None}
            if len(d) == len(FIELDS) and all(0 <= v <= 100 for v in d.values()):
                return d
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    d = {}
    for f in FIELDS:
        mm = re.search(rf'"?{f}"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
        if mm:
            v = float(mm.group(1))
            if 0 <= v <= 100:
                d[f] = v
    return d if len(d) == len(FIELDS) else None


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not PROMPTS.exists():
        raise SystemExit(f"{PROMPTS} not found — run build_form4_llm_prompts.py first.")
    df = pd.read_parquet(PROMPTS)
    print(f"Prompts: {len(df):,} | anonymised: {ANON} | model: {MODEL} | K={K} temp={TEMP}")

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"                      # required for correct batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # device_map="auto" rather than "cuda": shards across whatever GPUs the job was given, so the
    # same script runs on one 71GB H200 slice or on 2x11GB 2080Ti cards when the H200s are busy.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    devs = {str(p.device) for p in model.parameters()}
    print(f"Loaded across {sorted(devs)}")

    chats = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in df["prompt"]]
    lens = np.array([len(tok(c).input_ids) for c in chats])
    print(f"Prompt tokens: median {np.median(lens):.0f}  p90 {np.percentile(lens, 90):.0f}  "
          f"max {lens.max()}  total {lens.sum():,}")

    # sort by length so each batch pads to a similar width — the padding waste on a
    # 700-to-19,000 token spread is otherwise most of the compute
    order = np.argsort(lens)
    rows = {i: [] for i in range(len(df))}
    n_fail = {i: 0 for i in range(len(df))}

    def generate(ids):
        enc = tok([chats[i] for i in ids], return_tensors="pt",
                  padding=True, truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=0.9,
                                 num_return_sequences=K, max_new_tokens=MAX_NEW,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

    # Fail fast. Prompts are processed shortest-first (for padding efficiency), so the memory
    # worst case is the LAST batch — an 18GB MIG slice would OOM only after ~50 min of work.
    # Try the single longest prompt up front instead, where failing costs 30 seconds.
    try:
        generate([int(order[-1])])
        print(f"warm-up on the longest prompt ({lens.max()} tokens) OK", flush=True)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            f"OOM generating a SINGLE longest prompt ({lens.max()} tokens). This GPU is too small "
            f"for {MODEL}. Use a larger slice, or set RISK_LLM_MODEL to a smaller model of the "
            f"same family (Qwen/Qwen2.5-3B-Instruct has the same training cutoff, so the "
            f"post-cutoff argument for the 2024-2025 slice survives the swap).")

    t0, pos, bs, nb = time.time(), 0, BATCH, 0
    while pos < len(order):
        idx = order[pos:pos + bs]
        try:
            texts = generate(idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise SystemExit("OOM at batch size 1 despite the warm-up passing — "
                                 "reduce RISK_LLM_K or use a larger slice.")
            bs = max(1, bs // 2)
            print(f"  [oom] batch size -> {bs}, retrying", flush=True)
            continue
        for j, i in enumerate(idx):
            for k in range(K):
                d = parse_scores(texts[j * K + k])
                if d is None:
                    n_fail[i] += 1
                else:
                    rows[i].append(d)
        pos += len(idx)
        nb += 1
        if nb % 10 == 0 or pos >= len(order):
            el = time.time() - t0
            print(f"  {pos:>5d}/{len(order)}  bs={bs}  {el/60:6.1f} min elapsed  "
                  f"eta {el/max(pos,1)*(len(order)-pos)/60:6.1f} min", flush=True)

    recs = []
    for i in range(len(df)):
        r = {"ticker": df.ticker.iloc[i], "filing_date": df.filing_date.iloc[i],
             "year": int(df.year.iloc[i]), "n_txn": int(df.n_txn.iloc[i]),
             "llm_n_ok": len(rows[i]), "llm_n_fail": n_fail[i]}
        for f in FIELDS:
            v = np.array([d[f] for d in rows[i]], dtype=float)
            r[f"llm_{f}"] = float(v.mean()) if len(v) else np.nan
            r[f"llm_{f}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            r[f"llm_{f}_s0"] = float(v[0]) if len(v) else np.nan   # a single draw, for the
        recs.append(r)                                             # averaging-gain comparison
    res = pd.DataFrame(recs)
    res.to_parquet(OUT_PATH, index=False)

    print(f"\nSaved {OUT_PATH}  ({len(res):,} filings, {time.time()-t0:.0f}s)")
    tot = res.llm_n_ok.sum() + res.llm_n_fail.sum()
    print(f"Parse failures: {res.llm_n_fail.sum():,}/{tot:,} samples "
          f"({res.llm_n_fail.sum()/max(tot,1):.2%}) | filings with no usable sample: "
          f"{int((res.llm_n_ok == 0).sum())}")
    print("\nScore distribution and within-filing sample spread:")
    for f in FIELDS:
        s, sd = res[f"llm_{f}"], res[f"llm_{f}_sd"]
        print(f"  {f:22s} mean {s.mean():6.1f}  sd(across filings) {s.std():5.1f}  "
              f"|  median within-filing sd across {K} samples: {sd.median():5.1f}")
    print("\nThe last column is the instability measurement: a within-filing spread that is large "
          "relative to the across-filing spread means the score is mostly noise.")


if __name__ == "__main__":
    main()
