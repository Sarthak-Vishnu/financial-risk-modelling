#!/bin/bash
#SBATCH --job-name=gridctl
#SBATCH --partition=Teaching
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/grid_control_%j.out
#SBATCH --error=logs/grid_control_%j.err

# Controlled comparison for the val-2025 grid's non-reproducibility.
#
# Observed: adding `three` to ENCODERS moved tree-head lanes by up to 0.018 IC while ridge and
# sparse lanes held to 4e-5. Rows computed BEFORE the inserted encoder moved as much as rows
# after it, which rules out a shifted random stream — but that is an ordering argument, not a
# controlled one. Three runs settle it:
#
#   A  original 6 encoders, threads left as the environment provides them.
#      A vs the preserved PREV grid isolates the encoder-list change: if A still fails to
#      reproduce PREV, the list is not the cause and the environment is.
#   B  current 7 encoders, every thread pool pinned to 1.
#   C  identical to B. B vs C tests whether pinning actually buys determinism, which is the
#      remedy the write-up would recommend.
#
# None of these touch phase5/out/stress_grid_val2025_fixed.json. Each writes its own
# control_*.json, which stays gitignored. Backtests are never invoked (--mode val2025 only).

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1

echo "=== environment ==="
echo "node        : $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "cpus-per-task: $SLURM_CPUS_PER_TASK"
echo "nproc       : $(nproc)"
echo "OMP_NUM_THREADS (inherited): '${OMP_NUM_THREADS:-<unset>}'"
python - <<'PY'
import numpy, sklearn, scipy, sys
print("python  :", sys.version.split()[0])
print("numpy   :", numpy.__version__)
print("scipy   :", scipy.__version__)
print("sklearn :", sklearn.__version__)
try:
    from threadpoolctl import threadpool_info
    for d in threadpool_info():
        print(f"  pool: {d.get('user_api')} / {d.get('internal_api')} "
              f"threads={d.get('num_threads')} {d.get('filepath','')}")
except ImportError:
    print("  (threadpoolctl not installed — thread pools not enumerable)")
PY

# Build a variant of stress_grid.py that writes to a control filename, and optionally reverts
# ENCODERS to the pre-rerun list. Editing a copy keeps the committed script untouched; the copy
# lives in phase5/ so its ROOT resolution and eval_common import behave identically.
make_variant () {   # $1 = tag, $2 = "orig6" | "curr7"
    local tag="$1" enc="$2" src=phase5/stress_grid.py dst="phase5/_ctl_${1}.py"
    cp "$src" "$dst"
    python - "$dst" "$enc" <<'PY'
import re, sys
path, enc = sys.argv[1], sys.argv[2]
s = open(path).read()
if enc == "orig6":
    new = 'ENCODERS = ["dual", "sbert", "volaware", "three_lora", "ftvol", "bge"]'
    s, n = re.subn(r'^ENCODERS = \[.*\]$', new, s, count=1, flags=re.M)
    assert n == 1, "ENCODERS line not matched"
s, n = re.subn(r'out = OUT_DIR / f"stress_grid_val2025\{E\.CACHE_SUFFIX\}\.json"',
               'out = OUT_DIR / __import__("os").environ["CTL_OUT"]', s, count=1)
assert n == 1, "output line not matched"
open(path, "w").write(s)
print(f"  variant {path}: encoders={enc}, output redirected")
PY
}

run_one () {        # $1 = tag, $2 = encoder set, $3 = pin threads (yes/no)
    local tag="$1"
    echo
    echo "=========================================================="
    echo "=== RUN $tag  (encoders=$2, pinned=$3) ==="
    date +"    start %F %T"
    make_variant "$tag" "$2"
    if [ "$3" = "yes" ]; then
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
               NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
    else
        unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
              NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
    fi
    echo "    OMP_NUM_THREADS='${OMP_NUM_THREADS:-<unset>}'"
    CTL_OUT="control_${tag}.json" python "phase5/_ctl_${tag}.py" --mode val2025
    date +"    end   %F %T"
    rm -f "phase5/_ctl_${tag}.py"
}

run_one A orig6 no
run_one B curr7 yes
run_one C curr7 yes

echo
echo "=== COMPARISONS ==="
python - <<'PY'
import json, itertools
from pathlib import Path
OUT = Path("phase5/out")

def rows(p):
    d = json.load(open(p))
    R = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return {x["name"]: x for x in R}

def cmp(la, pa, lb, pb):
    if not (Path(pa).exists() and Path(pb).exists()):
        print(f"\n--- {la} vs {lb}: MISSING FILE, cannot compare ---"); return
    A, B = rows(pa), rows(pb)
    shared = [k for k in A if k in B]
    print(f"\n--- {la} vs {lb} ---  shared rows: {len(shared)} "
          f"(only-{la}: {len(A)-len(shared)}, only-{lb}: {len(B)-len(shared)})")
    worst, moved = 0.0, []
    for k in shared:
        d = abs(A[k]["ic"] - B[k]["ic"])
        worst = max(worst, d)
        if d > 1e-9:
            moved.append((d, k, A[k]["ic"], B[k]["ic"]))
    moved.sort(reverse=True)
    print(f"    identical IC: {len(shared)-len(moved)}/{len(shared)}   max |dIC|: {worst:.2e}")
    for d, k, a, b in moved[:12]:
        print(f"      {k:42s} {a:.4f} -> {b:.4f}  ({b-a:+.4f})")

P = str(OUT / "stress_grid_val2025_fixed.PREV.json")
N = str(OUT / "stress_grid_val2025_fixed.json")
cmp("PREV", P, "A(orig6)",  str(OUT / "control_A.json"))
cmp("NEW",  N, "A(orig6)",  str(OUT / "control_A.json"))
cmp("B",    str(OUT / "control_B.json"), "C", str(OUT / "control_C.json"))
cmp("NEW",  N, "B(pinned)", str(OUT / "control_B.json"))
PY

echo
echo "=== DONE ==="
echo "Reads: PREV vs A isolates the encoder-list change; B vs C tests pinning."
