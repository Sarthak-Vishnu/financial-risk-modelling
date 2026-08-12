#!/bin/bash
#SBATCH --job-name=gridxn
#SBATCH --partition=Teaching
#SBATCH --exclude=opencast
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/grid_xnode_%j.out
#SBATCH --error=logs/grid_xnode_%j.err

# Run D — the cross-node leg of the reproducibility control, and the gate on calling any grid
# "canonical".
#
# run_control.sh (job 3593789) established, all on `opencast`:
#   A (orig 6 encoders, unpinned) == NEW (7 encoders, unpinned)   bit-for-bit, 23/23 rows
#     -> adding an encoder to the list caused none of the drift.
#   B == C (7 encoders, threads pinned to 1)                      bit-for-bit, 29/29 rows
#     -> pinning is deterministic WITHIN a node.
#   NEW vs B                                                      max |dIC| 1.3e-04
#     -> thread count 2->1 is a 1e-4 effect, three orders too small to explain the
#        1.8e-2 gap between the July grid and today's.
# numpy/scipy/scikit-learn were all installed 2026-05-19, before the July run, so library
# version drift is ruled out too. What is left is the node: OpenBLAS dispatches its kernels on
# runtime CPU capability, so a different microarchitecture reduces floats in a different order.
#
# B == C proves same-node determinism only. Both ran inside one job on one node. Declaring a
# published table on that basis would assert something the evidence does not cover. This run
# repeats B's exact configuration somewhere other than opencast:
#   D == B  -> pinning survives a node change; the pinned grid can be called canonical.
#   D != B  -> pinning is not sufficient, and the noise floor stands instead.
#
# Writes only control_D.json. The canonical grid is not touched, and no backtest is invoked.

source ~/.bashrc
conda activate diss
cd /home/s2880814/financial-risk-modelling

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== environment ==="
echo "node        : $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "nproc       : $(nproc)"
echo "OMP_NUM_THREADS: '${OMP_NUM_THREADS}'"
grep -m1 "model name" /proc/cpuinfo
python - <<'PY'
import numpy, sklearn, scipy, sys
print("python  :", sys.version.split()[0], "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| sklearn", sklearn.__version__)
try:
    from threadpoolctl import threadpool_info
    for d in threadpool_info():
        print(f"  pool: {d.get('user_api')}/{d.get('internal_api')} threads={d.get('num_threads')}")
except ImportError:
    print("  (threadpoolctl not installed)")
PY

# Same trick as run_control.sh: edit a copy so the committed script stays untouched, and keep the
# copy in phase5/ so ROOT resolution and the eval_common import behave identically.
cp phase5/stress_grid.py phase5/_ctl_D.py
python - phase5/_ctl_D.py <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s, n = re.subn(r'out = OUT_DIR / f"stress_grid_val2025\{E\.CACHE_SUFFIX\}\.json"',
               'out = OUT_DIR / __import__("os").environ["CTL_OUT"]', s, count=1)
assert n == 1, "output line not matched"
open(p, "w").write(s)
print("  variant phase5/_ctl_D.py: output redirected")
PY

echo
date +"start %F %T"
CTL_OUT="control_D.json" python phase5/_ctl_D.py --mode val2025
date +"end   %F %T"
rm -f phase5/_ctl_D.py

echo
echo "=== D vs B (both pinned; different nodes) ==="
python - <<'PY'
import json
from pathlib import Path
O = Path("phase5/out")
def rows(p):
    d = json.load(open(p)); R = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return {x["name"]: x for x in R}
if not (O / "control_D.json").exists():
    print("control_D.json missing — run failed"); raise SystemExit(1)
B, D = rows(O / "control_B.json"), rows(O / "control_D.json")
sh = [n for n in B if n in D]
moved = [(abs(B[n]["ic"] - D[n]["ic"]), n, B[n]["ic"], D[n]["ic"]) for n in sh]
moved = [m for m in moved if m[0] > 0]
moved.sort(reverse=True)
print(f"shared rows {len(sh)} | identical IC {len(sh)-len(moved)}/{len(sh)} | "
      f"max |dIC| {(max(m[0] for m in moved) if moved else 0.0):.3e}")
for d, n, b, dd in moved[:20]:
    print(f"   {n:42s} B {b:.10f} -> D {dd:.10f}  ({dd-b:+.2e})")
print()
print("VERDICT: pinning survives the node change — pinned grid may be called canonical."
      if not moved else
      "VERDICT: pinning does NOT survive a node change — do not publish a pinned grid as canonical.")
PY
