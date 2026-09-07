#!/bin/bash
# One unofficial plumbing smoke tick. Never writes EXPERIMENT_LOG.
# Checks: CONTAINER required, parse_ncu.py on the L4 CSVs, tiny host ncu.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="$SCRIPT_DIR/runs/dry_run_t4"
PASS=0
FAIL=0

mkdir -p "$OUT"

ok() { echo "[pass] $*"; PASS=$((PASS + 1)); }
bad() { echo "[fail] $*"; FAIL=$((FAIL + 1)); }

echo "=== opt_loop dry run (unofficial) ==="

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r' || true)"
echo "HOST_GPU=${GPU_NAME:-unknown}"
ok "recorded host GPU (${GPU_NAME:-unknown}); L4 is not required"

unset CONTAINER CONTAINER_ID CONTAINER_NAME OPT_LOOP_EXEC
rm -f "$SCRIPT_DIR/container.env"

set +e
"$SCRIPT_DIR/start.sh" >"$OUT/start.out" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]] && grep -q "CONTAINER is required" "$OUT/start.out"; then
  ok "start.sh refuses when CONTAINER is unset"
else
  bad "start.sh should require CONTAINER (rc=$rc, see $OUT/start.out)"
fi

set +e
"$SCRIPT_DIR/measure.sh" >"$OUT/measure.out" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]] && grep -q "CONTAINER is required" "$OUT/measure.out"; then
  ok "measure.sh refuses when CONTAINER is unset"
else
  bad "measure.sh should require CONTAINER (rc=$rc, see $OUT/measure.out)"
fi

UP_CSV="$KERNELS/gemv_up_proj_ncu.csv"
DN_CSV="$KERNELS/gemv_down_proj_ncu.csv"
if [[ -f "$UP_CSV" && -f "$DN_CSV" ]]; then
  python3 "$SCRIPT_DIR/parse_ncu.py" "$UP_CSV" --shape up_proj --out "$OUT/l4_up_proj.json" \
    >"$OUT/parse_l4_up.out" 2>"$OUT/parse_l4_up.err"
  python3 "$SCRIPT_DIR/parse_ncu.py" "$DN_CSV" --shape down_proj --out "$OUT/l4_down_proj.json" \
    >"$OUT/parse_l4_down.out" 2>"$OUT/parse_l4_down.err"
  python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
up = json.loads((d / "l4_up_proj.json").read_text())
dn = json.loads((d / "l4_down_proj.json").read_text())
assert up["cc"] == "8.9" and dn["cc"] == "8.9"
assert abs(up["duration_us"] - 215.552) < 0.01
assert abs(dn["duration_us"] - 208.896) < 0.01
assert up["gain_vs_trt_pct"] is not None and up["gain_vs_trt_pct"] < 25
print("l4 parse ok", up["gain_vs_trt_pct"], dn["gain_vs_trt_pct"])
PY
  if [[ $? -eq 0 ]]; then
    ok "parse_ncu.py on L4 CSVs (sm_89, ~3.9% / ~7.0%)"
  else
    bad "parse_ncu.py L4 CSV check"
  fi
else
  bad "missing kernels/gemv_{up,down}_proj_ncu.csv"
fi

METRICS="dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
DUMMY_SRC="$SCRIPT_DIR/dummy_kernel.cu"
DUMMY_BIN="$OUT/dummy_kernel"
DUMMY_CSV="$OUT/dummy_ncu.csv"

nvcc -O3 -arch=sm_75 -o "$DUMMY_BIN" "$DUMMY_SRC"
if [[ $? -eq 0 ]]; then
  ok "nvcc dummy_kernel.cu (sm_75)"
else
  bad "nvcc dummy_kernel.cu"
fi

if [[ -x "$DUMMY_BIN" ]]; then
  ncu --csv --launch-skip 0 --launch-count 1 --metrics "$METRICS" \
      "$DUMMY_BIN" >"$DUMMY_CSV" 2>"$OUT/dummy_ncu.err"
  ncu_rc=$?
  if [[ $ncu_rc -eq 0 ]]; then
    ok "ncu dummy kernel (unofficial host)"
    python3 "$SCRIPT_DIR/parse_ncu.py" "$DUMMY_CSV" --shape up_proj --out "$OUT/dummy.json" \
      >"$OUT/parse_dummy.out" 2>"$OUT/parse_dummy.err"
    if grep -q "compute capability" "$OUT/parse_dummy.err"; then
      ok "parse_ncu.py noted non-sm_89 CC"
    else
      bad "parse_ncu.py should note T4 CC (see $OUT/parse_dummy.err)"
    fi
  else
    bad "ncu dummy kernel rc=$ncu_rc (see $OUT/dummy_ncu.err)"
  fi
fi

python3 - "$OUT" "${GPU_NAME:-unknown}" "$PASS" "$FAIL" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
gpu, passed, failed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dummy = {}
p = d / "dummy.json"
if p.exists():
    dummy = json.loads(p.read_text())
summary = {
    "tag": "dry_run_t4",
    "official": False,
    "gpu": gpu,
    "passed": passed,
    "failed": failed,
    "dummy": dummy,
    "note": "Plumbing smoke test only. Do not copy dummy timings into EXPERIMENT_LOG or VERDICT.md.",
}
(d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "=== dry run: $PASS passed, $FAIL failed ==="
echo "Wrote $OUT (official=false)"
[[ $FAIL -eq 0 ]]
