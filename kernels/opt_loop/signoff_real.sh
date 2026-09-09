#!/bin/bash
# Sign-off: real Llama-3.2-3B MLP weights + moment-matched synthetic activations,
# GPU checkreal + cold ncu on the shipped FP8 kernel.
set -euo pipefail

CONTAINER="${CONTAINER:?set CONTAINER}"
ROOT="/workspace/opt-worktrees/fp8"
KDIR="$ROOT/kernels"
LOOP="$KDIR/opt_loop"
OUT="$LOOP/runs/signoff_real_synth"
WDIR="/workspace/.fp8_tools/weights"
ADIR="/workspace/.fp8_tools/activations"
SDIR="/workspace/.fp8_tools/synth_x"
SITE="/workspace/.fp8_tools/site"

mkdir -p "$OUT" "$SDIR"

echo "=== idle GPU check ==="
docker exec "$CONTAINER" nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
docker exec "$CONTAINER" nvidia-smi --query-gpu=name,compute_cap,clocks.sm,clocks.mem,ecc.mode.current,ecc.mode.pending --format=csv

echo "=== synthetic activations (match captured RMS / kurtosis) ==="
docker exec -w "$LOOP/fp8_accuracy" -e PYTHONPATH="$SITE" "$CONTAINER" \
  python3 make_synthetic_activations.py \
    --manifest "$ADIR/x_manifest.json" \
    --out "$SDIR" --seed 20260909

echo "=== numpy cosine real-W + synthetic-x ==="
docker exec -e PYTHONPATH="$SITE" "$CONTAINER" \
  python3 "$LOOP/fp8_accuracy/score_synthetic_x.py" \
    --weights "$WDIR/manifest.json" \
    --synth-manifest "$SDIR/synth_x_manifest.json" \
    --out "$OUT/synth_cosine.json"

echo "=== build gemv_fp8 ==="
docker exec -w "$KDIR" "$CONTAINER" nvcc -O3 -arch=sm_89 -lineinfo gemv_fp8.cu -o gemv_fp8

echo "=== GPU checkreal shipped (real W + synthetic x) ==="
: > "$OUT/checkreal_synth.txt"
for layer in 00 13 27; do
  for proj in up_proj down_proj; do
    if [[ "$proj" == up_proj ]]; then
      w="$WDIR/L${layer}.${proj}.K3072.N8192.fp16.bin"
      x="$SDIR/L${layer}.${proj}.K3072.synth.x.fp16.bin"
    else
      w="$WDIR/L${layer}.${proj}.K8192.N3072.fp16.bin"
      x="$SDIR/L${layer}.${proj}.K8192.synth.x.fp16.bin"
    fi
    echo "---- $w ----" | tee -a "$OUT/checkreal_synth.txt"
    docker exec -w "$KDIR" "$CONTAINER" \
      ./gemv_fp8 checkreal "$w" shipped "x=$x" | tee -a "$OUT/checkreal_synth.txt"
  done
done

echo "=== GPU checkreal shipped (real W + captured decode x) L13 ==="
: > "$OUT/checkreal_captured_L13.txt"
docker exec -w "$KDIR" "$CONTAINER" \
  ./gemv_fp8 checkreal "$WDIR/L13.up_proj.K3072.N8192.fp16.bin" shipped \
    "x=$ADIR/L13.up_proj.K3072.x.fp16.bin" | tee -a "$OUT/checkreal_captured_L13.txt"
docker exec -w "$KDIR" "$CONTAINER" \
  ./gemv_fp8 checkreal "$WDIR/L13.down_proj.K8192.N3072.fp16.bin" shipped \
    "x=$ADIR/L13.down_proj.K8192.x.fp16.bin" | tee -a "$OUT/checkreal_captured_L13.txt"

METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_aperture_device_lookup_miss.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__sass_thread_inst_executed_op_ffma_pred_on.sum"

ncu_shape() {
  local shape="$1" w="$2" x="$3"
  local csv="$OUT/${shape}_ncu.csv"
  echo "=== ncu $shape (idle, cold, real W + synth x) ==="
  docker exec "$CONTAINER" nvidia-smi --query-compute-apps=pid --format=csv,noheader || true
  docker exec -w "$KDIR" "$CONTAINER" \
    ncu --csv --kernel-name-base demangled --kernel-name "regex:gemv_chunk" \
        --launch-skip 0 --launch-count 1 --metrics "$METRICS" \
        ./gemv_fp8 oncereal "$w" "x=$x" \
    | tee "$csv"
  python3 "$LOOP/parse_ncu.py" "$csv" --shape "$shape" --unique-bytes 25165824 \
      --out "$OUT/${shape}.json"
}

exec 9>/tmp/gemv_opt_loop.ncu.lock
flock 9

ncu_shape up_proj \
  "$WDIR/L13.up_proj.K3072.N8192.fp16.bin" \
  "$SDIR/L13.up_proj.K3072.synth.x.fp16.bin"
ncu_shape down_proj \
  "$WDIR/L13.down_proj.K8192.N3072.fp16.bin" \
  "$SDIR/L13.down_proj.K8192.synth.x.fp16.bin"

python3 - "$OUT" <<'PY'
import json, os, sys
from pathlib import Path
d = Path(sys.argv[1])
up = json.loads((d / "up_proj.json").read_text())
dn = json.loads((d / "down_proj.json").read_text())
summary = {
    "tag": "signoff_real_synth",
    "up_proj": up,
    "down_proj": dn,
    "min_gain_vs_trt_pct": min(up["gain_vs_trt_pct"], dn["gain_vs_trt_pct"]),
    "hit_pass_bar_both": bool(up["hit_pass_bar"] and dn["hit_pass_bar"]),
    "hit_stretch_bar_both": bool(up["hit_stretch_bar"] and dn["hit_stretch_bar"]),
}
(d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Wrote $OUT"
