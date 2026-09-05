#!/bin/bash
# One cold ncu capture of the current kernel, inside the bound docker container.
#
# Usage:
#   CONTAINER=<id> ./measure.sh
#   ./measure.sh --container <id>
#   ./measure.sh --container <id> up_proj
#   BIN=gemv_fp16 SRC=gemv_fp16.cu TAG=iter03 ./measure.sh
#
# Requires a running container with nvcc + ncu (see start.sh). Any GPU is allowed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
KERNELS="$(cd "$SCRIPT_DIR/.." && pwd)"
KERNELS_C="${KERNELS_CONTAINER:-$KERNELS}"
SRC="${SRC:-gemv_fp16.cu}"
BIN="${BIN:-${SRC%.cu}}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
SHAPE_ARG="both"
NCU_LOCK="${NCU_LOCK:-$SCRIPT_DIR/.ncu.lock}"

opt_loop_load_container "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container|-c)
      CONTAINER="$2"
      shift 2
      ;;
    both|up_proj|down_proj)
      SHAPE_ARG="$1"
      shift
      ;;
    --help|-h)
      echo "Usage: CONTAINER=<id> $0 [up_proj|down_proj|both]"
      echo "       $0 --container <id> [up_proj|down_proj|both]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

opt_loop_resolve_container

GPU_LINE="$(opt_loop_gpu_in_container)"
GPU_NAME="$(echo "$GPU_LINE" | awk -F', ' '{print $1}')"
ARCH="$(opt_loop_nvcc_arch)"
echo "GPU=$GPU_LINE  nvcc_arch=$ARCH  container=${CONTAINER_ID:0:12}"

METRICS="dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__sass_thread_inst_executed_op_ffma_pred_on.sum"

OUT_DIR="$SCRIPT_DIR/runs/$TAG"
mkdir -p "$OUT_DIR" "$(dirname "$NCU_LOCK")"

echo "[lock] waiting for GPU ($NCU_LOCK) — one ncu at a time"
exec 9>"$NCU_LOCK"
flock 9

UNIQUE_BYTES="${UNIQUE_BYTES:-}"
if [[ -z "$UNIQUE_BYTES" && "$SRC" == *fp8* ]]; then
  UNIQUE_BYTES=25165824
fi

docker exec -w "$KERNELS_C" "$CONTAINER" \
  bash -c "nvcc -O3 -arch=$ARCH -lineinfo '$SRC' -o '$BIN'"

run_shape() {
  local shape="$1"
  local csv="$OUT_DIR/${shape}_ncu.csv"
  local json="$OUT_DIR/${shape}.json"
  echo "=== ncu $shape (cold, launch-count 1) ==="
  # Do not use --kernel-name gemv — some ncu versions treat it as a regex
  # that matches nothing. once() launches the GEMV as the first kernel.
  docker exec -w "$KERNELS_C" "$CONTAINER" \
    ncu --csv --launch-skip 0 --launch-count 1 --metrics "$METRICS" \
        "./$BIN" once "$shape" \
    | tee "$csv"
  if [[ -n "$UNIQUE_BYTES" ]]; then
    python3 "$SCRIPT_DIR/parse_ncu.py" "$csv" --shape "$shape" --unique-bytes "$UNIQUE_BYTES" --out "$json"
  else
    python3 "$SCRIPT_DIR/parse_ncu.py" "$csv" --shape "$shape" --out "$json"
  fi
}

if [[ "$SHAPE_ARG" == "both" ]]; then
  run_shape up_proj
  run_shape down_proj
  MEASURE_GPU="$GPU_NAME" MEASURE_ARCH="$ARCH" MEASURE_CONTAINER="${CONTAINER_ID:0:12}" \
  python3 - "$OUT_DIR" <<'PY'
import json, os, sys
from pathlib import Path
d = Path(sys.argv[1])
up = json.loads((d / "up_proj.json").read_text())
dn = json.loads((d / "down_proj.json").read_text())
gains = [up["gain_vs_trt_pct"], dn["gain_vs_trt_pct"]]
summary = {
    "tag": d.name,
    "gpu": os.environ.get("MEASURE_GPU"),
    "nvcc_arch": os.environ.get("MEASURE_ARCH"),
    "container": os.environ.get("MEASURE_CONTAINER"),
    "up_proj": up,
    "down_proj": dn,
    "min_gain_vs_trt_pct": min(gains),
    "hit_25pct_both": bool(up["hit_25pct_vs_trt"] and dn["hit_25pct_vs_trt"]),
}
(d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if summary["hit_25pct_both"]:
    print("TARGET HIT: both shapes >= 25% vs TRT XMMA")
else:
    print(f"TARGET MISS: min gain vs TRT = {summary['min_gain_vs_trt_pct']}%")
PY
else
  run_shape "$SHAPE_ARG"
fi

echo "Wrote $OUT_DIR"
