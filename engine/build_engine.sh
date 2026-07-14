#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/baseline_decode.env}"

MODE="${1:-decode}"

if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
elif [[ -f "$SCRIPT_DIR/configs/baseline_decode.env.example" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/configs/baseline_decode.env.example"
else
  echo "Error: no config found. Copy engine/configs/baseline_decode.env.example to baseline_decode.env"
  exit 1
fi

mkdir -p "$(dirname "$TIMING_CACHE")" "$(dirname "$ENGINE")"

case "$MODE" in
  decode)
    OUT_ENGINE="$ENGINE"
    MIN="$MIN_SHAPES"
    OPT="$OPT_SHAPES"
    MAX="$MAX_SHAPES"
    ;;
  prefill)
    OUT_ENGINE="${PREFILL_ENGINE:-$PROJECT_ROOT/engines/baseline_prefill.engine}"
    MIN="${PREFILL_MIN_SHAPES:-$MIN_SHAPES}"
    OPT="${PREFILL_OPT_SHAPES:-$OPT_SHAPES}"
    MAX="${PREFILL_MAX_SHAPES:-$MAX_SHAPES}"
    ;;
  *)
    echo "Usage: $0 [decode|prefill]"
    exit 1
    ;;
esac

if [[ ! -f "$ONNX" ]]; then
  echo "Error: ONNX not found: $ONNX"
  echo "Run: ./engine/download_model.sh"
  exit 1
fi

# A 3B fp16 model always stores weights as external data next to the .onnx.
# TensorRT only errors when it reaches a tensor in a missing file, so check
# upfront that at least one external-data file exists.
ONNX_DIR="$(dirname "$ONNX")"
shopt -s nullglob
data_files=("$ONNX_DIR"/*.onnx_data* "$ONNX_DIR"/*.onnx.data)
shopt -u nullglob
if (( ${#data_files[@]} == 0 )); then
  echo "Error: no external weight files (*.onnx_data*) found next to $ONNX"
  echo "Run: ./engine/export_clean_onnx.sh (or ./engine/download_model.sh)"
  exit 1
fi

echo "=== Building TensorRT engine ($MODE) ==="
echo "ONNX:   $ONNX"
echo "Engine: $OUT_ENGINE"
echo "Shapes: min=[$MIN] opt=[$OPT] max=[$MAX]"
echo ""

# --fp16: the ONNX graph's weights/activations are already FP16, but that
# only sets tensor storage dtype. Without this builder flag TensorRT's
# tactic search is restricted to FP32 kernels, so the "baseline" tactic
# it picks for the target MLP GEMM would be an FP32 CUDA-core kernel
# instead of the Tensor-Core FP16 tactic this project needs to compare
# the custom CUTLASS kernel against.
trtexec \
  --onnx="$ONNX" \
  --saveEngine="$OUT_ENGINE" \
  --timingCacheFile="$TIMING_CACHE" \
  --minShapes="$MIN" \
  --optShapes="$OPT" \
  --maxShapes="$MAX" \
  --fp16 \
  --verbose

echo ""
echo "[ok] Engine saved to $OUT_ENGINE"
echo "Next: ./bench/trtexec_baseline.sh $MODE"
