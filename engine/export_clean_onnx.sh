#!/bin/bash
set -euo pipefail

# Export a CLEAN standard-ops ONNX graph of Llama-3.2-3B-Instruct for TensorRT.
#
# Why: the pre-exported onnx-community/Llama-3.2-3B-Instruct-ONNX repo is
# optimized for ONNX Runtime — every variant (fp16, fp32, quantized) uses
# com.microsoft contrib ops (GroupQueryAttention, SkipSimplifiedLayerNormalization)
# that TensorRT's parser cannot import ("Plugin not found" for each node).
# optimum's exporter emits only standard ONNX ops: attention is decomposed and
# every projection (q/k/v/o, gate/up/down) is a plain MatMul — exactly what we
# want for GEMM tactic profiling.
#
# Run INSIDE the container (needs GPU for fp16 export; L4 24GB is plenty).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# meta-llama/Llama-3.2-3B-Instruct is license-gated on HF; unsloth mirrors the
# exact same weights ungated. Override with MODEL_ID=... if you have a token.
MODEL_ID="${MODEL_ID:-unsloth/Llama-3.2-3B-Instruct}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/models/onnx-fp16-clean}"

# Since optimum v2.0 the ONNX exporter lives in the separate optimum-onnx
# package; without it `optimum-cli export onnx` is an unrecognized subcommand.
if ! optimum-cli export onnx --help >/dev/null 2>&1; then
  echo "=== Installing optimum-onnx (ONNX exporter) ==="
  pip3 install --break-system-packages "optimum-onnx[onnxruntime]"
fi

echo "=== Exporting $MODEL_ID -> $OUT_DIR (fp16, with KV cache) ==="
# --task text-generation-with-past: exports with past_key_values inputs /
# present outputs (KV cache), same interface as the previous model.
# --dtype fp16 --device cuda: fp16 export must trace on GPU.
# --no-post-process: post-processing dedups the tied lm_head/embed_tokens
# weight by serializing the whole model as one protobuf, which fails for
# >2 GB models ("Failed to serialize proto"). Skipping it just stores the
# tied weight twice (~790 MB extra on disk); numerically identical.
optimum-cli export onnx \
  --model "$MODEL_ID" \
  --task text-generation-with-past \
  --dtype fp16 \
  --device cuda \
  --no-post-process \
  "$OUT_DIR"

echo ""
echo "=== Export complete ==="
ls -lh "$OUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify inputs (optimum usually adds a position_ids input):"
echo "       python3 engine/inspect_onnx.py --onnx $OUT_DIR/model.onnx"
echo "  2. Build:  ./engine/build_engine.sh decode"
