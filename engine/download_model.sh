#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/onnx-fp16}"

HF_REPO="${HF_REPO:-onnx-community/Llama-3.2-3B-Instruct-ONNX}"

# model_fp16's external weights are split across multiple *.onnx_data_N shards
# (each ~2 GB). All shards must be present or TensorRT fails mid-parse with
# "Failed to open file: ...onnx_data_N".
FILES=(
  "onnx/model_fp16.onnx"
  "onnx/model_fp16.onnx_data"
  "onnx/model_fp16.onnx_data_1"
  "onnx/model_fp16.onnx_data_2"
  "onnx/model_fp16.onnx_data_3"
  "tokenizer.json"
  "tokenizer_config.json"
  "config.json"
  "generation_config.json"
)

echo "=== Downloading $HF_REPO ==="
echo "Target: $MODEL_DIR"
mkdir -p "$MODEL_DIR/onnx"

download_hf() {
  # huggingface_hub >= 0.24 replaced `huggingface-cli` with `hf`; the old
  # command still exists on PATH but just prints a deprecation notice and
  # exits without downloading anything. Prefer `hf` when available.
  if command -v hf &>/dev/null; then
    hf download "$HF_REPO" \
      --include "${FILES[@]}" \
      --local-dir "$MODEL_DIR"
    return
  fi

  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$HF_REPO" \
      --include "${FILES[@]}" \
      --local-dir "$MODEL_DIR" \
      --local-dir-use-symlinks False
    return
  fi

  python3 << PYEOF
from huggingface_hub import hf_hub_download

repo = "${HF_REPO}"
model_dir = "${MODEL_DIR}"
files = """${FILES[*]}""".split()

for f in files:
    print(f"Downloading {f}...")
    hf_hub_download(repo_id=repo, filename=f, local_dir=model_dir)
PYEOF
}

download_hf

ONNX="$MODEL_DIR/onnx/model_fp16.onnx"
ONNX_DATA_SHARDS=(
  "$MODEL_DIR/onnx/model_fp16.onnx_data"
  "$MODEL_DIR/onnx/model_fp16.onnx_data_1"
  "$MODEL_DIR/onnx/model_fp16.onnx_data_2"
  "$MODEL_DIR/onnx/model_fp16.onnx_data_3"
)

if [[ ! -f "$ONNX" ]]; then
  echo "Error: missing ONNX graph file: $ONNX" >&2
  exit 1
fi

TOTAL_BYTES=0
for shard in "${ONNX_DATA_SHARDS[@]}"; do
  if [[ ! -f "$shard" ]]; then
    echo "Error: missing external weights shard: $shard" >&2
    echo "       TensorRT will fail mid-parse without every shard present." >&2
    exit 1
  fi
  size=$(stat -c%s "$shard")
  TOTAL_BYTES=$((TOTAL_BYTES + size))
done

# Full FP16 3B model external weights are ~6.4 GB across 4 shards (the last
# shard is a small remainder, not a full ~2 GB chunk like the first three).
MIN_TOTAL_BYTES=$((6 * 1024 * 1024 * 1024))
if (( TOTAL_BYTES < MIN_TOTAL_BYTES )); then
  echo "Error: onnx_data shards look incomplete (total ${TOTAL_BYTES} bytes). Re-run this script." >&2
  exit 1
fi

echo ""
echo "[ok] Download complete"
echo "ONNX graph:   $ONNX ($(numfmt --to=iec-i --suffix=B "$(stat -c%s "$ONNX")"))"
for shard in "${ONNX_DATA_SHARDS[@]}"; do
  echo "ONNX weights: $shard ($(numfmt --to=iec-i --suffix=B "$(stat -c%s "$shard")"))"
done
echo "Total weights: $(numfmt --to=iec-i --suffix=B "$TOTAL_BYTES")"
echo ""
echo "Next: python3 engine/inspect_onnx.py"
