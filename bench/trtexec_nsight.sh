#!/bin/bash
# Short trtexec run intended to be the application under nsys/ncu.
# No --dumpProfile/--exportTimes — those add sync overhead and are not needed
# when Nsight is capturing the CUDA timeline / kernel metrics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/engine/configs/baseline_decode.env}"

MODE="${1:-decode}"

if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
elif [[ -f "$PROJECT_ROOT/engine/configs/baseline_decode.env.example" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/engine/configs/baseline_decode.env.example"
fi

case "$MODE" in
  decode)
    ENGINE="${ENGINE:-$PROJECT_ROOT/engines/baseline_decode.engine}"
    SHAPES="${OPT_SHAPES:-input_ids:1x1,attention_mask:1x1}"
    ;;
  prefill)
    ENGINE="${PREFILL_ENGINE:-$PROJECT_ROOT/engines/baseline_prefill.engine}"
    SHAPES="${PREFILL_OPT_SHAPES:-input_ids:1x512,attention_mask:1x512}"
    ;;
  *)
    echo "Usage: $0 [decode|prefill]"
    exit 1
    ;;
esac

if [[ ! -f "$ENGINE" ]]; then
  echo "Error: engine not found: $ENGINE"
  echo "Run: ./engine/build_engine.sh $MODE"
  exit 1
fi

echo "=== Nsight target run ($MODE) ==="
echo "Engine: $ENGINE"
echo "Shapes: $SHAPES"
echo ""

trtexec \
  --loadEngine="$ENGINE" \
  --shapes="$SHAPES" \
  --warmUp=500 \
  --duration=2 \
  --avgRuns=100
