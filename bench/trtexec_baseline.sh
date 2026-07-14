#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/engine/configs/baseline_decode.env}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/bench/results}"

MODE="${1:-decode}"

if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG"
elif [[ -f "$PROJECT_ROOT/engine/configs/baseline_decode.env.example" ]]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/engine/configs/baseline_decode.env.example"
fi

mkdir -p "$RESULTS_DIR"

case "$MODE" in
  decode)
    ENGINE="${ENGINE:-$PROJECT_ROOT/engines/baseline_decode.engine}"
    SHAPES="${OPT_SHAPES:-input_ids:1x1,attention_mask:1x1}"
    TAG="decode"
    ;;
  prefill)
    ENGINE="${PREFILL_ENGINE:-$PROJECT_ROOT/engines/baseline_prefill.engine}"
    SHAPES="${PREFILL_OPT_SHAPES:-input_ids:1x512,attention_mask:1x512}"
    TAG="prefill"
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

PROFILE_JSON="$RESULTS_DIR/${TAG}_profile.json"
TIMING_JSON="$RESULTS_DIR/${TAG}_timing.json"

echo "=== Benchmark ($MODE) ==="
echo "Engine: $ENGINE"
echo "Shapes: $SHAPES"
echo ""

# --separateProfileRun: --dumpProfile's per-layer instrumentation adds CUDA
# sync points that make end-to-end timing inaccurate, so trtexec silently
# drops --exportTimes output whenever --dumpProfile is active in the same
# run. This flag makes trtexec run the benchmark twice internally — once
# clean for --exportTimes, once instrumented for --dumpProfile/--exportProfile
# — so both files are produced correctly (at the cost of ~2x wall time here).
trtexec \
  --loadEngine="$ENGINE" \
  --shapes="$SHAPES" \
  --warmUp=500 \
  --duration=10 \
  --avgRuns=100 \
  --dumpProfile \
  --separateProfileRun \
  --exportProfile="$PROFILE_JSON" \
  --exportTimes="$TIMING_JSON"

echo ""
if [[ -f "$PROFILE_JSON" ]]; then
  echo "[ok] Profile: $PROFILE_JSON"
else
  echo "[!!] Profile NOT created: $PROFILE_JSON"
fi
if [[ -f "$TIMING_JSON" ]]; then
  echo "[ok] Timing:  $TIMING_JSON"
else
  echo "[!!] Timing NOT created: $TIMING_JSON"
fi
