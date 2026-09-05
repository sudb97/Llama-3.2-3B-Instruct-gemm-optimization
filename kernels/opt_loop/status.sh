#!/bin/bash
# Print container, GPU, latest verdict, latest run, loop PID.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
opt_loop_load_container "$SCRIPT_DIR"

echo "=== GEMV opt loop status ==="
if [[ -f "$SCRIPT_DIR/loop_state.json" ]]; then
  python3 - "$SCRIPT_DIR/loop_state.json" <<'PY'
import json, sys
s = json.loads(open(sys.argv[1]).read())
print(f"state:     {s.get('state')}")
print(f"started:   {s.get('started_at')}")
print(f"container: {s.get('container')}  name={s.get('container_name')}")
print(f"gpu:       {s.get('gpu')}  arch={s.get('nvcc_arch')}")
if s.get("stopped_at"):
    print(f"stopped:   {s['stopped_at']}")
PY
else
  echo "state:     not started (no loop_state.json)"
fi

if [[ -n "${CONTAINER:-}" ]] && command -v docker >/dev/null && docker inspect "$CONTAINER" >/dev/null 2>&1; then
  running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER")"
  echo "docker:    $CONTAINER  running=$running"
  if [[ "$running" == "true" ]]; then
    echo "live GPU:  $(opt_loop_gpu_in_container)"
  fi
elif [[ -n "${CONTAINER:-}" ]]; then
  echo "docker:    CONTAINER=$CONTAINER (not inspectable)"
else
  echo "docker:    CONTAINER unset"
fi

if [[ -f "$SCRIPT_DIR/loop.pid" ]]; then
  pid="$(cat "$SCRIPT_DIR/loop.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "loop pid:  $pid (alive)"
  else
    echo "loop pid:  $pid (dead)"
  fi
else
  echo "loop pid:  none"
fi

echo ""
echo "--- VERDICT.md (head) ---"
if [[ -f "$SCRIPT_DIR/VERDICT.md" ]]; then
  head -n 20 "$SCRIPT_DIR/VERDICT.md"
else
  echo "(missing)"
fi

echo ""
echo "--- latest runs/*/summary.json ---"
latest="$(ls -1dt "$SCRIPT_DIR"/runs/*/summary.json 2>/dev/null | head -1 || true)"
if [[ -n "$latest" ]]; then
  echo "$latest"
  python3 - "$latest" <<'PY'
import json, sys
s = json.loads(open(sys.argv[1]).read())
print(f"tag={s.get('tag')} official={s.get('official', True)} min_gain={s.get('min_gain_vs_trt_pct')} hit_25={s.get('hit_25pct_both')}")
PY
else
  echo "(none)"
fi

echo ""
echo "--- EXPERIMENT_LOG.md (tail) ---"
if [[ -f "$SCRIPT_DIR/EXPERIMENT_LOG.md" ]]; then
  tail -n 12 "$SCRIPT_DIR/EXPERIMENT_LOG.md"
fi
