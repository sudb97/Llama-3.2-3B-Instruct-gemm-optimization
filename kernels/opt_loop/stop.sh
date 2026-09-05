#!/bin/bash
# Stop the GEMV opt loop: kill the /loop ticker if any, mark state stopped.
# Subagents should not be dispatched after this.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "$SCRIPT_DIR/loop.pid" ]]; then
  pid="$(cat "$SCRIPT_DIR/loop.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    # process group in case the sleeper is a child
    kill -- -"$pid" 2>/dev/null || true
    echo "[ok] killed loop pid $pid"
  else
    echo "[..] loop pid $pid already dead"
  fi
  rm -f "$SCRIPT_DIR/loop.pid"
else
  echo "[..] no loop.pid"
fi

python3 - "$SCRIPT_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
d = Path(sys.argv[1])
p = d / "loop_state.json"
state = {}
if p.exists():
    state = json.loads(p.read_text())
state["state"] = "stopped"
state["stopped_at"] = datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(state, indent=2) + "\n")
print("state: stopped")
PY

echo "Coordinator will not re-arm. Confirm in Cursor: Stop the GEMV opt loop"
