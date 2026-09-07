#!/bin/bash
# Bind the workflow to the profiler environment (nvcc + ncu already set up).
# Inside the container: CONTAINER=local  (or CONTAINER=$(hostname))
# From the host:        CONTAINER=<docker id or name>
#
# Usage:
#   CONTAINER=local ./start.sh
#   ./start.sh <container_id_or_name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: CONTAINER=local $0          # already inside the profiler env"
  echo "       $0 <container_id_or_name>  # from the host (docker exec)"
  echo "       $0 --container <id>"
  exit 0
fi

if [[ "${1:-}" == "--container" || "${1:-}" == "-c" ]]; then
  CONTAINER="$2"
elif [[ $# -ge 1 && "$1" != --* ]]; then
  CONTAINER="$1"
fi
opt_loop_load_container "$SCRIPT_DIR"
opt_loop_resolve_container

GPU_LINE="$(opt_loop_gpu_in_container)"
ARCH="$(opt_loop_nvcc_arch)"
opt_loop_save_container_env "$SCRIPT_DIR"

python3 - "$SCRIPT_DIR" "$CONTAINER" "$CONTAINER_ID" "$CONTAINER_NAME" "$GPU_LINE" "$ARCH" "$OPT_LOOP_EXEC" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
d, cid, full, name, gpu, arch, exe = sys.argv[1:8]
state = {
    "state": "running",
    "container": cid,
    "container_id": full,
    "container_name": name,
    "gpu": gpu,
    "nvcc_arch": arch,
    "exec": exe,
    "started_at": datetime.now(timezone.utc).isoformat(),
}
Path(d, "loop_state.json").write_text(json.dumps(state, indent=2) + "\n")
print(json.dumps(state, indent=2))
PY

cat <<EOF

Workflow bound: exec=$OPT_LOOP_EXEC id=${CONTAINER_ID:0:12} name=$CONTAINER_NAME
GPU: $GPU_LINE
nvcc arch: $ARCH
Wrote $SCRIPT_DIR/container.env and loop_state.json

--- start (Cursor chat) ---
CONTAINER=${CONTAINER} Follow kernels/opt_loop/TICK.md

--- track ---
$SCRIPT_DIR/status.sh

--- stop ---
$SCRIPT_DIR/stop.sh
EOF
