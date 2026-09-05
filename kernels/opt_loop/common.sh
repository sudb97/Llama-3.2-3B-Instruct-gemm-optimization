# Shared CONTAINER + GPU helpers. Source from other opt_loop scripts.
# CONTAINER / CONTAINER_ID / --container: running docker id or name with nvcc+ncu.

opt_loop_load_container() {
  local dir="${1:-}"
  if [[ -z "${CONTAINER:-}" && -z "${CONTAINER_ID:-}" && -n "$dir" && -f "$dir/container.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    source "$dir/container.env"
    set +a
  fi
  CONTAINER="${CONTAINER:-${CONTAINER_ID:-${CONTAINER_NAME:-}}}"
}

opt_loop_resolve_container() {
  if [[ -z "${CONTAINER:-}" ]]; then
    echo "CONTAINER is required (docker id or name of the profiler environment)." >&2
    echo "  CONTAINER=<id> $0" >&2
    echo "  $0 --container <id>" >&2
    echo "  or: $(dirname "${BASH_SOURCE[0]}")/start.sh <id>" >&2
    return 1
  fi
  if ! command -v docker >/dev/null; then
    echo "docker not found on PATH" >&2
    return 1
  fi
  if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "No such container: $CONTAINER" >&2
    docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}' >&2 || true
    return 1
  fi
  local running
  running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER")"
  if [[ "$running" != "true" ]]; then
    echo "[..] starting $CONTAINER"
    docker start "$CONTAINER" >/dev/null
  fi
  CONTAINER_ID="$(docker inspect -f '{{.Id}}' "$CONTAINER")"
  CONTAINER_NAME="$(docker inspect -f '{{.Name}}' "$CONTAINER" | sed 's#^/##')"
  echo "[ok] container id=${CONTAINER_ID:0:12} name=$CONTAINER_NAME"
}

opt_loop_gpu_in_container() {
  docker exec "$CONTAINER" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 | tr -d '\r'
}

opt_loop_nvcc_arch() {
  local cc
  cc="$(docker exec "$CONTAINER" nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' \r')"
  echo "sm_${cc//./}"
}

opt_loop_save_container_env() {
  local dir="$1"
  cat >"$dir/container.env" <<EOF
CONTAINER=${CONTAINER}
CONTAINER_ID=${CONTAINER_ID}
CONTAINER_NAME=${CONTAINER_NAME}
EOF
}
