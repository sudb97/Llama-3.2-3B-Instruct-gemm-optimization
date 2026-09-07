# Shared CONTAINER + GPU helpers. Source from other opt_loop scripts.
# CONTAINER: docker id/name, or local/here/self when already inside the profiler env.

OPT_LOOP_EXEC="${OPT_LOOP_EXEC:-}"

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

opt_loop_hostname() {
  hostname -s 2>/dev/null || hostname | cut -d. -f1
}

opt_loop_has_profiler() {
  command -v nvidia-smi >/dev/null && command -v nvcc >/dev/null && command -v ncu >/dev/null
}

# True if CONTAINER means "this process's environment".
opt_loop_is_self_id() {
  local want="${1:-}"
  local host
  host="$(opt_loop_hostname)"
  case "$want" in
    local|here|self|.|this) return 0 ;;
  esac
  [[ -z "$want" || -z "$host" ]] && return 1
  # hostname is usually the short container id (e.g. 8c24291250f3)
  [[ "$want" == "$host" || "$want" == "$host"* || "$host" == "$want"* ]]
}

opt_loop_use_local() {
  OPT_LOOP_EXEC=local
  CONTAINER_NAME="${CONTAINER_NAME:-local}"
  CONTAINER_ID="${CONTAINER_ID:-$(opt_loop_hostname)}"
  if [[ -z "${CONTAINER:-}" || "$CONTAINER" == local || "$CONTAINER" == here || "$CONTAINER" == self || "$CONTAINER" == . || "$CONTAINER" == this ]]; then
    CONTAINER="${CONTAINER_ID}"
  fi
  if ! opt_loop_has_profiler; then
    echo "This environment is missing nvidia-smi, nvcc, or ncu." >&2
    echo "Need a profiler container (or host) with CUDA + Nsight Compute." >&2
    return 1
  fi
  echo "[ok] in-container mode (no docker exec) id=${CONTAINER_ID} host=$(opt_loop_hostname)"
}

opt_loop_use_docker() {
  OPT_LOOP_EXEC=docker
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
  echo "[ok] docker exec id=${CONTAINER_ID:0:12} name=$CONTAINER_NAME"
}

opt_loop_resolve_container() {
  if [[ -z "${CONTAINER:-}" ]]; then
    echo "CONTAINER is required (docker id/name, or local if already inside the profiler env)." >&2
    echo "  CONTAINER=local $0          # already inside the container" >&2
    echo "  CONTAINER=\$(hostname) $0    # same, using this container's id" >&2
    echo "  CONTAINER=<id> $0            # from the host, docker exec into <id>" >&2
    echo "  $0 --container <id>" >&2
    return 1
  fi

  if [[ "${OPT_LOOP_EXEC:-}" == local ]] || opt_loop_is_self_id "$CONTAINER"; then
    opt_loop_use_local
    return
  fi

  if command -v docker >/dev/null; then
    opt_loop_use_docker
    return
  fi

  if opt_loop_has_profiler; then
    echo "[..] docker not on PATH; CONTAINER=$CONTAINER is not this hostname ($(opt_loop_hostname))" >&2
    echo "    If you are already inside the profiler container, run:" >&2
    echo "      CONTAINER=local $0" >&2
    echo "      CONTAINER=$(opt_loop_hostname) $0" >&2
    return 1
  fi

  echo "docker not found on PATH, and this environment has no nvcc/ncu." >&2
  return 1
}

opt_loop_gpu_query() {
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 | tr -d '\r'
}

opt_loop_gpu_in_container() {
  if [[ "${OPT_LOOP_EXEC}" == local ]]; then
    opt_loop_gpu_query
  else
    docker exec "$CONTAINER" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1 | tr -d '\r'
  fi
}

opt_loop_nvcc_arch() {
  local cc
  if [[ "${OPT_LOOP_EXEC}" == local ]]; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' \r')"
  else
    cc="$(docker exec "$CONTAINER" nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' \r')"
  fi
  echo "sm_${cc//./}"
}

# Run a command in the profiler env (cwd = first arg).
opt_loop_run() {
  local workdir="$1"
  shift
  if [[ "${OPT_LOOP_EXEC}" == local ]]; then
    (cd "$workdir" && "$@")
  else
    docker exec -w "$workdir" "$CONTAINER" "$@"
  fi
}

opt_loop_save_container_env() {
  local dir="$1"
  cat >"$dir/container.env" <<EOF
CONTAINER=${CONTAINER}
CONTAINER_ID=${CONTAINER_ID}
CONTAINER_NAME=${CONTAINER_NAME}
OPT_LOOP_EXEC=${OPT_LOOP_EXEC}
EOF
}
