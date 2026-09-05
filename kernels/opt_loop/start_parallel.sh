#!/bin/bash
# Create three isolated git worktrees so three Cursor agents can edit kernels
# in parallel. They share one GPU: measure.sh flocks .ncu.lock so ncu is serial.
#
# Usage (from the main repo, after CONTAINER is bound via start.sh):
#   CONTAINER=<id> bash kernels/opt_loop/start_parallel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
BASE="$PARENT/opt-worktrees"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a git repo: $ROOT"
  exit 1
fi

mkdir -p "$BASE"

copy_wip() {
  local dest="$1"
  mkdir -p "$dest/kernels/opt_loop/prompts" "$dest/.cursor/skills"
  rsync -a --exclude runs --exclude '*.lock' \
    "$ROOT/kernels/opt_loop/" "$dest/kernels/opt_loop/"
  if [[ -d "$ROOT/.cursor/skills" ]]; then
    rsync -a "$ROOT/.cursor/skills/" "$dest/.cursor/skills/"
  fi
}

add_tree() {
  local name="$1"
  local branch="opt/$name"
  local dest="$BASE/$name"
  if [[ -d "$dest/.git" || -f "$dest/.git" ]]; then
    echo "[ok] worktree exists: $dest ($branch)"
  else
    if git show-ref --verify --quiet "refs/heads/$branch"; then
      git worktree add "$dest" "$branch"
    else
      git worktree add -b "$branch" "$dest" HEAD
    fi
    echo "[ok] created $dest on $branch"
  fi
  copy_wip "$dest"
}

add_tree dram
add_tree amp
add_tree fp8

cat <<EOF

Three agent worktrees are ready. Open THREE Cursor agents (or three chats)
with these folders as the workspace root:

  1. DRAM   $BASE/dram
     paste: $(realpath "$BASE/dram/kernels/opt_loop/prompts/agent_dram.md" 2>/dev/null || echo "$BASE/dram/kernels/opt_loop/prompts/agent_dram.md")

  2. AMP    $BASE/amp
     paste: $BASE/amp/kernels/opt_loop/prompts/agent_amp.md

  3. FP8    $BASE/fp8
     paste: $BASE/fp8/kernels/opt_loop/prompts/agent_fp8.md

Start command for each agent (copy the matching prompt file as the first message):

  Follow kernels/opt_loop/prompts/agent_<dram|amp|fp8>.md and kernels/opt_loop/WORKFLOW.md. Begin iteration 1.

Bind the profiler container (id or name; any GPU):

  CONTAINER=<id> $ROOT/kernels/opt_loop/start.sh

GPU lock: $ROOT/kernels/opt_loop/.ncu.lock
ncu is one-at-a-time. Kernel edits can proceed in parallel.

EOF
