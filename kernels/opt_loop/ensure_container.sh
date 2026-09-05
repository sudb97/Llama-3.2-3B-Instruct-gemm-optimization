#!/bin/bash
# Resolve CONTAINER (id or name). Does not create a new image/container.
# The environment must already be set up inside that container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
opt_loop_load_container "$SCRIPT_DIR"
opt_loop_resolve_container
