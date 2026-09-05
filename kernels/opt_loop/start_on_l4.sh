#!/bin/bash
# Backward-compatible wrapper. L4 is not required.
# Usage: CONTAINER=<id> ./start_on_l4.sh
#    or: ./start_on_l4.sh <container_id_or_name>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start.sh" "$@"
