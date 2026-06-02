#!/usr/bin/env bash
# Tear the board-demo stack down. Includes both opt-in profiles so the sandbox
# and searxng containers don't survive teardown of the rest of the stack.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$GATEWAY_ROOT"
if [[ -f "$HERE/.env" ]]; then
  exec docker compose --env-file "$HERE/.env" \
    --profile code-exec --profile web-search down "$@"
else
  exec docker compose --profile code-exec --profile web-search down "$@"
fi
