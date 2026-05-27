#!/usr/bin/env bash
# Stop the standalone routing gateway demo stack.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$GATEWAY_ROOT"
if [[ -f "$HERE/.env" ]]; then
  exec docker compose --env-file "$HERE/.env" down "$@"
else
  exec docker compose down "$@"
fi
