#!/usr/bin/env bash
# Tear the demo stack down.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$GATEWAY_ROOT"
# --profile code-exec ensures the sandbox container (opt-in profile) is
# included in `down` so it doesn't survive a teardown of the rest of the
# stack.
exec docker compose --env-file "$HERE/.env" --profile code-exec down "$@"
