#!/usr/bin/env bash
# Tear the demo stack down.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTARI_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$OTARI_ROOT"
# --profile web-search ensures the searxng container (opt-in profile) is
# included in `down` so it doesn't survive a teardown of the rest of the
# stack. --env-file is conditional — tear-down doesn't need any env vars,
# so missing .env (e.g. fresh checkout that never ran start.sh) isn't a
# blocker.
if [[ -f "$HERE/.env" ]]; then
  exec docker compose --env-file "$HERE/.env" --profile web-search down "$@"
else
  exec docker compose --profile web-search down "$@"
fi
