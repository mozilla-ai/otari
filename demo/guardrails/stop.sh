#!/usr/bin/env bash
# Tear the guardrails demo stack down.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTARI_ROOT="$(cd "$HERE/../.." && pwd)"

cd "$OTARI_ROOT"
# The `guardrails` profile covers both anyguardrails and the encoderfile
# container, so a single profile tears down everything regardless of how the
# stack was started. --env-file is conditional — tear-down needs no env vars,
# so a missing .env (e.g. fresh checkout that never ran start.sh) isn't a blocker.
if [[ -f "$HERE/.env" ]]; then
  exec docker compose --env-file "$HERE/.env" --profile guardrails down "$@"
else
  exec docker compose --profile guardrails down "$@"
fi
