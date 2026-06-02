#!/usr/bin/env bash
# Bring the board-demo gateway up in PLATFORM MODE for Act 4.
#
# Same stack as ./start.sh (gateway + sandbox + searxng + postgres), but with
# OTARI_AI_TOKEN set, which flips the gateway into platform mode: it delegates
# auth, credential resolution, budgets, and usage to Otari.ai per-request — so
# the SAME app can now reach managed providers like Mozilla.ai Inference (mzai)
# with no provider keys of its own.
#
# Prereq: OTARI_AI_TOKEN must be set in this folder's .env (mint a gateway
# token in the Otari.ai UI — see RUNBOOK.md, Act 4 setup).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

ENV_FILE="$HERE/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE — copy .env.example to .env first." >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a; source "$ENV_FILE"; set +a

if [[ -z "${OTARI_AI_TOKEN:-}" || "$OTARI_AI_TOKEN" == *REPLACE_ME* ]]; then
  cat <<EOF >&2
OTARI_AI_TOKEN is not set in $ENV_FILE.

Act 4 needs a gateway token from Otari.ai:
  1. Log in to Otari.ai.
  2. Create a gateway (or open an existing one) for this org.
  3. Mint a gateway token and copy it.
  4. Add it to $ENV_FILE:   OTARI_AI_TOKEN=otari_...

Then re-run ./start-platform.sh.
EOF
  exit 1
fi

cd "$GATEWAY_ROOT"

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [[ -n "$branch" && "$branch" != "main" ]]; then
  echo "ℹ branch '$branch' — if it has unreleased gateway code, build first: docker build -t mzdotai/otari:latest ." >&2
fi

echo "Starting gateway in PLATFORM MODE (token → Otari.ai)…"
# Platform mode forbids local provider credentials, so we mount the
# PROVIDERLESS config (gateway-config.platform.yml) instead of the standalone
# one. Exporting GATEWAY_CONFIG_PATH overrides whatever .env set, and compose
# reads it for the gateway's config volume mount.
#
# Both tool profiles, same as ./start.sh — the platform leg of Act 4 still
# exercises web_search + code_execution, now on a managed model.
export GATEWAY_CONFIG_PATH=./demo/board/gateway-config.platform.yml
# --force-recreate brings every service up fresh (esp. searxng, which can get
# stuck stale across runs) and ensures the gateway picks up platform mode.
exec docker compose --env-file "$ENV_FILE" \
  -f docker-compose.yml -f "$HERE/docker-compose.platform.yml" \
  --profile code-exec --profile web-search up --force-recreate "$@"
