#!/usr/bin/env bash
# Bring up the board-demo gateway stack (gateway + sandbox + searxng + postgres)
# with the keys and ports configured in this folder's .env.
#
# The board demo exercises BOTH built-in tools in a single question, so it
# opts in to both the `code-exec` (sandbox) and `web-search` (searxng) compose
# profiles — unlike the single-feature demos which each enable just one.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"

ENV_FILE="$HERE/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE — copy .env.example to .env (keys are optional for the local-only acts)." >&2
  exit 1
fi

if grep -qE '^[A-Z_]+=.*REPLACE_ME' "$ENV_FILE"; then
  echo "$ENV_FILE has an uncommented REPLACE_ME — fill in your real key or comment the line before starting." >&2
  exit 1
fi

cd "$GATEWAY_ROOT"

# If we're on a non-main branch, the published `mzdotai/otari:latest` may not
# have the unreleased code on this branch. Surface the manual-build recipe.
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [[ -n "$branch" && "$branch" != "main" ]]; then
  cat <<EOF
ℹ branch '$branch' detected — if it has unreleased gateway code, build the image locally first:
    docker build -t mzdotai/otari:latest .
  (skip this if you've already built, or if the branch's changes are already in mzdotai/otari:latest.)

EOF
fi

# Both profiles: sandbox (code_execution) AND searxng (web_search). The board
# demo's showcase question needs both tools in one round.
#
# The standalone override blanks any OTARI_AI_TOKEN from .env, so even when the
# token is present for Act 4 the gateway boots in STANDALONE mode here and the
# local providers in gateway-config.yml are honoured. (Platform mode forbids
# local providers, which is the error this avoids.)
# --force-recreate brings every service (notably searxng, which can get stuck
# in a stale/unhealthy state across runs) up fresh each start.
exec docker compose --env-file "$ENV_FILE" \
  -f docker-compose.yml -f "$HERE/docker-compose.standalone.yml" \
  --profile code-exec --profile web-search up --force-recreate "$@"
