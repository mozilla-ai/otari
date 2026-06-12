#!/usr/bin/env bash
# Bring up the OSS gateway demo (gateway + sandbox + postgres) with the keys
# and ports configured in this folder's .env. Loads .env via docker-compose's
# --env-file so the API keys never need to live in shell history.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OTARI_ROOT="$(cd "$HERE/../.." && pwd)"

ENV_FILE="$HERE/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE — copy .env.example to .env and fill in your keys." >&2
  exit 1
fi

if grep -qE '^[A-Z_]+=.*REPLACE_ME' "$ENV_FILE"; then
  echo "$ENV_FILE has an uncommented REPLACE_ME — fill in your real key before starting." >&2
  exit 1
fi

cd "$OTARI_ROOT"

# If we're on a non-main branch, the published `mzdotai/otari:latest` may not
# have the unreleased code on this branch. Surface the manual-build recipe
# so contributors don't get a silent ModuleNotFoundError at runtime.
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [[ -n "$branch" && "$branch" != "main" ]]; then
  cat <<EOF
ℹ branch '$branch' detected — if it has unreleased gateway code, build the image locally first:
    docker build -t mzdotai/otari:latest .
  (skip this if you've already built, or if the branch's changes are already in mzdotai/otari:latest.)

EOF
fi

# --profile code-exec opts the sandbox container in (gateway's compose
# leaves it opt-in so operators who don't run code_execution aren't forced
# to pull the image). The demo needs it, so request it here.
exec docker compose --env-file "$ENV_FILE" --profile code-exec up "$@"
