#!/usr/bin/env bash
# Start the standalone routing gateway demo stack.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GATEWAY_ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$HERE/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE - copy .env.example to .env first." >&2
  exit 1
fi

if grep -Eq '^[[:space:]]*(OPENAI_API_KEY|ANTHROPIC_API_KEY)=.*REPLACE_ME' "$ENV_FILE"; then
  echo "replace placeholder provider keys in $ENV_FILE or comment them out." >&2
  exit 1
fi

cd "$GATEWAY_ROOT"

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [[ -n "$branch" && "$branch" != "main" ]]; then
  cat <<EOF
branch '$branch' detected. If this branch has unreleased gateway code, build locally first:
    docker build -t mzdotai/otari:latest .

EOF
fi

exec docker compose --env-file "$ENV_FILE" up "$@" gateway postgres
