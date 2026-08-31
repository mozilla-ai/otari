#!/usr/bin/env bash
# Bring up the OSS gateway demo (gateway + searxng + postgres) with the keys
# and ports configured in this folder's .env. Loads .env via docker-compose's
# --env-file so the API keys never need to live in shell history.
#
#   ./start.sh                # SearXNG backend (free metasearch, can be flaky)
#   ./start.sh --brave        # Brave Search API backend (needs BRAVE_API_KEY)
#   ./start.sh --brave -d     # extra flags pass through to `docker compose up`

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

# --brave points the gateway at the Brave Search API, which it calls itself, so
# no search container comes up at all. Everything else passes through to
# `docker compose up`.
PROFILE="web-search"
PASSTHRU=()
for arg in "$@"; do
  case "$arg" in
    --brave)
      if ! grep -qE '^BRAVE_API_KEY=.+' "$ENV_FILE"; then
        echo "--brave needs an uncommented BRAVE_API_KEY in $ENV_FILE (key: https://brave.com/search/api/)." >&2
        exit 1
      fi
      PROFILE=""
      OTARI_WEB_SEARCH_PROVIDER_API_KEY=$(grep -E '^BRAVE_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2-)
      export OTARI_WEB_SEARCH_PROVIDER=brave OTARI_WEB_SEARCH_PROVIDER_API_KEY
      echo "ℹ --brave: web_search backed by the Brave Search API, called by the gateway itself"
      ;;
    *) PASSTHRU+=("$arg") ;;
  esac
done

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

# The profile opts in the SearXNG container. --brave clears it, because the
# gateway calls the Brave API itself and there is nothing to bring up. gateway +
# postgres have no profile, so they always come up.
PROFILE_ARGS=()
if [[ -n "$PROFILE" ]]; then
  PROFILE_ARGS=(--profile "$PROFILE")
fi
exec docker compose --env-file "$ENV_FILE" ${PROFILE_ARGS[@]+"${PROFILE_ARGS[@]}"} up ${PASSTHRU[@]+"${PASSTHRU[@]}"}
