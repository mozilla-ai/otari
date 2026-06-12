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

# --brave swaps the SearXNG metasearch backend for the Brave Search adapter
# (scripts/web-search-brave-adapter/). Everything else passes through to
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
      PROFILE="web-search-brave"
      export OTARI_WEB_SEARCH_URL=http://brave-adapter:8080
      echo "ℹ --brave: web_search backed by the Brave adapter (OTARI_WEB_SEARCH_URL → brave-adapter:8080)"
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

# The chosen profile opts in the right web-search backend container
# (searxng by default, brave-adapter with --brave). gateway + postgres have
# no profile, so they always come up.
exec docker compose --env-file "$ENV_FILE" --profile "$PROFILE" up ${PASSTHRU[@]+"${PASSTHRU[@]}"}
