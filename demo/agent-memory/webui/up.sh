#!/usr/bin/env bash
# Two Open WebUI instances, one per demo user, both talking to the local Otari gateway with that user's key.
#   up.sh        start them (Alice on http://127.0.0.1:3001, Bob on http://127.0.0.1:3002)
#   up.sh down   stop and remove them; their chats are kept in Docker volumes
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../keys.env"
IMAGE=ghcr.io/open-webui/open-webui:main
MODEL=openai/gpt-oss-120b

start() { # name port key-variable
  local name="$1" port="$2" key_var="$3" envfile="$HERE/.$1.env"
  # The key goes through a file only this user reads, never the command line.
  umask 077
  {
    echo "OPENAI_API_BASE_URL=http://host.docker.internal:8010/api/v1"
    echo "OPENAI_API_KEY=${!key_var}"
    echo "ENABLE_OLLAMA_API=False"
    echo "WEBUI_AUTH=False"
    echo "WEBUI_NAME=Team assistant ($name)"
    echo "DEFAULT_MODELS=$MODEL"
    # Open WebUI's own side requests (titles, tags, follow-ups) would reach memory as turns of their own.
    echo "ENABLE_TITLE_GENERATION=False"
    echo "ENABLE_TAGS_GENERATION=False"
    echo "ENABLE_FOLLOW_UP_GENERATION=False"
    echo "ENABLE_AUTOCOMPLETE_GENERATION=False"
    echo "ENABLE_RETRIEVAL_QUERY_GENERATION=False"
    echo "ENABLE_SEARCH_QUERY_GENERATION=False"
  } > "$envfile"
  docker rm -f "openwebui-$name" >/dev/null 2>&1 || true
  docker run -d --name "openwebui-$name" -p "127.0.0.1:$port:8080" --env-file "$envfile" \
    -v "openwebui-$name:/app/backend/data" "$IMAGE" >/dev/null
  echo "$name: http://127.0.0.1:$port"
}

if [ "${1:-}" = "down" ]; then
  docker rm -f openwebui-alice openwebui-bob >/dev/null 2>&1 || true
  echo "stopped"
  exit 0
fi
start alice 3001 ALICE_KEY
start bob 3002 BOB_KEY
