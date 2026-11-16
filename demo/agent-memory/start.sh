#!/usr/bin/env bash
# Starts the agent memory demo: Postgres, the guardrails service, then the gateway from this checkout in the
# foreground (http://127.0.0.1:8010). Ctrl-C stops the gateway; ./stop.sh stops the containers too.
#
# --split also starts the memory service (`otari memory-service`, http://127.0.0.1:8011) and points the gateway
# at it: recall ranks and the consolidator runs there, not in the gateway. Its log is memory-service.log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
[ -f "$HERE/.env" ] || { echo "missing $HERE/.env: run ./setup.sh first" >&2; exit 1; }
set -a; . "$HERE/.env"; set +a

docker start otari-memory-pg >/dev/null

# The guardrails service (docker-compose.yml here): PIGuard in an encoderfile container, whose image is
# published per architecture.
case "$(uname -m)" in
  arm64|aarch64) triple="aarch64-linux-gnu" ;;
  x86_64|amd64) triple="x86_64-linux-gnu" ;;
  *) echo "unsupported architecture $(uname -m) for the encoderfile image" >&2; exit 1 ;;
esac
export OTARI_ENCODERFILE_IMAGE="docker.io/mzdotai/piguard.${triple}-encoderfile:v0.6.2"
docker compose -f "$HERE/docker-compose.yml" -p otari-memory-guardrails up -d

# The gateway runs from this directory, so the files it stores land in ./otari-files, which is gitignored.
cd "$HERE"
if [ "${1:-}" != "--split" ]; then
  exec uv run --project "$ROOT" --frozen otari serve -c "$HERE/otari-config.yml"
fi

if [ -z "${OTARI_MEMORY_SERVICE_TOKEN:-}" ]; then
  OTARI_MEMORY_SERVICE_TOKEN="$(openssl rand -hex 32)"
  printf 'OTARI_MEMORY_SERVICE_TOKEN=%s\n' "$OTARI_MEMORY_SERVICE_TOKEN" >>"$HERE/.env"
  export OTARI_MEMORY_SERVICE_TOKEN
fi
# The gateway migrates the database and the memory service never does, so the gateway starts first.
OTARI_MEMORY_SERVICE_URL="http://127.0.0.1:8011" \
  uv run --project "$ROOT" --frozen otari serve -c "$HERE/otari-config.yml" &
gateway=$!
trap 'kill "$gateway" "${memory:-}" 2>/dev/null; wait' EXIT INT TERM
curl -sf -o /dev/null --retry 60 --retry-connrefused --retry-delay 1 http://127.0.0.1:8010/api/v1/health
uv run --project "$ROOT" --frozen otari memory-service -c "$HERE/otari-config.yml" --port 8011 \
  >"$HERE/memory-service.log" 2>&1 &
memory=$!
curl -sf -o /dev/null --retry 60 --retry-connrefused --retry-delay 1 http://127.0.0.1:8011/api/v1/health
echo "memory service up on http://127.0.0.1:8011 (log: memory-service.log)"
wait "$gateway"
