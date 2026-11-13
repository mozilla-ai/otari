#!/usr/bin/env bash
# Starts the agent memory demo: Postgres, the guardrails service, then the gateway from this checkout in the
# foreground (http://127.0.0.1:8010). Ctrl-C stops the gateway; ./stop.sh stops the containers too.
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
exec uv run --project "$ROOT" --frozen otari serve -c "$HERE/otari-config.yml"
