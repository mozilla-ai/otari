#!/usr/bin/env bash
# Stops the demo's containers: Open WebUI, the guardrails service and Postgres. Their data stays in Docker
# volumes; ./start.sh brings them back. The gateway stops with Ctrl-C in its own terminal.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker rm -f openwebui-alice openwebui-bob >/dev/null 2>&1 || true
docker compose -f "$HERE/docker-compose.yml" -p otari-memory-guardrails down
docker stop otari-memory-pg >/dev/null 2>&1 || true
echo "stopped"
