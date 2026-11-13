#!/usr/bin/env bash
# One-time setup for the agent memory demo. Safe to run again: it only does what is missing.
#   - the gateway's secrets, added to .env
#   - Postgres in the otari-memory-pg container (port 5433), with a non-superuser role for the gateway,
#     so row-level security applies to it
#   - the Ollama models memory uses
#   - the dashboard bundle
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$HERE/.env"

for tool in docker uv ollama pnpm python3 openssl; do
  command -v "$tool" >/dev/null || { echo "missing $tool: see Requirements in README.md" >&2; exit 1; }
done

if [ ! -f "$ENV_FILE" ]; then
  echo "missing $ENV_FILE: copy .env.example to .env and fill in your Fireworks key." >&2
  exit 1
fi
grep -qE '^FIREWORKS_API_KEY=REPLACE_ME' "$ENV_FILE" &&
  echo "note: FIREWORKS_API_KEY is still REPLACE_ME; see \"Without Fireworks\" in README.md."

echo "== the gateway's secrets"
chmod 600 "$ENV_FILE"
add_secret() { # name command
  grep -q "^$1=" "$ENV_FILE" && return 0
  printf '%s=%s\n' "$1" "$(eval "$2")" >>"$ENV_FILE"
  echo "added $1"
}
add_secret OTARI_SECRET_KEY \
  "(cd '$ROOT' && uv run --frozen python -c 'from gateway.services.secret_box import generate_secret_key; print(generate_secret_key())')"
add_secret OTARI_MASTER_KEY 'echo "sk-local-$(openssl rand -hex 16)"'
add_secret OTARI_PROVIDER_ACCOUNT_PEPPER 'openssl rand -base64 32'

echo "== postgres"
if ! docker container inspect otari-memory-pg >/dev/null 2>&1; then
  docker run -d --name otari-memory-pg -e POSTGRES_USER=otari -e POSTGRES_PASSWORD=otari -e POSTGRES_DB=otari \
    -v otari-memory-pg:/var/lib/postgresql/data -p 127.0.0.1:5433:5432 postgres:16-alpine >/dev/null
fi
docker start otari-memory-pg >/dev/null
for _ in $(seq 1 30); do docker exec otari-memory-pg pg_isready -U otari >/dev/null 2>&1 && break; sleep 1; done
pg() { docker exec otari-memory-pg psql -U otari -d otari -v ON_ERROR_STOP=1 -Atc "$1"; }
# The gateway connects as otari_app. Not a superuser and no BYPASSRLS: memory's row-level security binds it.
[ -n "$(pg "select 1 from pg_roles where rolname = 'otari_app'")" ] ||
  pg "create role otari_app login password 'otari_app' nosuperuser nobypassrls"
[ -n "$(pg "select 1 from pg_database where datname = 'otari_memory'")" ] ||
  pg "create database otari_memory owner otari_app"
echo "otari-memory-pg on 127.0.0.1:5433"

echo "== ollama models"
curl -sf http://127.0.0.1:11434/api/tags >/dev/null ||
  { echo "ollama is not running: start it (ollama serve, or the Ollama app) and run setup.sh again" >&2; exit 1; }
# nomic-embed-text: memory's embeddings. qwen2.5:3b: the small local model the terminal demo answers with.
for model in nomic-embed-text qwen2.5:3b; do
  ollama pull "$model" >/dev/null 2>&1 || { echo "ollama pull $model failed: run it by hand to see why" >&2; exit 1; }
  echo "$model"
done

echo "== dashboard"
make -C "$ROOT" dashboard >/dev/null 2>&1 ||
  { echo "the dashboard build failed: run make dashboard from the repo root to see why" >&2; exit 1; }
echo "built"

echo
echo "Next: ./start.sh, then ./seed.sh in another terminal."
