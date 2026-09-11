#!/usr/bin/env bash
# Run the gateway on this machine with three people to sign in as, for a smoke
# test on the LAN. Bring your own provider keys.
#
#   demo/catalog-smoke/run.sh [--port 8000] [--no-build] [--reset]
#
# Builds the dashboard, writes a config from config.template.yml, starts the
# gateway on 0.0.0.0, seeds three people to sign in as (seed.py), and prints
# who they are. State lives in demo/catalog-smoke/.state and survives runs;
# --reset starts the database over, --no-build skips the dashboard build.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
STATE="$HERE/.state"
PORT=8000
BUILD=1
RESET=0

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --reset) RESET=1; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

for tool in uv pnpm python3; do
  command -v "$tool" >/dev/null || { echo "$tool is required (see AGENTS.md)" >&2; exit 1; }
done

mkdir -p "$STATE"
[ -f "$STATE/.gitignore" ] || echo '*' > "$STATE/.gitignore"
if [ "$RESET" = 1 ]; then
  rm -f "$STATE/otari.db" "$STATE/otari.db-wal" "$STATE/otari.db-shm" "$STATE/seeded" "$STATE/config.yml"
fi

# One master key and one credential-encryption key per state directory, so a
# phone that signed in yesterday still can, and a provider key stored through
# the dashboard still decrypts.
if [ ! -f "$STATE/master-key" ]; then
  python3 -c 'import secrets; print("otari-mk-" + secrets.token_urlsafe(32))' > "$STATE/master-key"
fi
if [ ! -f "$STATE/secret-key" ]; then
  (cd "$ROOT" && uv run otari gen-secret-key) > "$STATE/secret-key"
fi
MASTER_KEY="$(cat "$STATE/master-key")"
export OTARI_SECRET_KEY="$(cat "$STATE/secret-key")"
# A stale platform token in the shell would boot this as a hybrid gateway with
# no management API at all; the smoke is the standalone edition.
unset OTARI_AI_TOKEN

LAN_IP="$( (ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}') || true)"
PUBLIC_BASE_URL="http://${LAN_IP:-localhost}:$PORT"

# The config is written once and then left alone, so a provider key you put in
# it stays put across runs. --reset writes it fresh.
if [ ! -f "$STATE/config.yml" ]; then
  sed "s|__MASTER_KEY__|$MASTER_KEY|; s|__DB__|$STATE/otari.db|; s|__PUBLIC_BASE_URL__|$PUBLIC_BASE_URL|" \
    "$HERE/config.template.yml" > "$STATE/config.yml"
fi

if [ "$BUILD" = 1 ]; then
  echo "Building the dashboard (pnpm install + vite build)…"
  (cd "$ROOT" && make dashboard >/dev/null)
fi

cd "$ROOT"
# A wide console for the log file: the logger wraps at 80 columns when it is
# not a terminal, which cuts a verification link in two.
COLUMNS=400 uv run otari serve --config "$STATE/config.yml" --host 0.0.0.0 --port "$PORT" > "$STATE/gateway.log" 2>&1 &
GATEWAY=$!
echo "$GATEWAY" > "$STATE/gateway.pid"
trap 'kill "$GATEWAY" 2>/dev/null; wait "$GATEWAY" 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  if ! kill -0 "$GATEWAY" 2>/dev/null; then
    echo "the gateway exited before answering; see $STATE/gateway.log" >&2
    exit 1
  fi
  sleep 1
done

PASSWORD="${OTARI_SMOKE_PASSWORD:-smoke-test-1234}"
if [ ! -f "$STATE/seeded" ]; then
  OTARI_BASE="http://127.0.0.1:$PORT" OTARI_MASTER_KEY="$MASTER_KEY" OTARI_LOG="$STATE/gateway.log" \
    OTARI_SMOKE_PASSWORD="$PASSWORD" python3 "$HERE/seed.py"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE/seeded"
fi

cat <<EOF

  Otari catalog smoke
  ───────────────────
  Dashboard   http://localhost:$PORT/            ${LAN_IP:+(LAN: http://$LAN_IP:$PORT/)}
  Public      http://localhost:$PORT/#/models    (signed out)
  Master key  $MASTER_KEY                        (the API credential; header Authorization: Bearer)
  Config      $STATE/config.yml                  (put provider keys here and restart, or add them on Providers)
  Log         $STATE/gateway.log                 (invitation and verification links land here)

  Sign in as                      password: $PASSWORD
  ───────────────────────────────────────────────────────
  Platform admin   operator@otari.local   owns the deployment: Settings, Model pricing's catalog
                                          controls, Accounts, every provider and every organization
  Org admin        admin@acme.local       admin of Acme: its rate overrides, provider keys, members,
                                          budgets; sees the deployment's prices, cannot change them
  Member           member@acme.local      member of Acme: browses the catalog at Acme's rates,
                                          makes their own API keys, sees their own usage

  Ctrl-C stops the gateway.

EOF
wait "$GATEWAY"
