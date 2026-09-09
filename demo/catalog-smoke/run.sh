#!/usr/bin/env bash
# Run the gateway from this checkout with the built dashboard and a seeded
# catalog, bound to every interface so a phone or a colleague on the LAN can
# open it.
#
#   demo/catalog-smoke/run.sh            build the dashboard, seed, serve on :8000
#   demo/catalog-smoke/run.sh --port 9000
#   demo/catalog-smoke/run.sh --reset    start from an empty database
#   demo/catalog-smoke/run.sh --no-build reuse the last dashboard bundle
#
# Everything it writes lives under demo/catalog-smoke/.state/ (gitignored by the
# otari.db pattern is not enough, so the whole directory is ignored below), and
# nothing here needs a real provider key: discovery is off, so the catalog is
# the priced rows in config.yml, and models.dev is fetched live for metadata.
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
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

for tool in uv pnpm; do
  command -v "$tool" >/dev/null || { echo "$tool is required (see AGENTS.md)" >&2; exit 1; }
done

mkdir -p "$STATE"
[ -f "$STATE/.gitignore" ] || echo '*' > "$STATE/.gitignore"
[ "$RESET" = 1 ] && rm -f "$STATE/otari.db"

# One master key per state directory, so a phone that signed in yesterday still
# has a working session after a restart.
if [ ! -f "$STATE/master-key" ]; then
  python3 -c 'import secrets; print("otari-mk-" + secrets.token_urlsafe(32))' > "$STATE/master-key"
fi
MASTER_KEY="$(cat "$STATE/master-key")"

# Written fresh every run, so an edit to the seed below lands on restart. The
# config loader inserts a price only where no row exists for that key and
# effective_at, so a rate you changed here after the first boot needs --reset
# (or a different effective_at) to take.
sed "s|__MASTER_KEY__|$MASTER_KEY|; s|__DB__|$STATE/otari.db|" "$HERE/config.template.yml" > "$STATE/config.yml"

if [ "$BUILD" = 1 ]; then
  echo "Building the dashboard (pnpm install + vite build)…"
  (cd "$ROOT" && make dashboard >/dev/null)
fi

# The address a phone on the same network reaches. macOS and Linux spell it
# differently; either way it is advice, not a bind address.
LAN_IP="$( (ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}') || true)"

cat <<EOF

  Otari catalog smoke
  ───────────────────
  Dashboard   http://localhost:$PORT/            ${LAN_IP:+(LAN: http://$LAN_IP:$PORT/)}
  Models      http://localhost:$PORT/#/models
  Catalog API http://localhost:$PORT/v1/catalog/models
  Master key  $MASTER_KEY
  State       $STATE

  Sign in with the master key, or open #/models signed out for the public
  catalog. Ctrl-C stops the gateway.

EOF

cd "$ROOT"
exec uv run otari serve --config "$STATE/config.yml" --host 0.0.0.0 --port "$PORT"
