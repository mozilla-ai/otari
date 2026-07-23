#!/usr/bin/env bash
# Regenerate assets/otari-demo.gif: seed a standalone gateway, drive the
# dashboard through a full page sweep with Playwright, and encode the recording
# to an optimized GIF.
#
# Prereqs (installed once in the dev/sandbox environment):
#   - ffmpeg on PATH                     (webm -> gif)
#   - web deps installed: (cd web && npm ci)
#   - a committed dashboard bundle in src/gateway/static/dashboard
#     (rebuild with: npm --prefix web run build)
#
# Usage: bash scripts/demo_gif/record.sh [--keep-server]
set -euo pipefail

KEEP_SERVER=0
[[ "${1:-}" == "--keep-server" ]] && KEEP_SERVER=1

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
DIR="scripts/demo_gif"
CONFIG="$DIR/otari.yml"
DB="$DIR/demo.db"
ART="$DIR/artifacts"
GIF_OUT="assets/otari-demo.gif"

# Throwaway Fernet key so provider credentials can be encrypted at rest for the
# demo. Not a secret: the DB is ephemeral and holds only fake provider keys.
export OTARI_SECRET_KEY="${OTARI_SECRET_KEY:-wdhWKyd1gwpMjxj9h4EbpW9B6pilzfrNTe0wTnwqPHg=}"
# arm64 sandbox: use Playwright's Ubuntu 24.04 arm64 Chromium build and skip the
# host-requirements re-check (see the project's environment recipe).
export PLAYWRIGHT_HOST_PLATFORM_OVERRIDE="${PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-ubuntu24.04-arm64}"
export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS="${PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS:-1}"

echo ">> Resetting demo database"
rm -f "$DB" "$DB-wal" "$DB-shm"
rm -rf "$ART"
mkdir -p "$ART"

echo ">> Migrating schema"
uv run otari migrate --config "$CONFIG"

echo ">> Seeding demo data"
uv run python "$DIR/seed.py" "sqlite:///./$DB"

echo ">> Starting gateway"
uv run otari serve --config "$CONFIG" >"$ART/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  if [[ "$KEEP_SERVER" != "1" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo ">> Waiting for /health"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "gateway did not come up"; cat "$ART/server.log"; exit 1; }

echo ">> Recording tour"
OUT_DIR="$ART" BASE_URL="http://127.0.0.1:8000" node "$DIR/tour.mjs"

WEBM=$(ls -t "$ART"/video/*.webm | head -1)
echo ">> Encoding GIF from $WEBM"
# Two-pass palette for clean colors; 12fps and 720px wide keeps the file small.
# START_TRIM drops the brief initial Overview loading skeleton so the GIF opens
# on populated content (the tour waits for data before the sweep begins).
PALETTE="$ART/palette.png"
FPS=12
WIDTH=720
START_TRIM=1.4
ffmpeg -y -ss "$START_TRIM" -i "$WEBM" -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen=stats_mode=diff" "$PALETTE" 2>/dev/null
ffmpeg -y -ss "$START_TRIM" -i "$WEBM" -i "$PALETTE" \
  -lavfi "fps=$FPS,scale=$WIDTH:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  "$ART/otari-demo.gif" 2>/dev/null

cp "$ART/otari-demo.gif" "$GIF_OUT"
BYTES=$(wc -c <"$GIF_OUT")
echo ">> Wrote $GIF_OUT ($((BYTES/1024)) KB)"
[[ "$KEEP_SERVER" == "1" ]] && echo "server left running (pid $SERVER_PID)"
true
