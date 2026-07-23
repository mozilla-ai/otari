#!/usr/bin/env bash
# Regenerate assets/otari-demo.gif: seed a standalone gateway, drive the
# dashboard through a full page sweep with Playwright, and encode the recording
# to an optimized GIF.
#
# Prereqs (installed once in the dev/sandbox environment):
#   - ffmpeg on PATH                     (webm -> gif)
#   - gifsicle on PATH                   (lossy GIF optimisation)
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
# Playwright ships no native arm64 Linux Chromium, so on arm64 force its Ubuntu
# 24.04 arm64 build and skip the host-requirements re-check. On x86_64 (or any
# other arch) leave Playwright's native detection and host checks alone, so the
# pipeline does not select an incompatible Chromium there. Caller-set values win.
case "$(uname -m)" in
  aarch64 | arm64)
    export PLAYWRIGHT_HOST_PLATFORM_OVERRIDE="${PLAYWRIGHT_HOST_PLATFORM_OVERRIDE:-ubuntu24.04-arm64}"
    export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS="${PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS:-1}"
    ;;
esac

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
# Encode at 1080 wide while the README displays it at 720, so it stays crisp on
# high-DPI screens (the same 1.5x density the previous hero GIF used). START_TRIM
# drops the brief initial Overview loading skeleton so the GIF opens on populated
# content. The flat dashboard UI needs few colors, so a 64-colour palette plus a
# gifsicle lossy pass roughly halves the file with no visible loss. Note: 12fps
# is smaller than 10fps here because the diff-based GIF stores smaller
# frame-to-frame rectangles when there is less motion per frame.
PALETTE="$ART/palette.png"
FPS=12
WIDTH=1080
START_TRIM=1.4
COLORS=64
LOSSY=80
ffmpeg -y -ss "$START_TRIM" -i "$WEBM" -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen=max_colors=$COLORS:stats_mode=diff" "$PALETTE" 2>/dev/null
ffmpeg -y -ss "$START_TRIM" -i "$WEBM" -i "$PALETTE" \
  -lavfi "fps=$FPS,scale=$WIDTH:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  "$ART/otari-demo-raw.gif" 2>/dev/null
# gifsicle squeezes the LZW stream further (lossy) and optimises frames.
gifsicle -O3 --lossy="$LOSSY" "$ART/otari-demo-raw.gif" -o "$ART/otari-demo.gif" 2>/dev/null

cp "$ART/otari-demo.gif" "$GIF_OUT"
BYTES=$(wc -c <"$GIF_OUT")
echo ">> Wrote $GIF_OUT ($((BYTES/1024)) KB)"
[[ "$KEEP_SERVER" == "1" ]] && echo "server left running (pid $SERVER_PID)"
true
