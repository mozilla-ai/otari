#!/usr/bin/env bash
# The pilot (pilot.md) on one machine: Firefox -> MLPA -> Otari -> fakes.
#
#   ./run.sh up      Postgres + Redis, fakes :9100, Otari :8100, MLPA :8080
#   ./run.sh check   end-to-end checks through MLPA, as each client calls it
#   ./run.sh firefox Firefox with Smart Window pointed at the local MLPA
#   ./run.sh logs    tail every process's log
#   ./run.sh down    stop everything and drop the databases
#
# MLPA_DIR points at the MLPA checkout on its otari-pilot branch (default ~/MLPA),
# FIREFOX_APP at the Firefox to run (default: the artifact build in ~/firefox).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OTARI_DIR="$(cd "${HERE}/.." && pwd)"
MLPA_DIR="${MLPA_DIR:-${HOME}/MLPA}"
STATE="${HERE}/state"
mkdir -p "${STATE}"

export OTARI_MASTER_KEY="${OTARI_MASTER_KEY:-sk-otari-pilot}"
MLPA_ENV="${STATE}/mlpa.env"

mlpa_env() {
  # Everything MLPA needs in otari mode. Env vars beat MLPA's own .env.
  cat <<EOF
GATEWAY_BACKEND=otari
OTARI_API_BASE=http://127.0.0.1:8100
OTARI_MASTER_KEY=${OTARI_MASTER_KEY}
MASTER_KEY=sk-mlpa-admin
MLPA_UI_ACCESS_KEY=sk-mlpa-ui
DB_HOST=127.0.0.1
DB_PORT=55432
DB_USERNAME=otari
DB_PASSWORD=otari
APP_ATTEST_DB_NAME=app_attest
PORT=8080
# false: verify FxA tokens against production accounts, which is where Firefox signs in.
MLPA_DEBUG=false
MLPA_ENFORCE_SIGNIN_CAP=true
MLPA_MAX_SIGNED_IN_USERS=1000
ENABLE_TRAFFIC_CONTRACT_ENFORCEMENT=false
MLPA_EXPERIMENTATION_AUTHORIZATION_TOKEN=pilot-dev-token
USER_FEATURE_BUDGET_TELEMETRY_MAX_BUDGET=0.000001
EOF
}

wait_for() {
  local url="$1" name="$2"
  for _ in $(seq 1 120); do
    if curl -fsS "${url}" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "${name} did not come up; see ${STATE}/${name}.log" >&2
  exit 1
}

start() {
  local name="$1"; shift
  "$@" >"${STATE}/${name}.log" 2>&1 &
  echo $! >"${STATE}/${name}.pid"
}

stop() {
  local pidfile="${STATE}/$1.pid"
  if [ -f "${pidfile}" ]; then
    # uv run starts the server as a child, which a kill of uv alone leaves running.
    pkill -TERM -P "$(cat "${pidfile}")" 2>/dev/null || true
    kill "$(cat "${pidfile}")" 2>/dev/null || true
    rm -f "${pidfile}"
  fi
}

up() {
  docker compose -f "${HERE}/docker-compose.yml" up -d --wait
  docker exec otari_pilot_postgres psql -U otari -tAc "SELECT 1 FROM pg_database WHERE datname='app_attest'" | grep -q 1 \
    || docker exec otari_pilot_postgres psql -U otari -c 'CREATE DATABASE app_attest'

  start fakes python3 "${HERE}/fakes.py" --port 9100
  wait_for http://127.0.0.1:9100/counts fakes

  (cd "${OTARI_DIR}" && start otari uv run otari serve -c "${HERE}/otari-config.yml" --host 127.0.0.1 --port 8100)
  wait_for http://127.0.0.1:8100/api/v1/health/readiness otari

  mlpa_env >"${MLPA_ENV}"
  set -a; source "${MLPA_ENV}"; set +a
  (cd "${MLPA_DIR}" \
    && uv run alembic --raiseerr -c alembic.ini \
      -x sqlalchemy.url="postgresql://otari:otari@127.0.0.1:55432/app_attest" upgrade head >"${STATE}/mlpa-migrate.log" 2>&1)
  docker exec otari_pilot_postgres psql -U otari -d app_attest -qc \
    "INSERT INTO mlpa_user_capacity (id, max_identities, current_identities) VALUES (1, ${MLPA_MAX_SIGNED_IN_USERS}, 0)
     ON CONFLICT (id) DO UPDATE SET max_identities = EXCLUDED.max_identities"

  # Budgets, one service key per service type, and the per-user limits.
  (cd "${MLPA_DIR}" && uv run python scripts/otari_provision.py --rotate --env-file "${MLPA_ENV}")
  set -a; source "${MLPA_ENV}"; set +a

  (cd "${MLPA_DIR}" && start mlpa uv run mlpa)
  wait_for http://127.0.0.1:8080/health/liveness mlpa
  echo "Up: MLPA http://127.0.0.1:8080 -> Otari http://127.0.0.1:8100 (dashboard, master key ${OTARI_MASTER_KEY}) -> fakes :9100"
}

case "${1:-}" in
  up) up ;;
  check)
    set -a; source "${MLPA_ENV}"; set +a
    (cd "${MLPA_DIR}" && uv run python "${HERE}/check.py") ;;
  firefox)
    app="${FIREFOX_APP:-$(ls -d "${HOME}"/firefox/obj-*/dist/Nightly.app 2>/dev/null | head -1)}"
    profile="${STATE}/firefox-profile"
    mkdir -p "${profile}"
    cp "${HERE}/firefox/user.js" "${profile}/user.js"
    "${app}/Contents/MacOS/firefox" -no-remote -profile "${profile}" >"${STATE}/firefox.log" 2>&1 & ;;
  logs) tail -n 50 -f "${STATE}"/*.log ;;
  down)
    stop mlpa; stop otari; stop fakes
    docker compose -f "${HERE}/docker-compose.yml" down -v ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
