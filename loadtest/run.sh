#!/usr/bin/env bash
# Compare two Otari images under load on this machine (README.md):
#
#   ./run.sh BASE_IMAGE HEAD_IMAGE
#
# Starts the stack, lets the two images take turns, runs the checks in
# comparison.py (pytest: one line per check), prints the report and removes the
# stack. Exits non-zero when a check fails.
set -euo pipefail
cd "$(dirname "$0")"
base=${1:?usage: ./run.sh BASE_IMAGE HEAD_IMAGE}
head=${2:?usage: ./run.sh BASE_IMAGE HEAD_IMAGE}
dir="results/ab-$(date +%Y%m%d-%H%M%S)"

dc() { docker compose "$@"; }
# tool [NAME=VALUE ...] -- COMMAND ...: COMMAND in the tools container, on the stack's network.
tool() {
  local env=()
  while [[ $1 != -- ]]; do env+=(-e "$1"); shift; done
  shift
  dc exec -T ${env[@]+"${env[@]}"} tools uv run --quiet --frozen --only-group loadtest "$@"
}
# use VARIANT IMAGE: IMAGE as the replica under test, on database ab_VARIANT, freshly started.
use() {
  echo ">>> $1: $2"
  OTARI_IMAGE=$2 LOADTEST_DATABASE_URL="postgresql://otari:otari@postgres:5432/ab_$1" \
    dc up -d --no-deps --force-recreate --wait otari >/dev/null 2>&1 || { dc logs --tail 50 otari; return 1; }
}

trap 'dc down -v >/dev/null 2>&1' EXIT
docker volume create otari-loadtest-uv-cache >/dev/null
dc up -d --wait postgres redis fakeprovider tools >/dev/null 2>&1
# A database per build: a migration in head would break base on a shared one.
for variant in base head; do
  dc exec -T postgres psql -U otari -q -c "CREATE DATABASE ab_$variant" >/dev/null
done

order=$(tool "BASE_REF=${BASE_REF:-$base}" "HEAD_REF=${HEAD_REF:-$head}" \
  "BASE_ID=$(docker image inspect -f '{{.Id}}' "$base")" "HEAD_ID=$(docker image inspect -f '{{.Id}}' "$head")" \
  -- python ab.py prepare --dir "$dir")
image_of() { if [[ $1 == base ]]; then echo "$base"; else echo "$head"; fi; }
# A turn that fails leaves its runs missing, which a check then reports.
turn=0
for variant in $order; do
  turn=$((turn + 1))
  use "$variant" "$(image_of "$variant")" \
    && tool -- python ab.py turn --dir "$dir" --variant "$variant" --turn "$turn" || echo "!!! turn $turn failed"
done
for variant in base head; do
  use "$variant" "$(image_of "$variant")" \
    && tool -- python ab.py statements --dir "$dir" --variant "$variant" || echo "!!! statements for $variant failed"
done

status=0
tool "LOADTEST_AB_DIR=$dir" "LOADTEST_ACCEPT_REGRESSION=${LOADTEST_ACCEPT_REGRESSION:-0}" -- \
  pytest comparison.py || status=$?
cat "$dir/report.md" 2>/dev/null || true
echo "report: loadtest/$dir/report.md, figures: loadtest/$dir/perf.json"
exit $status
