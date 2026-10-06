#!/usr/bin/env bash
# Drive the load test. Each scenario creates its own tenant, runs the load, then
# checks the ledgers once the tenant's requests have settled. Results land in
# results/, tenants in state/.
#
#   ./run.sh up                 build the image from OTARI_BUILD_CONTEXT and start the stack
#   ./run.sh baseline           the client and fake provider alone, no gateway
#   ./run.sh steady             300 RPM, then 1,000 RPM
#   ./run.sh spill              1,000 RPM at Model 1's 100 RPM cap
#   ./run.sh shared-budget      many end users draining one small shared pool
#   ./run.sh budget-reset       end-user budgets reset every 60s mid-run
#   ./run.sh provider-429       Model 1 throttles every call
#   ./run.sh stream-fail        a fifth of streams drop after the first chunk
#   ./run.sh redis-down         Redis stopped for a minute mid-run
#   ./run.sh kill-replica       otari-2 killed mid-run, restarted, holds expire
#   ./run.sh all                every scenario above, in order
#   ./run.sh bench LABEL        profiled 1,000 RPM, without and with a shared pool
#   ./run.sh count [spill]      database statements per request, direct or spilled
#   ./run.sh check NAME [RESULT]  re-run the checks for a tenant
#   ./run.sh logs | down
#
# Knobs (env): DURATION (seconds per phase, default 120), RPM_LOW, RPM_HIGH,
# USERS, STREAM_SHARE, OTARI_BUILD_CONTEXT (the checkout to build, default the
# one this is in), OTARI_LOADTEST_CONFIG (default ./otari-config.yml),
# PROFILE=1 to profile a scenario, PROFILE_FORMAT=flamegraph for SVGs instead
# of collapsed stacks.
set -euo pipefail
cd "$(dirname "$0")"

DURATION=${DURATION:-120}
RPM_LOW=${RPM_LOW:-300}
RPM_HIGH=${RPM_HIGH:-1000}
USERS=${USERS:-200}
STREAM_SHARE=${STREAM_SHARE:-0.5}
LATENCY_MS=${FAKE_LATENCY_MS:-300}
FAKE="http://127.0.0.1:${LOADTEST_FAKE_PORT:-19000}"

dc() { docker compose "$@"; }
tool() { dc run --rm --no-deps tools "$@"; }

fake() {
  local body=${2:-'{}'}
  curl -fsS -X POST "$FAKE/$1" -H 'Content-Type: application/json' -d "$body" >/dev/null
}
fake_defaults() {
  fake _control '{"*": {"error_429_rate": 0, "stream_fail_rate": 0}, "m1": {"quota_rpm": '"${FAKE_M1_QUOTA_RPM:-100}"'}}'
  fake _reset
}
fake_defaults_keep_stats() {
  fake _control '{"*": {"error_429_rate": 0, "stream_fail_rate": 0}}'
}

latest_result() { ls -t results/"$1"-*.json 2>/dev/null | head -1 || true; }

setup() { tool setup_tenant.py "$@"; }

load() {
  local name=$1; shift
  [[ "${PROFILE:-0}" == 1 ]] && profile_start "$name"
  tool loadgen.py --key-file "state/$name.json" --label "$name" --users "$USERS" \
    --stream-share "$STREAM_SHARE" --provider-latency-ms "$LATENCY_MS" "$@" || true
  [[ "${PROFILE:-0}" == 1 ]] && profile_stop
  return 0
}

# PROFILE=1: snapshot metrics, Postgres and Redis around the load, sample
# Postgres's lock waits during it, and record each replica with py-spy.
# Everything lands in results/profile-NAME-TIMESTAMP/, report.md first.
PROFILE_SIDECARS=(lt-sampler lt-pyspy-1 lt-pyspy-2)
profile_start() {
  PROFILE_DIR="results/profile-$1-$(date +%Y%m%d-%H%M%S)"
  local format=raw extension=txt
  if [[ "${PROFILE_FORMAT:-raw}" == flamegraph ]]; then format=flamegraph extension=svg; fi
  docker rm -f "${PROFILE_SIDECARS[@]}" >/dev/null 2>&1 || true
  tool profile.py begin --out "$PROFILE_DIR"
  dc run -d --no-deps --name lt-sampler tools profile.py sample --out "$PROFILE_DIR" --seconds 86400 >/dev/null
  for i in 1 2; do
    # --nonblocking reads the stacks without pausing the gateway, so the profile
    # does not add to the latency it is explaining.
    dc --profile profile run -d --no-deps --name "lt-pyspy-$i" "pyspy-$i" \
      record --pid 1 --rate 100 --nonblocking --format "$format" -o "/work/$PROFILE_DIR/otari-$i.$extension" >/dev/null
  done
}
profile_stop() {
  # py-spy writes its recording when interrupted.
  docker kill -s INT lt-pyspy-1 lt-pyspy-2 >/dev/null 2>&1 || true
  for _ in $(seq 30); do
    [[ -z "$(docker ps -q -f name=lt-pyspy)" ]] && break
    sleep 2
  done
  docker rm -f "${PROFILE_SIDECARS[@]}" >/dev/null 2>&1 || true
  tool profile.py end --out "$PROFILE_DIR" >/dev/null
  echo "profile: $PROFILE_DIR/report.md"
}

# check NAME [RESULT]: RESULT defaults to the newest loadgen result for NAME.
check() {
  local name=$1 result=${2:-}
  [[ -n "$result" ]] || result=$(latest_result "$name")
  tool check.py --name "$name" ${result:+--loadgen-result "$result"} \
    --drain-timeout "${DRAIN_TIMEOUT:-120}" ${CHECK_FLAGS:-} || true
}

scenario_baseline() {
  fake_defaults
  # Straight at the fake provider's Model 2 slot: the client, the network and the
  # fake's fixed latency, with no gateway. Subtract this from a gateway run.
  tool loadgen.py --base-url http://fakeprovider:9000 --path /m2/v1/chat/completions \
    --model llama-3.3-70b --users 0 --label baseline --stream-share "$STREAM_SHARE" \
    --provider-latency-ms "$LATENCY_MS" --phase "$RPM_HIGH:60"
}

scenario_steady() {
  fake_defaults
  setup --name steady
  load steady --phase "$RPM_LOW:$DURATION" --phase "$RPM_HIGH:$DURATION"
  check steady
}

scenario_spill() {
  fake_defaults
  setup --name spill
  load spill --phase "$RPM_HIGH:$DURATION" --stream-share 0
  check spill
}

scenario_shared_budget() {
  fake_defaults
  # About $0.60 a minute at 1,000 RPM, so a $0.50 pool runs dry inside the
  # first minute and every later request must be refused, by the budget.
  setup --name shared-budget --pool-usd 0.50
  USERS=${USERS_MANY:-2000} load shared-budget --phase "$RPM_HIGH:$DURATION"
  check shared-budget
}

scenario_budget_reset() {
  fake_defaults
  # A small end-user budget that resets every 60s: users hit it, get refused,
  # then are let in again at each boundary.
  setup --name budget-reset --end-user-usd 0.002 --end-user-period 60
  USERS=50 load budget-reset --phase "$RPM_LOW:$(( DURATION * 2 ))"
  check budget-reset
}

scenario_provider_429() {
  fake_defaults
  setup --name provider-429
  fake _control '{"m1": {"error_429_rate": 1.0}}'
  load provider-429 --phase "$RPM_LOW:$DURATION"
  fake_defaults_keep_stats
  # Every Model 1 call was refused by the provider here, not the gateway's cap.
  CHECK_FLAGS=--skip-cap check provider-429
}

scenario_stream_fail() {
  fake_defaults
  setup --name stream-fail
  fake _control '{"*": {"stream_fail_rate": 0.2}}'
  STREAM_SHARE=1 load stream-fail --phase "$RPM_LOW:$DURATION"
  fake_defaults_keep_stats
  check stream-fail
}

scenario_redis_down() {
  fake_defaults
  setup --name redis-down
  load redis-down --phase "$RPM_HIGH:$(( DURATION + 120 ))" &
  local loader=$!
  sleep 60
  echo ">>> stopping redis"; dc stop redis
  sleep 60
  echo ">>> starting redis"; dc start redis
  wait "$loader"
  # With Redis gone each replica counted Model 1's cap on its own.
  CHECK_FLAGS=--skip-cap check redis-down
}

scenario_kill_replica() {
  fake_defaults
  setup --name kill-replica
  # Long streams, so the kill lands on requests holding budget and slots.
  fake _control '{"*": {"chunks": 200, "chunk_interval_ms": 50}}'
  STREAM_SHARE=1 load kill-replica --phase "$RPM_LOW:$(( DURATION + 60 ))" &
  local loader=$!
  sleep 60
  echo ">>> killing otari-2"; dc kill -s KILL otari-2
  sleep 20
  echo ">>> restarting otari-2"; dc start otari-2
  wait "$loader"
  fake _control '{"*": {"chunks": 20, "chunk_interval_ms": 20}}'
  # The killed replica's holds expire after budget_reservation_ttl_sec (120s)
  # and the sweeper runs every 30s.
  DRAIN_TIMEOUT=300 check kill-replica
}

# bench LABEL: the before/after comparison. Profiled 1,000 RPM runs on a fresh
# tenant without a shared pool, then with one, each checked afterwards.
bench() {
  local label=${1:?bench needs a label}
  for variant in nopool pool; do
    local name="$label-$variant"
    fake_defaults
    if [[ $variant == nopool ]]; then setup --name "$name" --no-pool; else setup --name "$name"; fi
    PROFILE=1 load "$name" --phase "$RPM_HIGH:$DURATION" | grep '^{"label"' || true
    check "$name"
  done
}

# count [spill]: statements per request over 100 identical non-streamed
# requests on end users that already exist. "spill" sends them at the policy
# once Model 1 is full, so each spills to Model 2; otherwise straight to Model 2.
count() {
  local mode=${1:-direct} model=togethersim:llama-3.3-70b
  [[ $mode == spill ]] && model=summarize
  local tenant="count-$mode"
  fake_defaults
  [[ -f "state/$tenant.json" ]] || setup --name "$tenant" >/dev/null
  # Warm: creates the end users and, for a spill, fills Model 1's minute.
  tool loadgen.py --key-file "state/$tenant.json" --label warm --users 50 --stream-share 0 \
    --model "$model" --phase 1200:10 >/dev/null
  DRAIN_TIMEOUT=30 CHECK_FLAGS=--skip-cap check "$tenant" >/dev/null
  tool profile.py reset-statements
  tool loadgen.py --key-file "state/$tenant.json" --label "count-$mode" --users 50 --stream-share 0 \
    --model "$model" --phase 600:10 >/dev/null
  DRAIN_TIMEOUT=30 CHECK_FLAGS=--skip-cap check "$tenant" >/dev/null
  tool profile.py statements --requests 100 --out "results/statements-$mode-$(date +%Y%m%d-%H%M%S).txt"
}

case "${1:-}" in
  up)
    mkdir -p results state
    dc up -d --build --wait
    echo "stack up: load balancer on 127.0.0.1:${LOADTEST_LB_PORT:-18080}, fake provider on $FAKE"
    ;;
  down) dc --profile tools down -v ;;
  logs) dc logs -f otari-1 otari-2 ;;
  baseline) scenario_baseline ;;
  steady) scenario_steady ;;
  spill) scenario_spill ;;
  shared-budget) scenario_shared_budget ;;
  budget-reset) scenario_budget_reset ;;
  provider-429) scenario_provider_429 ;;
  stream-fail) scenario_stream_fail ;;
  redis-down) scenario_redis_down ;;
  kill-replica) scenario_kill_replica ;;
  bench) shift; bench "$@" ;;
  count) shift; count "$@" ;;
  check) shift; check "$@" ;;
  all)
    for s in baseline steady spill shared-budget budget-reset provider-429 stream-fail redis-down kill-replica; do
      echo "===== $s ====="; "$0" "$s"
    done
    ;;
  *) sed -n '2,26p' "$0"; exit 1 ;;
esac
