#!/usr/bin/env bash
# Puts the local environment in the demo's starting state. Safe to run again: it erases memory first.
#   prepare.sh             everything, a savings experiment of 50 sessions included (~3 min)
#   prepare.sh --quick     no experiment
source "$(dirname "$0")/lib.sh"

heading "Preflight"
curl -sf "$API/health" >/dev/null && say "gateway   ok" || { echo "${RED}gateway is down: run ../start.sh${OFF}"; exit 1; }
curl -sf http://127.0.0.1:11434/api/tags >/dev/null && say "ollama    ok" || { echo "${RED}ollama is down${OFF}"; exit 1; }
pg_super "select 1" >/dev/null && say "postgres  ok"

heading "Reset"
admin PUT /organizations/me/memory '{"enabled": true}' >/dev/null
switches '{}'
admin DELETE "/workspaces/$WS/memory" >/dev/null
say "memory erased; organization on; workspace on, shared, auto policy; recall and experiment off"

heading "Seed what the team's agents learned last week"
seed() { tool "$1" memory_propose "$2" >/dev/null; }
seed ALICE '{"kind":"procedure","scope":"workspace","text":"To deploy the gateway to staging, tag the release, then run make deploy-staging from the repo root."}'
seed BOB '{"kind":"fact","scope":"workspace","text":"Staging runs migrations at startup (auto_migrate), so a failed migration shows up as the gateway never turning healthy, not as an error in the deploy."}'
seed ALICE '{"kind":"decision","scope":"workspace","text":"Agent memory uses Postgres row-level security, so a bug in a query filter can never return another workspace'"'"'s rows."}'
seed BOB '{"kind":"procedure","scope":"workspace","text":"When an Ollama model fails with MissingApiKeyError, register Ollama as a hosted provider: providers in config.yml are no longer read."}'
seed ALICE '{"kind":"fact","scope":"member","text":"My own staging dashboard is pinned to the eu-west-1 region."}'
wait_consolidated
say "$(pg_super "select count(*) from memory_entries where status='active'") entries in memory"

say "warming the models"
tool BOB memory_search '{"query":"warm up"}' >/dev/null
ask "Say ok." -H 'Otari-Memory: off' >/dev/null

if [ "${1:-}" != "--quick" ]; then
  heading "Savings experiment: 50 sessions, half without memory"
  switches '{"auto_recall": true, "experiment_holdout_percent": 50}'
  questions=(
    "Our staging deploy just failed. How do we deploy the gateway to staging?"
    "The gateway never turns healthy after a staging deploy. What should I check?"
    "Why does agent memory use row-level security?"
    "An Ollama model fails with MissingApiKeyError. How do I fix it?"
    "Write a one-line commit message for a typo fix."
  )
  for index in $(seq 1 50); do
    question="${questions[$((index % ${#questions[@]}))]}"
    ask "$question" -H "Otari-Conversation-Id: demo-session-$index" >/dev/null
    printf '.'
  done
  echo
  switches '{"auto_recall": false}'
  say "experiment stopped; its results stay on the Memory tab"
fi

heading "Ready"
say "Dashboard: http://127.0.0.1:8010  (sign in with OTARI_MASTER_KEY from $ENV_DIR/.env)"
say "Then: $DEMO_DIR/run.sh"
