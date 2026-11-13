# Shared helpers for the agent memory demo. Sourced by prepare.sh and run.sh.
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$DEMO_DIR")"
source "$ENV_DIR/.env"
source "$ENV_DIR/keys.env"

API=http://127.0.0.1:8010/api/v1
WS="$OTARI_DEMO_WORKSPACE"
MODEL=ollama:qwen2.5:3b
PG_SUPER=$(docker exec otari-memory-pg printenv POSTGRES_USER)

BOLD=$'\e[1m'; DIM=$'\e[2m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; OFF=$'\e[0m'

heading() { printf '\n%s━━ %s ━━%s\n' "$BOLD$CYAN" "$1" "$OFF"; }
say() { printf '%s%s%s\n' "$DIM" "$1" "$OFF"; }
# DEMO_AUTO=1 runs straight through, and does through the API what the presenter would do in the dashboard.
pause() { [ -n "${DEMO_AUTO:-}" ] && return 0; read -r -p $'\n'"${YELLOW}⏎ next${OFF} " _ </dev/tty; }
in_dashboard() { # what to do there; the API call that does the same under DEMO_AUTO
  printf '\n%s%s%s\n' "$YELLOW" "$1" "$OFF"
  [ -n "${DEMO_AUTO:-}" ] && eval "$2" >/dev/null
  return 0
}

# The master key's calls: the operator's, as the dashboard makes them.
admin() { # method path [body]
  curl -sf -X "$1" "$API$2" -H "Authorization: Bearer $OTARI_MASTER_KEY" -H 'Content-Type: application/json' ${3:+-d "$3"}
}

switches() { # json patch merged over the demo's baseline switches
  admin PUT "/workspaces/$WS/memory" "$(python3 -c '
import json, sys
base = {"enabled": True, "workspace_scope_enabled": True, "write_policy": "auto",
        "offer_to_every_request": False, "hide_external": False, "retention_days": None,
        "auto_recall": False, "recall_min_similarity": 0.65, "recall_max_entries": 3,
        "recall_rerank": False, "recall_min_relevance": 0.8,
        "experiment_holdout_percent": 0}
base.update(json.loads(sys.argv[1]))
print(json.dumps(base))' "${1:-{\}}")" >/dev/null
}

current() { admin GET "/workspaces/$WS/memory"; }

# One memory tool call over MCP, as ALICE or BOB; prints what the agent reads.
tool() { # who tool [arguments-json]
  local key_var="$1_KEY"
  curl -s "$API/memory/mcp" -H "Authorization: Bearer ${!key_var}" -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"$2\",\"arguments\":${3:-{\}}}}" |
    python3 "$DEMO_DIR/show.py" tool
}

# One chat completion through the gateway as BOB; prints the answer and what memory did.
ask() { # question [extra curl args...]
  local question="$1"; shift
  local headers; headers=$(mktemp)
  curl -s -D "$headers" -m 170 "$API/chat/completions" -H "Authorization: Bearer $BOB_KEY" -H 'Content-Type: application/json' "$@" \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 120, "messages": [{"role": "user", "content": sys.argv[2]}]}))' "$MODEL" "$question")" |
    python3 "$DEMO_DIR/show.py" chat
  grep -i '^otari-memory' "$headers" | tr -d '\r' | sed "s/^/  ${DIM}/;s/\$/${OFF}/" || say "  (memory not involved)"
  rm -f "$headers"
}

pg_super() { docker exec otari-memory-pg psql -U "$PG_SUPER" -d otari_memory -Atc "$1"; }
pg_app() { docker exec otari-memory-pg psql -U otari_app -d otari_memory -Atc "$1"; }

wait_consolidated() { # until no proposal is pending, or 90 s
  for _ in $(seq 1 60); do
    [ "$(pg_super "select count(*) from memory_proposals where state='pending'")" = 0 ] && return 0
    sleep 1.5
  done
  printf '%sconsolidator still busy%s\n' "$RED" "$OFF"
}
