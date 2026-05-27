#!/usr/bin/env bash
# End-to-end routing policy walkthrough.
#
# This script exercises the management and dry-run routing control plane
# without spending tokens. Set RUN_PROVIDER_CALL=1 in .env to send one real
# default_routing chat completion and then inspect route traces.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$HERE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$HERE/.env"
  set +a
fi

GATEWAY_PORT=${OTARI_PORT:-${GATEWAY_PORT:-8000}}
GATEWAY_URL=${GATEWAY_URL:-http://localhost:${GATEWAY_PORT}}
GATEWAY_KEY=${GATEWAY_KEY:-demo-master-key}
DEMO_ID=${DEMO_ID:-routing-demo-$(date +%s)}

api() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  if [[ -n "$body" ]]; then
    curl -fsS -X "$method" "$GATEWAY_URL$path" \
      -H "Authorization: Bearer $GATEWAY_KEY" \
      -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -fsS -X "$method" "$GATEWAY_URL$path" \
      -H "Authorization: Bearer $GATEWAY_KEY"
  fi
}

json_get() {
  python3 -c '
import json, sys
obj = json.load(sys.stdin)
for part in sys.argv[1].split("."):
    obj = obj[part]
print(obj)
' "$1"
}

pretty() {
  python3 -m json.tool
}

section() {
  printf "\n== %s ==\n" "$1"
}

section "Health"
api GET /health | pretty

section "Create default lowest-cost policy"
default_policy=$(
  api POST /v1/routing-policies '{
    "name": "Default cost router",
    "strategy": "lowest_cost",
    "is_default": true,
    "status": "active",
    "change_note": "routing demo default policy",
    "config": {
      "fallback_enabled": true,
      "health": {
        "enabled": true,
        "mode": "downrank",
        "min_samples": 3
      },
      "candidates": [
        {
          "model": "openai:gpt-4o",
          "input_price_per_million": 5.0,
          "output_price_per_million": 15.0
        },
        {
          "model": "openai:gpt-4o-mini",
          "input_price_per_million": 0.15,
          "output_price_per_million": 0.60
        }
      ]
    }
  }'
)
printf '%s\n' "$default_policy" | pretty
default_policy_id=$(printf '%s\n' "$default_policy" | json_get policy_id)

section "Create project attached to default policy"
api POST /v1/projects "{
  \"project_id\": \"$DEMO_ID\",
  \"name\": \"Routing demo $DEMO_ID\",
  \"routing_policy_id\": \"$default_policy_id\"
}" | pretty

section "Dry-run project routing"
api POST /v1/routing/resolve '{
  "model": "default_routing",
  "project_id": "'"$DEMO_ID"'",
  "tags": {"tenant": "standard"},
  "messages": [{"role": "user", "content": "Summarize this short note"}],
  "max_tokens": 120
}' | pretty

section "Create active canary policy at 25 percent for tenant=vip"
canary_policy=$(
  api POST /v1/routing-policies '{
    "name": "VIP canary",
    "strategy": "priority",
    "is_default": false,
    "status": "active",
    "change_note": "routing demo canary",
    "config": {
      "match": {
        "tags": {"tenant": "vip"},
        "priority": 20,
        "rollout_percentage": 25,
        "bucket_by": "tenant"
      },
      "candidates": ["anthropic:claude-3-5-haiku-latest"]
    }
  }'
)
printf '%s\n' "$canary_policy" | pretty
canary_policy_id=$(printf '%s\n' "$canary_policy" | json_get policy_id)

section "Dry-run canary decision"
api POST /v1/routing/resolve '{
  "model": "default_routing",
  "tags": {"tenant": "vip"},
  "messages": [{"role": "user", "content": "Say hello from the canary cohort"}]
}' | pretty

section "Clone default policy into a draft"
draft_policy=$(
  api POST "/v1/routing-policies/$default_policy_id/clone" '{
    "name": "Default cost router draft",
    "change_note": "stage alternate policy"
  }'
)
printf '%s\n' "$draft_policy" | pretty
draft_policy_id=$(printf '%s\n' "$draft_policy" | json_get policy_id)

section "Dry-run explicit draft policy without sending live traffic"
api POST /v1/routing/resolve "{
  \"model\": \"default_routing\",
  \"policy_id\": \"$draft_policy_id\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Preview the draft policy\"}]
}" | pretty

section "Update default policy, then roll it back to revision 1"
api PATCH "/v1/routing-policies/$default_policy_id" '{
  "strategy": "priority",
  "change_note": "temporary route preference",
  "config": {
    "candidates": ["openai:gpt-4o"]
  }
}' | pretty

api POST "/v1/routing-policies/$default_policy_id/revisions/1/apply" '{
  "change_note": "rollback demo policy to original"
}' | pretty

section "Policy revisions"
api GET "/v1/routing-policies/$default_policy_id/revisions" | pretty

if [[ "${RUN_PROVIDER_CALL:-0}" == "1" ]]; then
  section "Create demo user for one real provider call"
  api POST /v1/users "{
    \"user_id\": \"$DEMO_ID-user\"
  }" >/dev/null 2>&1 || true

  section "Send one real default_routing chat completion"
  api POST /v1/chat/completions '{
    "model": "default_routing",
    "project_id": "'"$DEMO_ID"'",
    "tags": {"tenant": "standard"},
    "user": "'"$DEMO_ID-user"'",
    "messages": [{"role": "user", "content": "Reply with one sentence."}],
    "max_tokens": 80
  }' | pretty
else
  section "Skipping real provider call"
  echo "Set RUN_PROVIDER_CALL=1 in .env to generate route traces with a live provider request."
fi

section "Route trace summary"
api GET /v1/route-traces/summary | pretty

section "Done"
cat <<EOF
Default policy: $default_policy_id
Canary policy:  $canary_policy_id
Draft policy:   $draft_policy_id
Project:        $DEMO_ID
Gateway:        $GATEWAY_URL
EOF
