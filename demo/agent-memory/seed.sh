#!/usr/bin/env bash
# Seeds the demo once the gateway is up (./start.sh): the models' providers, two people with keys, memory on,
# and a prompt-injection guardrail the organization mandates. Safe to run again.
# Writes keys.env: Alice's and Bob's keys, and the workspace the demo uses.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; . "$HERE/.env"; set +a
API=http://127.0.0.1:8010/api/v1
ALICE=11111111-1111-4111-8111-111111111111
BOB=22222222-2222-4222-8222-222222222222

curl -sf "$API/health" >/dev/null || { echo "the gateway is down: run ./start.sh first" >&2; exit 1; }

# method path [body]: prints the status. The body goes through stdin, so no key shows on a command line.
call() {
  local status
  status=$(printf '%s' "${3:-}" | curl -s -o "$HERE/.last.json" -w '%{http_code}' -X "$1" "$API$2" \
    -H "Authorization: Bearer $OTARI_MASTER_KEY" -H 'Content-Type: application/json' ${3:+--data-binary @-})
  echo "$1 $2 -> $status"
}
field() { python3 -c 'import json,sys; print(eval("d" + sys.argv[2], {"d": json.load(open(sys.argv[1]))}))' "$HERE/.last.json" "$1"; }
sql() { docker exec otari-memory-pg psql -v ON_ERROR_STOP=1 -U otari_app -d otari_memory -Atc "$1"; }

echo "== the organization and its default workspace"
call GET /workspaces >/dev/null
WS=$(field '["data"][0]["id"]')
ORG=$(field '["data"][0]["organization_id"]')
echo "workspace $WS"

echo "== people: Alice (workspace admin) and Bob (member)"
sql "insert into \"user\" (id, email, is_active, is_superuser, full_name, active_organization_id)
     values ('$ALICE','alice@local.test',true,false,'Alice','$ORG'),('$BOB','bob@local.test',true,false,'Bob','$ORG')
     on conflict (id) do nothing;
     insert into organization_member (id, organization_id, user_id, role, status)
     select gen_random_uuid(), '$ORG', u, 'member', 'active' from unnest(array['$ALICE','$BOB']::uuid[]) u
     where not exists (select 1 from organization_member m where m.user_id = u and m.organization_id = '$ORG');
     insert into workspace_member (id, workspace_id, user_id, role, status)
     select gen_random_uuid(), '$WS', u, r, 'active' from (values ('$ALICE'::uuid,'admin'),('$BOB'::uuid,'member')) v(u, r)
     where not exists (select 1 from workspace_member m where m.user_id = u and m.workspace_id = '$WS');" >/dev/null

# provider model price: offers the model at that price per million tokens, in and out. A provider lists the
# models it finds when it is added, unpriced, so one already offered is repriced instead.
offer() {
  local price="\"input_price_per_million\":$3,\"output_price_per_million\":$3" status
  status=$(call POST "/hosted-providers/$1/models" "{\"model\":\"$2\",$price}")
  if [ "${status##* }" != 409 ]; then
    echo "$status"; return 0
  fi
  call GET "/hosted-providers/$1/models" >/dev/null
  local id; id=$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1])); models = d.get("data", d) if isinstance(d, dict) else d
print(next(m["id"] for m in models if m["model"] == sys.argv[2]))' "$HERE/.last.json" "$2")
  call PATCH "/hosted-providers/$1/models/$id" "{$price}" >/dev/null && echo "$1 $2 repriced"
}

echo "== providers (409: already there)"
call POST /hosted-providers '{"provider":"ollama","api_base":"http://127.0.0.1:11434","serves_unlisted_models":false}'
# Prices are nominal: the savings experiment needs a price to count cost by.
for model in qwen2.5:3b qwen2.5:7b nomic-embed-text; do offer ollama "$model" 0.01; done
if [ -n "${FIREWORKS_API_KEY:-}" ] && [ "$FIREWORKS_API_KEY" != REPLACE_ME ]; then
  call POST /hosted-providers "$(python3 -c 'import json,os; print(json.dumps({"provider": "fireworks", "api_key": os.environ["FIREWORKS_API_KEY"], "enabled": True, "serves_unlisted_models": True}))')"
  offer fireworks accounts/fireworks/models/qwen3-reranker-8b 0.001
else
  echo "no Fireworks key: skipped"
fi

echo "== memory on"
call PUT /organizations/me/memory '{"enabled":true}'
call PUT "/workspaces/$WS/memory" '{"enabled":true,"workspace_scope_enabled":true,"write_policy":"auto","offer_to_every_request":false,"hide_external":false,"retention_days":null}'

echo "== the organization's prompt-injection guardrail"
call GET /organizations/me/guardrails >/dev/null
if [ "$(field '["count"] if "count" in d else len(d["data"])')" = 0 ]; then
  call POST /organizations/me/guardrails '{"profile":"prompt-injection","mode":"block","applies_to_all_workspaces":true}'
else
  echo "already mandated"
fi

echo "== keys"
touch "$HERE/keys.env"; chmod 600 "$HERE/keys.env"
. "$HERE/keys.env"
{
  for who in ALICE BOB; do
    key_var="${who}_KEY"
    if [ -z "${!key_var:-}" ]; then
      call POST /keys "{\"key_name\":\"memory-demo-$who\",\"user_id\":\"${!who}\",\"workspace_id\":\"$WS\"}" >&2
      printf '%s_KEY=%s\n' "$who" "$(field '.get("key") or d.get("api_key") or d.get("secret")')"
    else
      printf '%s_KEY=%s\n' "$who" "${!key_var}"
    fi
  done
  printf 'OTARI_DEMO_WORKSPACE=%s\n' "$WS"
} >"$HERE/keys.env.new"
mv "$HERE/keys.env.new" "$HERE/keys.env"; chmod 600 "$HERE/keys.env"
rm -f "$HERE/.last.json"
echo "keys.env written"

echo
echo "Next: terminal/prepare.sh --quick, then pick a demo in README.md."
