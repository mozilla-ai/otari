#!/usr/bin/env bash
# Calls one memory MCP tool as ALICE or BOB: mcp.sh ALICE memory_search '{"query":"deploy"}'
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/keys.env"
KEY_VAR="$1_KEY"
METHOD=tools/call
if [ "$2" = list ]; then METHOD=tools/list; PARAMS='{}'; else PARAMS="{\"name\":\"$2\",\"arguments\":${3:-{\}}}"; fi
curl -s http://127.0.0.1:8010/api/v1/memory/mcp \
  -H "Authorization: Bearer ${!KEY_VAR}" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"$METHOD\",\"params\":$PARAMS}" | python3 -m json.tool
