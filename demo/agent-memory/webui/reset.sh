#!/usr/bin/env bash
# Puts the Open WebUI demo back where it starts: the demo's memory, recall on, and memory offered to every chat,
# so an assistant can save what a person asks it to remember.
#   reset.sh          memory only; the chats in Open WebUI stay
#   reset.sh --chats  memory and every chat of both users
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(cd "$HERE/../visual" && python3 -c "import server; server.seed()")
bash -c "source '$HERE/../terminal/lib.sh'; switches '{\"auto_recall\": true, \"offer_to_every_request\": true, \"recall_rerank\": true}'"
# The organization's prompt-injection guardrail starts paused: step 7 turns it on from the dashboard.
bash -c "source '$HERE/../terminal/lib.sh'; for id in \$(admin GET /organizations/me/guardrails | python3 -c 'import json,sys; [print(g[\"id\"]) for g in json.load(sys.stdin)[\"data\"]]'); do admin PATCH /organizations/me/guardrails/\$id '{\"enabled\": false}' >/dev/null; done"
if [ "${1:-}" = "--chats" ]; then
  "$HERE/up.sh" down
  docker volume rm openwebui-alice openwebui-bob >/dev/null
  "$HERE/up.sh"
fi
echo "ready: Alice http://127.0.0.1:3001, Bob http://127.0.0.1:3002"
