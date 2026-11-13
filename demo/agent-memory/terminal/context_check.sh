#!/usr/bin/env bash
# Which entries automatic recall hands over for the Alejandro example, with and without the earlier turns.
source "$(dirname "$0")/lib.sh"

converse() { # label, then the conversation's user turns, the newest last
  local label="$1"; shift
  local headers; headers=$(mktemp)
  curl -s -D "$headers" -o /dev/null -m 170 "$API/chat/completions" -H "Authorization: Bearer $ALICE_KEY" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 5,
      "messages": [{"role": "user", "content": turn} for turn in sys.argv[2:]]}))' "$MODEL" "$@")"
  heading "$label"
  local recalled; recalled=$(grep -i '^otari-memory-recalled' "$headers" | tr -d '\r' | awk '{print $2}')
  rm -f "$headers"
  printf '  newest turn: %s\n  recalled: %s\n' "${!#}" "$recalled"
  [ "$recalled" = 0 ] && return 0
  pg_super "select json_array_elements_text(entry_ids::json) from (select entry_ids from memory_reads
      where tool = 'auto_recall' order by read_at desc limit 1) last_read" | while read -r id; do
      [ -n "$id" ] && tool ALICE memory_get "{\"id\":\"$id\"}" | head -1
    done
}

converse "Follow-up alone" "What did we decide in our last meeting?"
converse "Follow-up in its conversation" \
  "I've been talking with Alejandro about how agents could share memory across models and sessions." \
  "We compared keeping it in the gateway versus a separate service." \
  "What did we decide in our last meeting?"
converse "Naming Alejandro" \
  "I've been talking with Alejandro about how agents could share memory across models and sessions." \
  "What did Alejandro and I agree in our last meeting?"
converse "Changing the subject" \
  "I've been talking with Alejandro about how agents could share memory across models and sessions." \
  "Unrelated: our staging deploy failed with tag not found, what do I do?"
converse "Asking about the person" "I'm buying a birthday present for Alejandro." "What does Alejandro do for fun?"
