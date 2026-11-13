#!/usr/bin/env bash
# One conversation through the gateway as Alice: what automatic recall handed over, and the model's answer.
#   converse.sh "an earlier turn" ... "the newest turn"
source "$(dirname "$0")/lib.sh"
[ $# -ge 1 ] || { echo "usage: $0 \"earlier turn\" ... \"newest turn\""; exit 2; }

headers=$(mktemp)
curl -s -D "$headers" -m 170 "$API/chat/completions" -H "Authorization: Bearer $ALICE_KEY" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "max_tokens": 150,
    "messages": [{"role": "user", "content": turn} for turn in sys.argv[2:]]}))' "$MODEL" "$@")" |
  python3 "$DEMO_DIR/show.py" chat
recalled=$(grep -i '^otari-memory-recalled' "$headers" | tr -d '\r' | awk '{print $2}')
rm -f "$headers"
printf '%srecalled: %s%s\n' "$DIM" "${recalled:-memory not involved}" "$OFF"
[ -n "$recalled" ] && [ "$recalled" != 0 ] || exit 0
pg_super "select json_array_elements_text(entry_ids::json) from (select entry_ids from memory_reads
    where tool = 'auto_recall' order by read_at desc limit 1) last_read" | while read -r id; do
  tool ALICE memory_get "{\"id\":\"$id\"}" | python3 -c 'import json,sys; print("  ->", json.loads(sys.stdin.read().strip())["entry"]["text"])'
done
