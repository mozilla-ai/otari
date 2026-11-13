#!/usr/bin/env bash
# The agent memory demo, one step per keypress. Run prepare.sh first.
#   run.sh        from the start
#   run.sh 4      from step 4
source "$(dirname "$0")/lib.sh"
FROM="${1:-1}"
step() { [ "$1" -ge "$FROM" ]; }

if step 1; then
  heading "1 · What the team's agents already know"
  say "Bob's agent searches the shared memory. Alice's and Bob's agents wrote these last week."
  tool BOB memory_search '{"query":"how do we deploy to staging","limit":5}'
  pause
fi

if step 2; then
  heading "2 · Alice's agent learns something new"
  say "Alice hit a failure on her staging deploy, and her agent proposes what it learned."
  tool ALICE memory_propose '{"kind":"procedure","scope":"workspace","text":"A staging deploy fails with tag not found unless the release tag was pushed first: run git push --tags before make deploy-staging."}'
  say "The consolidator screens it, embeds it, compares it with what memory holds, and decides..."
  wait_consolidated
  tool ALICE memory_search '{"query":"staging deploy tag not found","limit":2}'
  pause
fi

if step 3; then
  heading "3 · What the database actually holds"
  say "Every entry is sealed with its workspace's own data key (AES-256-GCM):"
  pg_super "select kind, scope, left(content_sealed, 48) || '…' from memory_entries where status = 'active' limit 3" | sed 's/^/  /'
  say "And the app's own database role sees no rows unless it names the workspace (row-level security):"
  printf '  without the workspace: %s rows\n' "$(pg_app "select count(*) from memory_entries")"
  printf '  naming the workspace:  %s rows\n' "$(pg_app "begin; select set_config('otari.memory_workspace', '$WS', true); select count(*) from memory_entries; commit;" | sed -n 3p)"
  pause
fi

if step 4; then
  heading "4 · Bob, on another model, without memory"
  say "Bob runs a small local model (qwen2.5 3B via Ollama) through Otari. Memory declined for this request:"
  ask "Our staging deploy failed with 'tag not found'. What should I do? Answer in two sentences." -H 'Otari-Memory: off'
  in_dashboard 'Now turn on "Recall automatically" in the dashboard: Workspaces → Default workspace → Memory.' \
    "switches '{\"auto_recall\": true}'"
  until current | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["auto_recall"] else 1)'; do sleep 1; done
  say "Recall is on."
  pause
fi

if step 5; then
  heading "5 · The same question, with automatic recall"
  say "No tool, no extra round: Otari adds what memory holds to Bob's message before the model sees it."
  ask "Our staging deploy failed with 'tag not found'. What should I do? Answer in two sentences."
  pause
fi

if step 6; then
  heading "6 · What memory refuses, and what it keeps private"
  say "An agent tries to save a credential:"
  tool BOB memory_propose '{"kind":"fact","text":"The staging token is ghp_abcdefghijklmnopqrstuvwxyz0123456789"}'
  say "Alice keeps notes of her own. Bob searching for them finds none:"
  tool BOB memory_search '{"query":"which region is my staging dashboard pinned to","limit":3}' | grep -i 'eu-west-1' || printf '  %s(not visible to Bob)%s\n' "$GREEN" "$OFF"
  say "Alice finds hers:"
  tool ALICE memory_search '{"query":"which region is my staging dashboard pinned to","limit":1}'
  pause
fi

if step 7; then
  heading "7 · One switch for the whole organization"
  in_dashboard 'Turn off "Agent memory" in the dashboard: Organization → General.' \
    "admin PUT /organizations/me/memory '{\"enabled\": false}'"
  until ! admin GET /organizations/me/memory | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["enabled"] else 1)'; do sleep 1; done
  say "Memory is off for every workspace at once, and nothing was deleted:"
  tool BOB memory_search '{"query":"staging deploy"}'
  ask "Our staging deploy failed with 'tag not found'. What should I do? Answer in one sentence."
  in_dashboard "Turn it back on." "admin PUT /organizations/me/memory '{\"enabled\": true}'"
  until admin GET /organizations/me/memory | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["enabled"] else 1)'; do sleep 1; done
  say "Back on."
  pause
fi

if step 8; then
  heading "8 · Does it pay off? Measure it"
  say "Open the Memory tab: 'Measure what memory saves'. prepare.sh ran 50 sessions, half without memory."
  admin GET "/workspaces/$WS/memory/experiment" | python3 "$DEMO_DIR/show.py" experiment
  say "On one-off questions memory costs tokens: it adds context, and there is nothing to rediscover."
  say "What it bought is in steps 4 and 5: the right answer instead of a generic one. Tokens saved come from"
  say "agent sessions that would otherwise explore; this experiment is how we measure that on real traffic."
fi

heading "End"
