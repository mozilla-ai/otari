# Demo: shared agent memory, from the terminal

About 12 minutes. A local model answers, the terminal shows each step, and the dashboard holds the switches.

**The message in one sentence:** a team's agents share what they learn, with any model, encrypted, under the
organization's control, and we can measure whether it pays off.

## Before you start (10 minutes ahead)

1. The gateway is up (`../start.sh`) and seeded (`../seed.sh`), with Ollama running.
2. Prepare the state. It takes about 3 minutes and can always be run again:
   ```bash
   ./prepare.sh
   ```
   It erases memory, seeds 5 entries Alice's and Bob's agents "learned", warms the models, and runs a savings
   experiment of 50 sessions so that step 8 has data.
3. A browser at http://127.0.0.1:8010, signed in with `OTARI_MASTER_KEY` (in `../.env`). Keep two tabs open:
   - **Workspaces → Default workspace → Memory**
   - **Organization → General**
4. A large terminal beside the browser, 16 pt or larger.
5. Check that "Recall automatically" is **off** and the organization's "Agent memory" is **on**. `prepare.sh`
   leaves them so.

## The script

Run `./run.sh`. Each step waits for Enter.

### 1 · What the team's agents already know (1 min)
- **On screen:** Bob searches the shared memory and finds 4 entries that Alice's and Bob's agents wrote last
  week.
- **Say:** "Today every agent session starts from nothing. What one finds out (how we deploy, why something
  fails) is lost, and another rediscovers it, paying in tokens. This is memory per workspace, shared by its
  members' agents."

### 2 · Alice's agent learns something new (1.5 min)
- **On screen:** Alice proposes "the deploy fails with *tag not found* unless `git push --tags` ran first". The
  proposal is `pending`, and seconds later it shows as **v2**: it replaced the earlier deploy entry.
- **Say:** "An agent doesn't write to memory; it proposes. A consolidator screens out secrets and personal
  data, compares it with what memory holds, and decides to add, update, change nothing, or flag a conflict.
  Here it updated, and the earlier version is kept."

### 3 · What the database actually holds (1 min)
- **On screen:** the content shows as `mem1.…`, encrypted. The app's database role sees 0 rows unless it names
  the workspace, and 6 when it does.
- **Say:** "Each workspace has its own data key, AES-256-GCM, rotated every 30 days. Postgres row-level
  security means a bug in a query filter can't return another workspace's rows. And deleting a workspace
  deletes its keys: what's left in a backup can no longer be read."

### 4 · Bob, on another model, without memory (1 min)
- **On screen:** qwen 3B, local, through Otari, gives a generic answer ("check that the tag exists…").
- **Do:** when the script asks, open the **Memory** tab and turn on **Recall automatically**. Show the
  settings that appear (similarity and the most entries). The script notices the change by itself.

### 5 · The same question, with automatic recall (1 min)
- **On screen:** "run `git push --tags` before the deploy", with the header `otari-memory-recalled: 1`.
- **Say:** "Another model, another provider, and it knows what Alice's agent learned. No tool and no extra
  turn: Otari searches before calling the model and adds what's relevant to the message, marked as data, not
  as instructions."
- **The key moment:** let the difference between steps 4 and 5 land.

### 6 · What it refuses, and what it keeps private (1 min)
- **On screen:** a proposal holding a GitHub token is refused. Alice's private note doesn't show in Bob's
  search, but Alice finds it.
- **Say:** "There's shared memory and each member's own. A member's own memory is seen by no one else, not
  even an admin."

### 7 · One switch for the whole organization (1 min)
- **Do:** when the script asks, turn off "Agent memory" in **Organization → General** and confirm. Then turn
  it back on.
- **On screen:** memory is off for every workspace at once. The model is back to the generic answer, and
  nothing was deleted.
- **Say:** "The organization decides first, then each workspace. All of it from the dashboard."

### 8 · Does it pay off? Measure it (2 min)
- **On screen:** the 50 sessions of the experiment: control against memory, tokens per session and the
  interval. Show it in **Memory → Measure what memory saves** too.
- **Say, honestly:** "On one-off questions memory **costs** tokens, more than double (read the exact figure on
  screen; it changes with each preparation): it adds context, and there is nothing to rediscover. What it
  buys is in steps 4 and 5: the right answer. Tokens are saved in agent sessions that, without memory, would
  explore for several turns. That's why we built the experiment: it keeps a share of real sessions without
  memory and compares."
- If someone asks why no saving shows: the experiment measures cost, not quality. Real agent traffic (Claude
  Code, Codex) is where the saving is measured.

## If something goes wrong

| Problem | What to do |
|---|---|
| Ollama is slow or answers oddly | `./run.sh 5` repeats from step 5. The model is warm after `prepare.sh`. |
| The dashboard doesn't respond | `DEMO_AUTO=1 ./run.sh 4` does through the API what you would do in the dashboard. |
| You skipped a step or want to repeat one | `./run.sh N` starts at step N. |
| The state looks wrong | `./prepare.sh` restores it in about 3 minutes (`--quick`: 20 s, no new experiment). |

## Likely questions

- **Who sees the plain text?** The gateway, while it uses it, like any prompt that goes through Otari. At rest
  it's encrypted with per-workspace keys. Customer-held keys (HYOK) are planned.
- **What about prompt injection through memory?** Everything recalled is marked as data from other agents.
  The consolidator only answers a decision and an index, never text, and an entry can't close the
  `<otari_memory>` block. The organization's mandated guardrails also check what agents propose.
- **Does it work with Claude Code?** Yes, as an MCP server (`/api/v1/memory/mcp`) with the same tools. If its
  traffic also goes through Otari, it gets automatic recall too.
- **What does it cost?** One embedding per message, plus a few hundred tokens only when it finds something.
  Memory's own calls are metered as the workspace's usage.
- **What if memory fails?** The request goes on without it; it never fails a request or waits more than 3 s.

## Optional extra: Claude Code, live

With the memory MCP server added to Claude Code (`claude mcp add --transport http otari-memory
http://127.0.0.1:8010/api/v1/memory/mcp --header "Authorization: Bearer $ALICE_KEY"`, the key from
`../keys.env`), open a session and ask: *"Search the shared memory for what we know about staging deploys"*.
It calls `memory_search` and answers with the same entries.

## Other scripts here

- `converse.sh "earlier turn" ... "newest turn"`: one conversation as Alice, with what recall handed over.
- `context_check.sh`: what recall hands over for a follow-up, with and without the earlier turns.
- `chat_check.py WHO "message"`: one streamed turn as Open WebUI sends it, and what memory did.
