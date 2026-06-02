# Board Demo Runbook — "The Research Assistant that grows up"

A 10–12 minute live demo for the board. One tiny app, unchanged throughout.
We change only *what it points at* and watch it gain capabilities, privacy,
and governance. The arc moves from **open-source Otari** to **Otari.ai**.

> **The single message:** *"We didn't give up capabilities to go open. We
> gained governance, privacy, and cost control on top of them."*

---

## What's in this folder

| File | Role |
|------|------|
| `app.sh` | The **constant**. A ~tiny "Research Assistant" that calls `/v1/chat/completions`. Never edited during the demo. |
| `demo_flow.sh` | The **orchestrator** for all four acts. Paced with "press Enter" beats; pauses mid-way for the platform-mode restart, then runs Act 4 itself. |
| `start.sh` / `stop.sh` | Bring the OSS (standalone) stack up/down (gateway + sandbox + searxng + postgres). |
| `start-platform.sh` | Bring the stack up in **platform mode** (gateway connected to Otari.ai via `OTARI_AI_TOKEN`). |
| `gateway-config.yml` | OSS standalone gateway config (local llamafile + optional frontier providers). |
| `gateway-config.platform.yml` | **Providerless** config for platform mode — platform mode forbids local providers, so Act 4 mounts this one. |
| `docker-compose.standalone.yml` | Override used by `start.sh` to blank `OTARI_AI_TOKEN`, so the gateway stays standalone in Acts 1–3 even though the token is in `.env`. |
| `docker-compose.platform.yml` | Override used by `start-platform.sh` to inject `OTARI_AI_TOKEN` (platform mode). Keeps platform config out of the shared `docker-compose.yml`. |
| `.env.example` | Copy to `.env`. Keys are **optional** for Acts 1–3; Act 4 needs `OTARI_AI_TOKEN` + `OTARI_USER_TOKEN` + `OTARI_MODEL`. |

---

## Pre-flight (do this BEFORE the board walks in)

1. **Config**
   ```bash
   cd otari/demo/board
   cp .env.example .env
   # for the Act-1 frontier contrast, set your OpenAI key (model defaults to gpt-5.5):
   #   echo 'OPENAI_API_KEY=sk-...' >> .env
   # optional: set LLAMAFILE_BIN to auto-start a local model, e.g.
   #   curl -L -o /tmp/qwen3-4b.llamafile \
   #     https://huggingface.co/mozilla-ai/Qwen3-4B-llamafile/resolve/main/Qwen_Qwen3-4B-Q5_K_M.llamafile
   #   chmod +x /tmp/qwen3-4b.llamafile
   #   echo 'LLAMAFILE_BIN=/tmp/qwen3-4b.llamafile' >> .env
   ```

2. **Bring the OSS stack up** (terminal 1, leave it running):
   ```bash
   ./start.sh
   ```
   Wait for the gateway to report healthy. Confirm:
   ```bash
   curl -sf http://localhost:8088/health && echo OK
   ```

3. **Warm the model.** Run the flow once with `--fast` so the llamafile is
   loaded and the first (slow) inference is already paid for:
   ```bash
   ./demo_flow.sh --fast
   ```
   This warms Acts 1–3 (web_search + code_execution fire on the local model).
   When it reaches the Act 4 restart pause, **Ctrl-C** — you'll do the real
   platform-mode restart during the live run.
   This also confirms web_search and code_execution actually fire on your
   machine before you're in front of the board.

4. **Otari.ai (Act 4) tab.** Log in to the hosted platform in a browser, on an
   org that has: a **wallet with balance**, the **Mozilla.ai provider**
   enabled, a **frontier managed provider** available, at least one **budget**,
   and a **routing policy**. Have these tabs pre-opened (see Act 4 below).

5. **Terminal size.** Big font. The tool-call lines (`▶ web_search`,
   `▶ code_execution`) are the visual payoff — they need to be readable.

---

## Running it live

Terminal 2:
```bash
./demo_flow.sh
```
Press Enter to advance between beats. The script prints the talk-track for each
act; the notes below are *your* lines on top of what's on screen.

---

### Act 1 — Frontier vs. open, side by side  (~3 min)

**On screen:** two questions, each asked **twice** — first to OpenAI's frontier
model with its *native hosted tools*, then to a bare local open-weights model
with no tools. The contrast is the whole point.

| Question | Frontier (OpenAI Responses API) | Open-weights (local, no tools) |
|----------|----------------------------------|--------------------------------|
| **Q1 — current info** ("who won the most recent EuroLeague title + final date") | `tools: [{ type: web_search }]` → reaches out, gets the live answer | answers from stale training memory (or admits it can't know) |
| **Q2 — computation** (Python 3.14 EOL + days remaining) | `tools: [{ type: shell, … }]` → runs code, computes exactly | does the date math in its head — untrustworthy |

These are OpenAI's **own** hosted tools (`web_search`, `shell`), called directly
against `api.openai.com/v1/responses` — *not* through Otari. That's deliberate:
we're showing what the closed provider gives you out of the box.

**Your lines:**
> "Here's why teams pay for the frontier: the model doesn't just answer, it
> *acts*. It searches the live web, it runs a shell. Those tools ship with the
> closed provider."
>
> *(then the local model)* "Now the open-weights model everyone says we should
> switch to. Same questions — but it's stuck in the past and does math in its
> head. **This is the fear: go open, lose your tools.** Hold that thought."

**Setup note:** the frontier half needs `OPENAI_API_KEY` in `.env` (model via
`OPENAI_MODEL`, default `gpt-5.5`). Without a key, those legs print a clear
"skipped" line and the local legs still run.

---

### Act 2 — Same app, same model, now through Otari  (~3 min)  ⭐ the moment

**On screen:** two calls, because the gateway runs **one server-side tool per
request** today (combining them is a planned refinement). Each call adds *one*
line to the request:

1. `tools: [{ type: otari_web_search }]` → the most recent EuroLeague champion + final date.
2. `tools: [{ type: otari_code_execution }]` → Python 3.14's EOL date and exact
   days-remaining.

Two independent capabilities: live knowledge, then exact computation.

> **Why the `otari_` prefix?** It's what tells the gateway to run the tool
> *server-side*. The bare `web_search` / `code_execution` keywords now pass
> *through* to the upstream provider to run natively — so for gateway-managed
> execution (our story here) we use `otari_web_search` / `otari_code_execution`.

**Each call prints the full round-trip in three numbered sections**, so the
board can see the tool was genuinely invoked — not faked by the model:

| Section | What it shows |
|---------|---------------|
| **1) REQUEST** | The exact JSON the app POSTs — note the single `tools` entry. |
| **2) RESPONSE** | The tool call the model emitted (`▶ web_search` / `▶ code_execution`) and the final answer. |
| **3) TOOL ran server-side** | New log lines from the **searxng** / **sandbox** container — proof the gateway actually executed the tool. |

**Your line over section 3:**
> "That's the search engine — and the sandbox — logging the request the
> *gateway* made on the model's behalf. The tool really ran; the model didn't
> make it up."

**Your lines:**
> "The app didn't change. We didn't write a search integration or a sandbox.
> We added one line, pointed at Otari, and the gateway ran the tool
> server-side, on our infrastructure. Swap the line — different superpower."
>
> "The open-weights model just matched the frontier experience. **That's the
> whole thesis: open didn't cost us capability — Otari gave it back.**"

> If asked "why two calls?": today the gateway dispatches one server-side tool
> backend per request; multi-tool dispatch is on the roadmap. The point — *the
> app gains capabilities with no code* — holds either way.

**The privacy beat (say it, it lands with a board):**
> "And notice — that model and that gateway both ran *here*. The prompt never
> left the building. When you self-host Otari, your prompts are yours. No
> third party sees them. For regulated or sensitive workloads, that's a real
> boundary."

---

### Act 3 — Now scale it to a team of 50  (~1 min, the bridge)

**On screen:** the honest framing — the OSS gateway *already* does budgets,
rate limits, virtual keys, and usage/cost tracking via its API. The gap is the
*team operational layer*: shared master key (no SSO/roles/org isolation),
provider keys in env files, no dashboards, no managed providers, hand-edited
config.

**Your line (don't undersell the OSS engine):**
> "To be clear — the open-source gateway already does budgets, rate limits,
> keys, and cost tracking. That's not the gap. The gap is running it for *50
> people*: single sign-on, roles, dashboards finance can read, keys that
> aren't in a file on a box, models without everyone bringing a key. The
> engine is necessary but not sufficient. That's the platform."

→ **Connect the gateway to Otari.ai now** (next section).

---

### Act 4 — Connect to Otari.ai, re-run on a stronger model  (~4 min)

Act 4 is **live**: we connect the local gateway to Otari.ai with a gateway
token, restart it in platform mode, and re-run the exact same flow on a much
stronger model served through **Mozilla.ai Inference** — no provider key in the
request.

**One-time setup (before the meeting):** platform mode uses **two** tokens.
1. Log in to Otari.ai. Create a **gateway** for your org (or open an existing one).
2. **Mint the gateway token** (server-to-server) and copy it.
3. **Mint a project/API token** (the per-request client bearer) and copy it.
4. In `demo/board/.env` set:
   - `OTARI_AI_TOKEN=otari_…` — the **gateway** token
   - `OTARI_USER_TOKEN=otari_…` — the **project/API** token (per-request bearer)
   - `OTARI_MODEL=mzai:<model>` — provider-prefixed, e.g. `mzai:deepseek-ai/DeepSeek-V4-Pro`
5. Point the gateway at the platform via `PLATFORM_BASE_URL`:
   - **Local platform** (repo-root docker-compose, nginx on host `:8100`): leave
     it unset — `docker-compose.platform.yml` defaults to
     `http://host.docker.internal:8100/api/v1`. Make sure the platform stack is
     up (`docker compose up` at the repo root). **Never use `localhost`** — inside
     the gateway container that's the container itself, which is why a `localhost`
     base URL yields `{"detail":"Authorization service unavailable"}`.
   - **Hosted platform**: set `PLATFORM_BASE_URL=https://api.otari.ai/api/v1`.
6. Ensure the org has a **wallet with balance**, the **Mozilla.ai provider**
   enabled, a **budget**, and a **routing policy** (for the UI beats).

**Live:** `demo_flow.sh` runs Acts 1–3, then pauses and tells you to restart the
gateway in platform mode. In the **gateway terminal**:
```bash
./stop.sh            # tear down standalone mode
./start-platform.sh  # restart with the token → platform mode
```
Back in the **flow terminal**, press Enter. The flow waits for `/health` to
report `mode=platform`, then runs Q1 (otari_web_search) and Q2
(otari_code_execution) against `OTARI_MODEL` itself — no separate script. Each call shows the same
REQUEST/RESPONSE sections; note there's **no raw provider key** anywhere. Two
Otari.ai tokens do the work: the gateway carries one (server-to-server), the
app sends a project token as its bearer; the platform resolves the actual model
credential and bills the wallet.

**Your line over the calls:**
> "No provider key changed hands. The gateway holds an Otari.ai token, the app
> sends a project token — and the platform resolves the real credential on its
> side. Same app, same tools — now a far stronger model, with no key of our
> own, every token billed and traced. The open path and the managed path are
> the same code."

**Then the cockpit (switch to the browser)** — walk these five beats; click,
don't narrate from memory:

| # | Beat | Where to click | The line |
|---|------|----------------|----------|
| 1 | **Managed providers** | Project → Keys / Providers. Show a **frontier** model and the **Mozilla.ai** open-weights provider, both usable with **no API key in the app**. | "Two model classes, zero keys. Frontier *and* open weights, billed to the org wallet." |
| 2 | **Budgets** | Org/Workspace → budgets. Show a per-provider-key cap. If you have a tripped budget, show the **blocked** request. | "Finance sets the ceiling. Over-limit calls don't bill us — they get refused." |
| 3 | **Observability** | Traces → session explorer. Open the trace from the call you just made. Show tokens + cost per request. | "Every call is attributable — who, what model, how many tokens, what it cost." |
| 4 | **Declarative config** | Org config (YAML import/export). Show the whole org — workspaces, keys, budgets, routing — as one file. | "Governance you can review in a pull request, not click-ops." |
| 5 | **Routing policy** | Routing policies (YAML). Show a rule sending cheap queries to an open model, hard ones to frontier. | "Open-weights as the default, frontier on demand — cost drops, quality holds." |

> If the live platform misbehaves, fall back to screenshots of these five
> screens. Don't debug in front of the board.

---

## The close (~30 sec)

Back to the terminal (the closing line is printed there too):

> **"We didn't give up capabilities to go open. We gained governance, privacy,
> and cost control on top of them."**

Open-source Otari closes the capability gap. Otari.ai adds the controls a
company needs — and makes open-weights a first-class, cost-saving default
rather than a downgrade.

---

## Teardown

```bash
./stop.sh
```
(Auto-started llamafile is killed when `demo_flow.sh` exits.)

---

## If something breaks

| Symptom | Fix |
|---------|-----|
| `Gateway not reachable` | `./start.sh` not running, or port ≠ 8088. Check `.env` `GATEWAY_PORT`. |
| `No llamafile reachable` | Set `LLAMAFILE_BIN` in `.env`, or start a model on `:8080` yourself. |
| Model answers without calling a tool | Small models hedge. The app already injects directive `purpose_hint`s; re-run, or use a frontier model for Act 2 by adding `OPENAI_API_KEY` to `.env` and passing `--model openai:gpt-4o` to `app.sh`. |
| Web search returns nothing | SearXNG cold-start; the `web-search` profile container needs a few seconds after `start.sh`. Re-run the warm-up. |
| Non-main branch, `ModuleNotFoundError` | `start.sh` prints the `docker build -t mzdotai/otari:latest .` recipe — run it once. |
