# Agent memory demo

Runs the gateway from this checkout with agent memory on, two people (Alice and Bob) who share a workspace, and
three ways to show it: two chat windows, a scripted terminal walk-through, and a side-by-side page. See
[Agent memory](../../docs/memory.md) for what memory does.

| What | Where |
| --- | --- |
| Gateway and dashboard | http://127.0.0.1:8010 (sign in with `OTARI_MASTER_KEY` from `.env`) |
| Postgres (`otari-memory-pg` container) | 127.0.0.1:5433 |
| Guardrails service (PIGuard) | 127.0.0.1:8183, 8184 |
| Open WebUI: Alice, Bob | http://127.0.0.1:3001, http://127.0.0.1:3002 |
| Side-by-side page | http://127.0.0.1:8020 |

## Requirements

- Docker
- [uv](https://docs.astral.sh/uv/) 0.13, which the repo requires (`uv self update 0.13.8`), and Python 3.11 or later
- [Ollama](https://ollama.com), running
- pnpm, to build the dashboard
- A [Fireworks](https://fireworks.ai) API key (or see [Without Fireworks](#without-fireworks))

## Set up, once

```bash
cd demo/agent-memory
cp .env.example .env        # and fill in FIREWORKS_API_KEY
./setup.sh                  # the gateway's secrets, Postgres, Ollama models, the dashboard
./start.sh                  # the guardrails service, then the gateway, in the foreground
```

In another terminal, once the gateway answers:

```bash
./seed.sh                   # providers, Alice and Bob with keys, memory on, the org's guardrail
terminal/prepare.sh --quick # the demo's memory
```

`.env` (your key and the gateway's secrets) and `keys.env` (Alice's and Bob's keys) are gitignored. Never
commit or paste them.

From then on, `./start.sh` is enough; `./stop.sh` stops the containers and keeps their data.

## Pick a demo

- **Two chat windows**, as anyone would use it: Alice teaches her assistant, Bob's assistant knows.
  `webui/up.sh`, `webui/reset.sh --chats`, then follow [webui/DEMO.md](webui/DEMO.md).
- **Terminal walk-through**, one step per keypress, with encryption, row-level security and the savings
  experiment: `terminal/prepare.sh` (about 3 minutes), then follow [terminal/DEMO.md](terminal/DEMO.md).
- **Side-by-side page**, the same question with memory and without: `python3 visual/server.py --seed` once,
  then `python3 visual/server.py` and open http://127.0.0.1:8020.

To call a memory tool directly: `./mcp.sh ALICE memory_search '{"query":"staging deploy"}'`, or
`./mcp.sh BOB list`.

The [benches](bench/README.md) measure recall and consolidation on a bigger memory.

## Dashboard: where memory lives

- **Organization → General → Agent memory**: one switch for every workspace.
- **Organization → Guardrails**: the mandated prompt-injection guardrail, and **Also check what agents save
  to agent memory**.
- **Workspaces → Default workspace → Memory**: the workspace's switches (automatic recall, relevance check,
  write policy, retention), the entries, **Waiting for review**, and **Measure what memory saves**.

## Without Fireworks

Leave `FIREWORKS_API_KEY` as it is; `seed.sh` skips the provider. In `otari-config.yml`, set
`memory_consolidation_model: "ollama:qwen2.5:7b"` (`ollama pull qwen2.5:7b`) and remove
`memory_rerank_model`. The terminal demo and the side-by-side page answer with local qwen2.5 3B and work as is.
The Open WebUI demo needs a model that calls tools well: set `MODEL` in `webui/up.sh` to one your providers
serve.

## Start over

```bash
./stop.sh
docker rm otari-memory-pg && docker volume rm otari-memory-pg
docker volume rm openwebui-alice openwebui-bob
rm keys.env
./setup.sh
```
