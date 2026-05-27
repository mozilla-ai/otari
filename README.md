# otari gateway

[![Tests](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-tests.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-tests.yml)
[![Lint](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-lint.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-lint.yml)
[![Typecheck](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-typecheck.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-typecheck.yml)
[![Docker](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-docker.yml/badge.svg)](https://github.com/mozilla-ai/gateway/actions/workflows/gateway-docker.yml)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)

OpenAI-compatible LLM gateway with API key management, budget enforcement, and usage tracking.

</div>

## Why gateway?

`gateway` sits between your applications and LLM providers so you can control access, cost, and observability in one place.

- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`)
- Virtual API key management (`/v1/keys`) for safe client access
- User and budget controls (`/v1/users`, `/v1/budgets`)
- Project-scoped routing and budget controls (`/v1/projects`)
- Tag-scoped team/customer-tier budget groups from request `tags`
- Budget alert thresholds, durable alert events, optional webhooks, and background delivery retries before hard caps
- Usage and pricing tracking (`/v1/messages`, `/v1/pricing`)
- Database-backed routing policies with `model: "default_routing"`, passive provider health, and weighted custom scoring
- Eval/benchmark score imports and local generated-eval pipelines for tuning weighted routing candidates
- Merge-style omitted/null/case-insensitive `default_routing` request handling
- Immutable routing policy revision history for auditability
- Draft/active/archived routing policy states plus clone-to-draft rollout flow
- Audited rollback by applying previous routing policy revisions
- Merge-style tag-condition routing and deterministic canary rollout with percentage buckets
- Region-aware model constraints driven by request tags and candidate metadata
- Policy guardrails for routed request DLP, redaction, credential-leak, and prompt-injection checks, including managed presets and external classifier hooks
- Policy context trimming and summarization for long routed conversations before provider dispatch
- Merge-style `default_strategy` policy definitions for fallback, intelligent, and weighted-score routing
- Responses include the canonical served model and execution vendor
- Dry-run route resolution (`/v1/routing/resolve`) and endpoint-aware route traces for inspecting policy decisions
- Project/tag-attributed usage logs and summary rollups for billing and analytics exports
- Self-hosted operator admin dashboard at `/admin` for policies, traces, usage, budget alerts, and revision rollback
- Local routing gateway walkthrough in [`demo/routing-gateway`](demo/routing-gateway)
- Review-ready compatibility map in [`docs/merge-gateway-compatibility.md`](docs/merge-gateway-compatibility.md)
- Health and metrics endpoints (`/health`, optional `/metrics`) including request, token, cost, budget, alert webhook, retry, and dead-letter counters
- Built-in tools dispatched server-side — `code_execution` (sandboxed Python REPL) and `web_search`. See [Built-in tools](#built-in-tools).

## Quickstart

### 1) Install

```bash
uv venv
source .venv/bin/activate
uv sync --dev
```

### 2) Configure

```bash
cp config.example.yml config.yml
```

Edit `config.yml` and set at least:

- `master_key`
- one provider credential in `providers` (for example `openai.api_key`)

### 3) Run

```bash
uv run gateway serve --config config.yml
```

Open API docs at `http://localhost:8000/docs`.

## Start in platform mode

Platform mode is enabled automatically when `OTARI_AI_TOKEN` is set.

1) Export platform env vars:

```bash
export OTARI_AI_TOKEN=gw_xxx
```

2) Start the gateway:

```bash
uv run gateway serve --config config.yml
```

Notes:

- `GATEWAY_MODE` is optional; effective mode is derived from `OTARI_AI_TOKEN`.
- If you explicitly set `GATEWAY_MODE=platform`, startup fails unless `OTARI_AI_TOKEN` is also set.
- In platform mode, local `providers` configuration is not used.
- The gateway/platform wire contract (resolve and usage endpoints, request/response shapes, retry semantics) is documented in [`docs/platform-protocol.md`](docs/platform-protocol.md).

## First request (OpenAI SDK)

On startup, the gateway can bootstrap an API key in logs. Export it as `GATEWAY_API_KEY`, then call the gateway as an OpenAI-compatible server:

```python
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GATEWAY_API_KEY"],
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "Hello from gateway"}],
)

print(response.choices[0].message.content)
```

## Local development

Run with hot reload and `.env`:

```bash
cp .env.example .env
make dev
```

## Tests and checks

```bash
make test
make lint
make typecheck
```

Run a single test file:

```bash
uv run pytest tests/unit/test_gateway_cli.py -v
```

## Docker

The gateway image is published on [Docker Hub](https://hub.docker.com/r/mzdotai/otari).

### Run with docker compose (gateway + PostgreSQL)

```bash
cp config.example.yml config.yml
docker compose up -d
```

### Run with docker only

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/config.yml:/app/config.yml:ro" \
  mzdotai/otari:latest \
  gateway serve --config /app/config.yml
```

Gateway will be available at `http://localhost:8000`.

## Built-in tools

The gateway dispatches a couple of tools server-side so any model — including
open-weight ones — gets parity with what frontier APIs expose as managed
tools. Both are opt-in via the request's `tools` array (matching OpenAI's and
Anthropic's wire shape) and run inside docker-compose profiles so operators
who don't use them don't pull extra images.

### `code_execution` — sandboxed Python REPL

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Compute 23 factorial."}],
  "tools": [{"type": "code_execution"}]
}
```

`code_interpreter` (OpenAI) and `code_execution_<date>` (Anthropic) are
accepted as aliases. Bring up with `docker compose --profile code-exec up`.
See `demo/code-exec/` for a runnable walkthrough.

### `web_search` — current-information search

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "What's the latest stable Python release?"}],
  "tools": [{"type": "web_search"}]
}
```

`web_search_<date>` (Anthropic) is accepted as an alias. Bring up with
`docker compose --profile web-search up`. See `demo/web-search/` for a
runnable walkthrough.

The bundled backend is a SearXNG metasearch container restricted to engines
that don't forbid automated querying (duckduckgo, mojeek, qwant, wikipedia)
— see `scripts/searxng/settings.yml`. Top results are fetched and content
is extracted via trafilatura in-process so the model sees LLM-ready
Markdown, not raw SERP snippets.

**For commercial or production use**, swap the SearXNG container for a
backend that uses a licensed API (Tavily, Brave Search API, Exa, Linkup,
Serper). `WebSearchBackend` is configured purely by URL
(`GATEWAY_WEB_SEARCH_URL`), so any HTTP service that exposes a
SearXNG-compatible `/search?format=json` endpoint is a drop-in replacement
— including thin adapters in front of commercial APIs. Adapters that
already extract content can pass it through on the optional
`extracted_content` result field to bypass the gateway-side extraction.

Per-tool overrides (`max_results`, `allowed_domains`, `blocked_domains`,
`purpose_hint`) live on the tool entry; operator-level env knobs
(`GATEWAY_WEB_SEARCH_ENGINES`, `GATEWAY_WEB_SEARCH_MAX_RESULTS`,
`GATEWAY_WEB_SEARCH_EXTRACT`, `GATEWAY_WEB_SEARCH_PURPOSE_HINT`) live
alongside `GATEWAY_WEB_SEARCH_URL`.

## API surface

- `GET /health`
- `GET /admin`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/embeddings`
- `POST /v1/moderations`
- `GET /v1/models`
- `GET /v1/vendors`
- `POST/GET /v1/keys`
- `POST/GET /v1/users`
- `POST/GET /v1/budgets`
- `GET /v1/budgets/alerts`
- `POST /v1/routing/resolve`
- `POST/GET /v1/routing-policies`
- `POST/GET /v1/projects`
- `GET /v1/route-traces`
- `GET /v1/messages`
- `GET /v1/pricing`

Full schema: `docs/public/openapi.json`

## Useful CLI commands

```bash
uv run gateway init-db --config config.yml
uv run gateway migrate --config config.yml
uv run gateway migrate --config config.yml --revision <rev>
uv run python scripts/generate_openapi.py --check
```

## License

Apache 2.0. See `LICENSE`.
