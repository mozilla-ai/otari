# Deployment

Use this guide after the [Quickstart](quickstart.md). The quickstart gets a standalone Otari running locally and walks through the first authenticated request. This page picks up from there with deployment-specific setup: hybrid mode, optional services, and environment-based configuration.

Otari is distributed as a Docker image on [Docker Hub](https://hub.docker.com/r/mzdotai/otari).

## Prerequisites

- Docker and Docker Compose

## Standalone deployment notes

For the local standalone path, start with the [Quickstart](quickstart.md).

When turning that setup into a longer-lived deployment:

- Set `database_url` to a durable Postgres instance.
- Keep `master_key` for management endpoints, but use generated API keys for application traffic.
- Add `pricing` entries for every model you want budget enforcement on.
- Point container health checks at `/health` and `/health/readiness`.

## Deploy on Render

This repo contains a Render Blueprint ([`render.yaml`](../deploy/render/render.yaml)), an infrastructure-as-code file that defines your stack.

It deploys the published Otari image (`docker.io/mzdotai/otari:0.2.0`) as a free web service alongside a free Render Postgres 16 database.

- On Apply, provide whichever provider credentials you use. Render provisions the web service and Postgres, injects the database’s internal connection string as `OTARI_DATABASE_URL`, generates `OTARI_MASTER_KEY`, sets `PORT` and `OTARI_PORT` to `8000`, enables fail-closed bundled pricing (`OTARI_REQUIRE_PRICING` and `OTARI_DEFAULT_PRICING`), and checks `/health/readiness`.
- On Otari startup, the app runs database migrations and creates the bootstrap API key (printed once in the service logs).

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mozilla-ai/otari&blueprintPath=deploy/render/render.yaml)

If you create the Blueprint from the Dashboard (**New → Blueprint**) instead of the button, set **Blueprint Path** to `deploy/render/render.yaml`.

**Note:** Free instances are intended for evaluation. The web service spins down after 15 minutes without traffic, and the database is limited to 1 GB, expires after 30 days, and has no backups.

The Blueprint, upgrade instructions, and full env table are available at [`deploy/render/`](https://github.com/mozilla-ai/otari/tree/main/deploy/render).

## Deploy on Railway

For a hosted standalone deployment without local setup, use the one-click
[Railway](https://railway.com) template. It stands up two services: Otari
(`docker.io/mzdotai/otari:latest`, target port `8000`, healthcheck `/health`)
and a managed Postgres, wired together with
`OTARI_DATABASE_URL=${{Postgres.DATABASE_URL}}`.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/otari-railway-template-demo)

You set at least one provider key (the form prompts for `OPENAI_API_KEY`; add a
variable like `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, or `GEMINI_API_KEY` to use
another provider). The master key is auto-generated and `OTARI_REQUIRE_PRICING=false`
is pre-set so an env-only deploy is usable out of the box. The template
definition lives in
[`deploy/railway/`](https://github.com/mozilla-ai/otari/tree/main/deploy/railway).

## Connect to otari.ai

In hybrid mode, Otari delegates provider routing, authentication, and usage tracking to [otari.ai](https://otari.ai). No local database or provider credentials are needed.

### 1. Create a minimal config file

```yaml
host: "0.0.0.0"
port: 8000
```

No `providers` block, no `database_url`, no `master_key`.

### 2. Set your otari.ai credentials

You need the gateway token (`gw-...`) for this Otari instance from otari.ai.
In otari.ai, go to `Organisation > Gateways`, create or open a gateway, then
click `Create token`. This is not the per-request user token (`tk_...`) that
clients send in `Authorization: Bearer ...`.

Pass it as an environment variable. Create a `.env` file:

```bash
OTARI_AI_TOKEN=gw_your_token_here
```

### 3. Start Otari

```bash
docker run --rm \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/config.yml:/app/config.yml:ro" \
  mzdotai/otari:latest \
  otari serve --config /app/config.yml
```

No postgres container is needed -- otari.ai handles storage.

### 4. Verify

```bash
curl http://localhost:8000/health
```

The response includes platform reachability status:

```json
{"status": "healthy", "mode": "hybrid", "platform_reachable": "yes"}
```

Check readiness:

```bash
curl http://localhost:8000/health/readiness
```

Then verify a chat request using an otari.ai user token:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <your-otari-user-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

## Optional services

Otari supports two opt-in services via Docker Compose profiles.

### Code execution sandbox

A sandboxed Python REPL for `otari_code_execution` tool calls:

```bash
docker compose --profile code-exec up -d
```

### Web search

A SearXNG-based web search backend for `otari_web_search` tool calls:

```bash
docker compose --profile web-search up -d
```

SearXNG's free engines (DuckDuckGo, mojeek, qwant, …) rate-limit/CAPTCHA
automated queries by IP, so they can be unreliable. For dependable results,
use a licensed search API instead. A ready-to-run **Brave Search** adapter
ships in `scripts/web-search-brave-adapter/`:

```bash
export BRAVE_API_KEY=...   # from https://brave.com/search/api/
export OTARI_WEB_SEARCH_URL=http://brave-adapter:8080
docker compose --profile web-search-brave up -d --build brave-adapter otari
```

A ready-to-run **Tavily** adapter also ships in
`scripts/web-search-tavily-adapter/`:

```bash
export TAVILY_API_KEY=...   # from https://tavily.com/
export OTARI_WEB_SEARCH_URL=http://tavily-adapter:8080
docker compose --profile web-search-tavily up -d --build tavily-adapter otari
```

`WebSearchBackend` is URL-configured, so any service exposing a
SearXNG-compatible `/search?format=json` endpoint works — copy the adapter to
front Exa, Serper, etc.

Both code-exec and web-search profiles can be combined:

```bash
docker compose --profile code-exec --profile web-search up -d
```

When a profile is not running, Otari returns a 502 to requests that try to use that tool.

## Environment variables

Provider API keys can be passed as environment variables instead of putting them in `config.yml`:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OTARI_PORT` | Otari server bind port (default: `8000`) |
| `OTARI_MASTER_KEY` | Master key for management endpoints |
| `OTARI_AI_TOKEN` | Platform token from otari.ai |

See [Configuration](configuration.md) for the full reference.
