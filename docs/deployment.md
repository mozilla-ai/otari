# Deployment

The gateway is distributed as a Docker image on [Docker Hub](https://hub.docker.com/r/mzdotai/otari).

## Prerequisites

- Docker and Docker Compose

## Deploy standalone

In this setup, the gateway manages its own database, API keys, users, budgets, and usage tracking. You provide the LLM provider credentials directly.

### 1. Create a config file

```bash
cp config.example.yml config.yml
```

Edit `config.yml` with at least a master key and one provider:

```yaml
database_url: "postgresql://gateway:gateway@postgres:5432/gateway"
host: "0.0.0.0"
port: 8000

master_key: "your-secret-master-key"

providers:
  openai:
    api_key: "sk-..."
```

### 2. Start the services

```bash
docker compose up -d
```

This starts two services:
- **gateway** on port 8000
- **postgres** on port 5433 (host) / 5432 (container)

### 3. Verify

```bash
curl http://localhost:8000/health
```

You should get `{"status": "healthy"}`.

Check readiness as well:

```bash
curl http://localhost:8000/health/readiness
```

Then verify an authenticated management call with your master key:

```bash
curl http://localhost:8000/v1/keys \
  -H "Otari-Key: Bearer <your-master-key>"
```

## Deploy with otari.ai

When connected to [otari.ai](https://otari.ai), the gateway delegates provider routing, authentication, and usage tracking to otari.ai. No local database or provider credentials are needed.

### 1. Create a minimal config file

```yaml
host: "0.0.0.0"
port: 8000
```

No `providers` block, no `database_url`, no `master_key`.

### 2. Set your otari.ai credentials

You need your gateway token from your otari.ai account.

Pass them as environment variables. Create a `.env` file:

```bash
OTARI_AI_TOKEN=gw_your_token_here
```

### 3. Start the gateway

```bash
docker run --rm \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/config.yml:/app/config.yml:ro" \
  mzdotai/otari:latest \
  gateway serve --config /app/config.yml
```

No postgres container is needed -- otari.ai handles storage.

### 4. Verify

```bash
curl http://localhost:8000/health
```

The response includes platform reachability status:

```json
{"status": "healthy", "mode": "platform", "platform_reachable": "yes"}
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
    "model": "openai:gpt-4o",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

## Optional services

The gateway supports two opt-in services via Docker Compose profiles.

### Code execution sandbox

A sandboxed Python REPL for `code_execution` tool calls:

```bash
docker compose --profile code-exec up -d
```

### Web search

A SearXNG-based web search backend for `web_search` tool calls:

```bash
docker compose --profile web-search up -d
```

Both profiles can be combined:

```bash
docker compose --profile code-exec --profile web-search up -d
```

When a profile is not running, the gateway returns a 502 to requests that try to use that tool.

## Environment variables

Provider API keys can be passed as environment variables instead of putting them in `config.yml`:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GATEWAY_PORT` | Host port for the gateway (default: `8000`) |
| `GATEWAY_MASTER_KEY` | Master key for management endpoints |
| `OTARI_AI_TOKEN` | otari.ai gateway token |

See [Configuration](configuration.md) for the full reference.
