# Quickstart

This guide runs a persistent Otari with Docker Compose. For an
ephemeral, no-clone example, use the [README quickstart](../README.md#quickstart).

## Prerequisites

- Docker with Compose
- A provider API key, or a self-hosted model server such as
  [llama.cpp](#self-hosted-llamacpp)

## Configure

```bash
git clone https://github.com/mozilla-ai/otari
cd otari
```

Compose reads the settings from `.env`. Create it once; the guard leaves an
existing `.env` alone, because a second `OTARI_SECRET_KEY` would make stored
provider keys undecryptable:

```bash
test -e .env || cat > .env <<EOF
OTARI_CONFIG_PATH=/dev/null
OTARI_DATABASE_URL=postgresql://otari:otari@postgres:5432/otari
OTARI_MASTER_KEY=$(openssl rand -hex 32)
OTARI_DEFAULT_PRICING=true
OTARI_SECRET_KEY=$(docker run --rm mzdotai/otari:latest otari gen-secret-key)
OTARI_PROVIDER_ACCOUNT_PEPPER=$(openssl rand -base64 32)
EOF
```

- `OTARI_CONFIG_PATH=/dev/null` runs Otari on environment variables alone.
- `OTARI_MASTER_KEY` signs you in to the dashboard.
- `OTARI_DEFAULT_PRICING` prices common models from bundled rates.
- `OTARI_SECRET_KEY` encrypts the provider keys you add in the dashboard.
- `OTARI_PROVIDER_ACCOUNT_PEPPER` is required while provider copies of
  attached files are on, which they are by default. See
  [Provider copies](configuration.md#provider-copies).

`.env` is gitignored. For a shared deployment, inject these through your
platform's secret manager instead.

## Start

```bash
docker compose pull
docker compose up -d
curl http://localhost:8000/api/v1/health
```

The first startup on an empty database creates an API key and prints it once.
Find the `tk-` value in the Otari container logs. You can also create a key
from the dashboard at `http://localhost:8000/`.

## Add a provider

1. Open the dashboard at `http://localhost:8000/` and sign in with the master
   key.
2. Open the account menu, choose **Deployment**, then **Hosted providers**
   (`http://localhost:8000/admin/hosted-providers`).
3. Choose **Add provider**, pick the provider (for example OpenAI), paste its
   API key, and save. Otari lists the models the key reaches.
4. Check that each model you want to serve is switched on and priced.
   `OTARI_DEFAULT_PRICING` prices common models from bundled rates. A model
   nothing prices arrives switched off, and Otari rejects unpriced budgeted
   traffic by default, so set a price on the model's row and switch it on.

See [Hosted providers](hosted-providers.md) and
[Admin dashboard](dashboard.md) for the rest of the page.

## Send a request

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer tk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello."}]
  }'
```

The request is authenticated, checked against applicable budgets, routed with
the provider credential held by Otari, and recorded in Activity and Usage.

## Self-hosted llama.cpp

Start `llama-server` with `--alias`, so the model id it serves is a stable name
rather than the path of the GGUF file:

```bash
llama-server -m ./models/qwen3-8b-q4_k_m.gguf --alias qwen3-8b --host 0.0.0.0 --port 8080
```

On the Hosted providers page, add a provider of type **llama.cpp**
(`llamacpp`). It needs no API key, so leave the key blank, rather than using
the `openai` provider with a placeholder key. Set the API base URL to where
Otari reaches the server: `http://host.docker.internal:8080/v1` from the
Compose container on macOS or Windows, or the host's address on Linux.

Bundled pricing does not cover a local model, so set a price on its row (0 is
accepted) and switch it on. The id you price is the alias, and it is also the
model you request:

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer tk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llamacpp:qwen3-8b",
    "messages": [{"role": "user", "content": "Say hello."}]
  }'
```

Without `--alias`, the server reports the file path as the model id, so a price
set under a shorter name never matches it and requests are refused.

## Next steps

- [Admin dashboard](dashboard.md)
- [Hosted providers](hosted-providers.md)
- [Configuration](configuration.md)
- [Deployment](deployment.md)
- [Models](models.md)
- [Routing](routing.md)
- [Built-in tools](tools.md)
