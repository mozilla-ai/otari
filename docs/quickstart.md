# Quickstart

Get the Otari Gateway running locally and make your first request in about five minutes. This guide uses **standalone mode**: the gateway runs entirely on your own machine, manages its own database, and talks to providers using credentials you supply. Nothing leaves your environment, and you don't need an otari.ai account.

## Prerequisites

- Docker and Docker Compose
- An API key for at least one LLM provider (this guide uses OpenAI)

## 1. Configure

Copy the example config and open it:

```bash
cp config.example.yml config.yml
```

Set a master key and add one provider:

```yaml
host: "0.0.0.0"
port: 8000

master_key: "your-secret-master-key"

providers:
  openai:
    api_key: "sk-..."

pricing:
  openai:gpt-4o-mini:
    input_price_per_million: 0.15
    output_price_per_million: 0.60
```

The `pricing` entry matters: by default the gateway rejects requests for any model it can't price, so it can enforce budgets. Add an entry for the model you plan to call.

## 2. Start your gateway

```bash
docker compose up -d
```

This starts the gateway on port 8000 and a Postgres database. Confirm it's healthy:

```bash
curl http://localhost:8000/health
```

You should get `{"status": "healthy"}`.

## 3. Create an API key

Use your master key (the one you set in `config.yml`) to create an API key for making requests:
 
```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Otari-Key: Bearer your-secret-master-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "quickstart"}'
```
 
The response includes your new key. Copy it, this is the only time it's shown in full. You'll use it in the next step.
 
> The gateway also creates a bootstrap key on first startup and prints it to the logs. Creating a key explicitly, as above, is more reliable for getting started.

## 4. Make your first request

The gateway is OpenAI-compatible, so any standard client works. With curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

Or with the OpenAI Python SDK, just point `base_url` at the gateway:

```python
from openai import OpenAI

client = OpenAI(
    api_key="<your-api-key>",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="openai:gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print(response.choices[0].message.content)
```

Either way, you'll get a standard chat completion back. Prefer a typed client? Use one of the Otari SDKs for Python, TypeScript, Rust, or Go.

## What just happened

Your request went to the gateway, not directly to OpenAI. The gateway authenticated your key, checked the request against your budget, resolved the provider credential from your config, forwarded the call, and logged the usage. Your provider key never left the gateway, and every request is now tracked and queryable through `/v1/usage`.

## Next steps

- [Deployment](deployment.md) picks up from here with platform mode, optional services, and deployment-specific configuration.
- [Built-in tools](tools.md) — let any model use a sandboxed code REPL or web search the gateway runs itself.
- [Guardrails](guardrails.md) — request-level checks like prompt-injection detection, enforced before the provider is called.
- [Configuration](configuration.md) is the full reference for budgets, rate limits, pricing, and provider options.
- [Models](models.md) lists every supported provider and the model format.
- [API reference](api-reference.md) documents every endpoint.
