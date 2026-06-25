# Quickstart

Get Otari running locally and make your first request in about five minutes. This guide uses **standalone mode**: Otari runs entirely on your own machine, manages its own database, and talks to providers using credentials you supply. Nothing leaves your environment, and you don't need an otari.ai account.

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

The `pricing` entry matters: by default Otari rejects requests for any model it can't price, so it can enforce budgets. Add an entry for the model you plan to call.

## 2. Start your Otari

```bash
docker compose up -d
```

This starts Otari on port 8000 and a Postgres database. Confirm it's healthy:

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
 
> Otari also creates a bootstrap key on first startup and prints it to the logs. Creating a key explicitly, as above, is more reliable for getting started.

## 4. Make your first request

Otari is OpenAI-compatible, so any standard client works. With curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Otari-Key: Bearer <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

Or with the official Otari Python SDK (`pip install otari`, requires Python 3.11+):

```python
from otari import OtariClient

client = OtariClient(
    api_base="http://localhost:8000",
    api_key="<your-api-key>",
)

response = client.completion(
    model="openai:gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)

print(response.choices[0].message.content)
```

Either way, you'll get a standard chat completion back. Otari is also OpenAI-compatible, so any OpenAI client works by pointing its `base_url` at `http://localhost:8000/v1`. Prefer another language? Otari has SDKs for TypeScript, Rust, and Go too.

## What just happened

Your request went to Otari, not directly to OpenAI. Otari authenticated your key, checked the request against your budget, resolved the provider credential from your config, forwarded the call, and logged the usage. Your provider key never left Otari, and every request is now tracked and queryable through `/v1/usage`.

## Next steps

- [Deployment](deployment.md) picks up from here with hybrid mode, optional services, and deployment-specific configuration.
- [Built-in tools](tools.md) — let any model use a sandboxed code REPL or web search Otari runs itself.
- [Guardrails](guardrails.md) — request-level checks like prompt-injection detection, enforced before the provider is called.
- [Configuration](configuration.md) is the full reference for budgets, rate limits, pricing, and provider options.
- [Models](models.md) lists every supported provider and the model format.
- [API reference](api-reference.md) documents every endpoint.
