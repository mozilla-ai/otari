# Quickstart

Get Otari running locally and make your first request in about five minutes. This guide uses **standalone mode**: Otari runs entirely on your own machine, manages its own database, and talks to providers using credentials you supply. Nothing leaves your environment, and you don't need an otari.ai account.

This guide runs the local Docker Compose stack (Otari plus Postgres) from a cloned checkout. If you want the fastest no-clone path, use the Quickstart in the project [README](../README.md).

## Two ways to configure Otari

You can give Otari its provider credentials in one of two ways. Both use the same Docker Compose stack and end at the same first request; pick whichever fits how you like to work.

- **Config-first (Option A):** put your provider key in `config.yml` before starting. Best when you already have keys and want everything in one version-controllable file.
- **Dashboard-first (Option B):** start with almost nothing, then add providers from the admin dashboard in your browser. Best for trying Otari out, or when you would rather manage credentials through a UI.

## Prerequisites

- Docker and Docker Compose
- An API key for at least one LLM provider (this guide uses OpenAI)

Clone the repo first; both options need the Compose file and the example config:

```bash
git clone https://github.com/mozilla-ai/otari
cd otari
```

## Option A: configure with `config.yml`

### 1. Configure

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

The `pricing` entry matters because `require_pricing` is on by default: Otari rejects requests for any model it can't resolve a price for, so it can enforce budgets. Add a pricing entry for the model you plan to call, or enable `default_pricing: true` to use the bundled fallback pricing for common models.

> To add more providers through the dashboard later instead of editing `config.yml`, you also need to set `OTARI_SECRET_KEY`, which encrypts stored provider keys at rest. See [Option B](#option-b-configure-from-the-dashboard) below and the [Admin dashboard guide](dashboard.md).

### 2. Start Otari

```bash
docker compose pull
docker compose up -d
```

This starts Otari on port 8000 and a Postgres database. Confirm it's healthy:

```bash
curl http://localhost:8000/health
```

You should get `{"status": "healthy"}`. Now skip ahead to [Create an API key](#create-an-api-key).

## Option B: configure from the dashboard

Here you start Otari with no providers configured and add them in the browser. To store a provider key from the dashboard, Otari encrypts it at rest with a Fernet key you supply as `OTARI_SECRET_KEY`, so you set that first. See the [Admin dashboard guide](dashboard.md) for the full tour.

### 1. Write a minimal config

Create `config.yml` with a master key (your dashboard sign-in) and no providers:

```yaml
host: "0.0.0.0"
port: 8000

master_key: "your-secret-master-key"

# Serve common models using bundled fallback pricing, so a first request works
# before you set pricing yourself. Set per-model prices later on the Models page.
default_pricing: true
```

### 2. Generate an encryption key and start Otari

Generate a Fernet key and export it so Compose passes it into the container:

```bash
export OTARI_SECRET_KEY="$(docker run --rm mzdotai/otari:latest otari gen-secret-key)"
docker compose pull
docker compose up -d
```

Keep `OTARI_SECRET_KEY` safe and separate from the database: losing it makes every provider key you store in the dashboard undecryptable. Confirm Otari is healthy:

```bash
curl http://localhost:8000/health
```

You should get `{"status": "healthy"}`.

### 3. Add a provider in the dashboard

Open `http://localhost:8000/` in your browser, sign in with the master key from your config, then open **Providers** and add OpenAI with your `sk-...` key. Use **Test the connection** to confirm the credential works. The key is encrypted at rest with `OTARI_SECRET_KEY`; the UI only ever shows its last four characters.

## Create an API key

Use your master key (the one you set in `config.yml`) to create an API key for making requests:

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Otari-Key: your-secret-master-key" \
  -H "Content-Type: application/json" \
  -d '{"key_name": "quickstart"}'
```

The response includes your new key. Copy it, this is the only time it's shown in full. You'll use it in the next step.

> Otari also creates a bootstrap key on first startup and prints it to the logs. Creating a key explicitly, as above, is more reliable for getting started. You can also issue keys from the dashboard's **API keys** page.

## Make your first request

Otari is OpenAI-compatible, so any standard client works. With curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Otari-Key: <your-api-key>" \
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

- [Admin dashboard](dashboard.md) is the operator's tour of the browser UI: the two-key model, a first-run walkthrough, and a page-by-page reference.
- [Deployment](deployment.md) picks up from here with hybrid mode, optional services, and deployment-specific configuration.
- [Built-in tools](tools.md): let any model use a sandboxed code REPL or web search Otari runs itself.
- [Guardrails](guardrails.md): request-level checks like prompt-injection detection, enforced before the provider is called.
- [Configuration](configuration.md) is the full reference for budgets, rate limits, pricing, and provider options.
- [Models](models.md) lists every supported provider and the model format.
- [API reference](api-reference.md) documents every endpoint.
