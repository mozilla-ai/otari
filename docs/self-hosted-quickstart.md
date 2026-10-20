# Self-hosted quickstart

In this tutorial you run Otari on your own machine with Docker Compose, connect
it to OpenAI, and send your first request through it. For a throwaway container
with no clone, see the [README quickstart](../README.md#quickstart) instead.

You need:

- Docker with Compose
- `git`, `curl` and `openssl`
- An OpenAI API key from <https://platform.openai.com/api-keys>

## 1. Get the code

```bash
git clone https://github.com/mozilla-ai/otari
cd otari
```

Run every command that follows from this directory.

## 2. Write the configuration file

Create `config.yml`. It tells Otari to keep its data in the PostgreSQL database
that Compose starts, and to price common models with the prices bundled in
Otari:

```bash
cat > config.yml <<'EOF'
database_url: "postgresql://otari:otari@postgres:5432/otari"
default_pricing: true
EOF
```

## 3. Create the secrets

Otari needs three secrets: a master key you manage the gateway with, a key that
encrypts the provider credentials it stores, and a pepper for the copies of
attached files it makes at providers. Generate all three into `.env`, which
Compose reads:

```bash
export OTARI_MASTER_KEY=$(openssl rand -hex 32)
cat > .env <<EOF
OTARI_MASTER_KEY=$OTARI_MASTER_KEY
OTARI_SECRET_KEY=$(docker run --rm docker.io/mzdotai/otari:latest otari gen-secret-key)
OTARI_PROVIDER_ACCOUNT_PEPPER=$(openssl rand -base64 32)
EOF
```

The first command also keeps the master key in your shell for step 6. Git
ignores both `config.yml` and `.env`.

## 4. Start Otari

```bash
docker compose pull
docker compose up -d
```

Wait a few seconds, then check that it answers:

```bash
curl http://localhost:8000/api/v1/health
```

The answer includes `"status":"healthy"`. If the connection is refused, wait a
few more seconds and try again.

## 5. Copy your API key

On its first start, Otari creates an API key and prints it once. Find it in the
logs:

```bash
docker compose logs otari | grep 'tk-'
```

Copy the `tk-...` value. This is the key an application calls Otari with.

## 6. Connect OpenAI

Add OpenAI as a provider, replacing `sk-...` with your OpenAI API key:

```bash
curl -X POST http://localhost:8000/api/v1/hosted-providers \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "api_key": "sk-..."}'
```

Otari stores the key encrypted, asks OpenAI which models the key can use, and
switches on every one it has a price for. The answer describes the new provider
and shows only the last four characters of your key.

## 7. Send a request

Replace `tk-...` with the key from step 5:

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer tk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello."}]
  }'
```

The answer is an OpenAI-style chat completion, with the model's reply in
`choices[0].message.content` and the tokens it used in `usage`.

## 8. See it in the dashboard

Open <http://localhost:8000/> and sign in with your master key (`echo
$OTARI_MASTER_KEY` prints it). Open **Activity**: your request is listed with
its model, its tokens and its cost.

You now have a gateway that authenticates its callers, holds the provider
credential for them, and records every request.

## Next steps

- [Admin dashboard](dashboard.md): manage keys, users and providers in the browser.
- [Access control](access-control.md): give each application its own key and budget.
- [Models](models.md): connect more providers and choose model names.
- [Routing](routing.md): fall back to another model when one fails.
- [Deployment](deployment.md): run Otari in production.
