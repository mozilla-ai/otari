# Self-hosted quickstart

In this tutorial you run Otari on your own machine with Docker Compose, connect
it to OpenAI, and send your first request through it. For a throwaway container
with no clone, see the [README quickstart](https://github.com/mozilla-ai/otari/blob/main/README.md#quickstart) instead.

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

## 2. Create the settings

Otari runs here on environment variables alone, which Compose reads from
`.env`. Generate the file:

```bash
export OTARI_MASTER_KEY=$(openssl rand -hex 32)
cat > .env <<EOF
OTARI_CONFIG_PATH=/dev/null
OTARI_DATABASE_URL=postgresql://otari:otari@postgres:5432/otari
OTARI_DEFAULT_PRICING=true
OTARI_MASTER_KEY=$OTARI_MASTER_KEY
OTARI_SECRET_KEY=$(docker run --rm docker.io/mzdotai/otari:latest otari gen-secret-key)
OTARI_PROVIDER_ACCOUNT_PEPPER=$(openssl rand -base64 32)
EOF
```

- `OTARI_CONFIG_PATH=/dev/null` tells Compose to start Otari with no
  configuration file.
- `OTARI_DATABASE_URL` keeps Otari's data in the PostgreSQL database that
  Compose starts.
- `OTARI_DEFAULT_PRICING` prices common models with the prices bundled in Otari.
- `OTARI_MASTER_KEY` is the key you manage the gateway with.
- `OTARI_SECRET_KEY` encrypts the provider credentials Otari stores.
- `OTARI_PROVIDER_ACCOUNT_PEPPER` names the provider accounts that hold copies
  of attached files. Compose requires it.

The first command also keeps the master key in your shell for steps 4 and 5. Git
ignores `.env`.

## 3. Start Otari

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

## 4. Connect OpenAI

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

## 5. Create an API key

Create the key an application calls Otari with:

```bash
curl -X POST http://localhost:8000/api/v1/keys \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_name": "quickstart"}'
```

Copy the `tk-...` value of `key` from the answer. It is shown only this once.

## 6. Send a request

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

## 7. See it in the dashboard

Open <http://localhost:8000/> and sign in with your master key (`echo
$OTARI_MASTER_KEY` prints it). Open **Activity**: your request is listed with
its model, its tokens and its cost. The OpenAI provider you added is listed
under **Deployment**, **Hosted providers**, in the account menu.

You now have a gateway that authenticates its callers, holds the provider
credential for them, and records every request.

## Next steps

- [Admin dashboard](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/dashboard/dashboard): manage keys, users and providers in the browser.
- [Access control](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides): give each application its own key and budget.
- [Models](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/reference/gateway/models): connect more providers and choose model names.
- [Connect a llama.cpp server](providers/llamacpp.md): serve a model you run yourself.
- [Routing](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/reference/gateway/routing): fall back to another model when one fails.
- [Deployment](deployment.md): run Otari in production.
