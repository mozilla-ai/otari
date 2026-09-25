# Deploy Otari on Railway

One-click deploy of a self-hosted [Otari](https://github.com/mozilla-ai/otari)
gateway in front of key-only providers (OpenAI, Anthropic, Mistral, Gemini),
backed by a managed Postgres database. No local setup: deploy, then add your
provider keys on the dashboard.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/otari-railway-template-demo)

## What you get

The template stands up two services:

| Service | Source | Notes |
| --- | --- | --- |
| **otari** | `docker.io/mzdotai/otari:latest` | Target port `8000`, healthcheck `/api/v1/health`. Pulls the published image; builds nothing. |
| **Postgres** | Railway managed | Durable storage for keys, users, budgets, and usage. |

Otari is a good fit for a one-click deploy: the app is stateless, its only
stateful dependency is Postgres, the image is published, and `auto_migrate` plus
`bootstrap_api_key` are on by default, so the schema is created and a first-use
API key is minted on startup with no extra steps.

## Configuration

The template wires the two services together and generates the keys it needs. All of
Otari's scalar config is reachable through `OTARI_<FIELD>` environment variables;
the snapshot of what the template sets lives in [`template.json`](template.json).

| Variable | Value | Notes |
| --- | --- | --- |
| `OTARI_DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Pre-wired; leave as-is. |
| `OTARI_MASTER_KEY` | auto-generated (`${{secret(48)}}`) | Auto-set; read it from the otari service's Variables tab. |
| `OTARI_SECRET_KEY` | auto-generated Fernet key | Encrypts the provider credentials you add on the Providers page. Keep it: losing it makes them unrecoverable. |
| `OTARI_REQUIRE_PRICING` | `false` | Pre-set, so a fresh deploy serves models that have no configured pricing. |
| `OTARI_DEFAULT_PRICING` | `true` | Pre-set, so common models are metered from the bundled genai-prices dataset without configuring each one. Prices you set in the dashboard or via `/api/v1/pricing` always override it. |

Notes:

- Providers are added after deploy, on the dashboard's Providers page (sign in
  with the master key). That declares the provider, so it serves requests and
  its models are listed. Any [any-llm provider](https://docs.mozilla.ai/any-llm/providers/)
  works. To keep keys in Railway variables instead, declare the providers
  through `OTARI_CONFIG_YAML` with `api_key: ${ANTHROPIC_API_KEY}` references;
  see [Full config via environment](../../docs/configuration.md#full-config-via-environment).
  Setting a provider's native variable without declaring it is deprecated.
- Otari normalizes a `postgresql://` URL to the async driver automatically, so
  Railway's `DATABASE_URL` works without edits.
- `OTARI_REQUIRE_PRICING=false` is deliberate. The image default is `true`
  (fail-closed), which rejects any model without configured pricing; that would
  make a fresh deploy unusable until pricing is added. To serve priced
  models instead, supply `pricing` (and any other structured config like custom
  `api_base` or Vertex settings) through `OTARI_CONFIG_YAML` / `OTARI_CONFIG_B64`;
  see [Full config via environment](../../docs/configuration.md#full-config-via-environment).
- `OTARI_DEFAULT_PRICING=true` is also pre-set so that, paired with the above,
  common models are metered using community-maintained rates (the bundled
  genai-prices dataset) instead of being served unpriced. These are estimates
  and can lag real provider rates, so set explicit prices on the dashboard's
  Models page or via `/api/v1/pricing` for anything you bill on; database prices
  always win over the fallback. The Models page shows, per model, whether this
  fallback is active.

## Deploy

1. Click **Deploy on Railway** above.
2. Deploy. The master key and secret key are generated for you. Railway
   provisions Postgres, pulls the Otari image, runs migrations on startup, and
   bootstraps a first-use API key.
3. Generate a public domain for the otari service (Settings → Networking).
4. Open that domain, sign in with the master key (from the Variables tab), and
   add at least one provider on the Providers page.

## Verify

Once both services are healthy:

```bash
# Replace with your service's public domain.
export OTARI_URL=https://your-otari.up.railway.app

curl "$OTARI_URL/api/v1/health"
```

Grab the bootstrapped API key from the otari service's deploy logs (printed once
on first startup), then make a real request:

```bash
curl "$OTARI_URL/api/v1/chat/completions" \
  -H "Authorization: Bearer <bootstrapped-or-generated-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

Use a provider you added (for example `anthropic:...`, `mistral:...`, or
`gemini:...`).

## Maintaining the template

A Railway multi-service template (the Postgres service, the env-var input form,
and the `${{Postgres.DATABASE_URL}}` reference wiring) is a Railway-hosted object
and cannot be fully round-tripped from a file in this repo. This directory is the
human source of truth plus a reviewable snapshot; the live template lives on the
mozilla-ai Railway account and the button above points at its deploy link.

When changing the template:

1. Edit the template on the mozilla-ai Railway account, then deploy it once to a
   throwaway project and confirm a real `/api/v1/chat/completions` round-trip plus
   that the bootstrapped key works.
2. Update [`template.json`](template.json) in the same change so the snapshot
   matches the live config (services, variables, defaults, target port).
3. If the deploy link changes, update the **Deploy on Railway** button here, in
   the project root `README.md`, and in `docs/deployment.md`.

Listing the template in Railway's public marketplace is optional: the deploy
link works without it. Publishing only adds marketplace discoverability and
usage-kickback eligibility.
