# Deploy Otari on Railway

One-click deploy of a self-hosted [Otari](https://github.com/mozilla-ai/otari)
gateway in front of key-only providers (OpenAI, Anthropic, Mistral, Gemini),
backed by a managed Postgres database. No local setup, bring a provider key and
go.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/template/REPLACE_ME)

> The button URL is a placeholder until the template is published on the
> mozilla-ai Railway account. See [Publishing the template](#publishing-the-template).

## What you get

The template stands up two services:

| Service | Source | Notes |
| --- | --- | --- |
| **otari** | `docker.io/mzdotai/otari:latest` | Target port `8000`, healthcheck `/health`. Pulls the published image; builds nothing. |
| **Postgres** | Railway managed | Durable storage for keys, users, budgets, and usage. |

Otari is a good fit for a one-click deploy: the app is stateless, its only
stateful dependency is Postgres, the image is published, and `auto_migrate` plus
`bootstrap_api_key` are on by default, so the schema is created and a first-use
API key is minted on startup with no extra steps.

## Configuration

The template wires the two services together and asks for a provider key. All of
Otari's scalar config is reachable through `OTARI_<FIELD>` environment variables;
the snapshot of what the template sets lives in [`template.json`](template.json).

| Variable | Value | Required |
| --- | --- | --- |
| `OTARI_DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Pre-wired; leave as-is. |
| `OTARI_MASTER_KEY` | auto-generated (`${{secret(48)}}`) | Auto-set; read it from the otari service's Variables tab. |
| `OTARI_REQUIRE_PRICING` | `false` | Pre-set, so an env-only deploy serves models that have no configured pricing. |
| `OPENAI_API_KEY` | your key | At least one provider key is required. |
| `ANTHROPIC_API_KEY` | your key | At least one provider key is required. |
| `MISTRAL_API_KEY` | your key | At least one provider key is required. |
| `GEMINI_API_KEY` | your key | At least one provider key is required. |

Notes:

- Otari normalizes a `postgresql://` URL to the async driver automatically, so
  Railway's `DATABASE_URL` works without edits.
- `OTARI_REQUIRE_PRICING=false` is deliberate. The image default is `true`
  (fail-closed), which rejects any model without configured pricing; that would
  make an env-only deploy unusable until pricing is added. Configuring per-model
  pricing via env is tracked in
  [#208](https://github.com/mozilla-ai/otari/issues/208).
- Provider keys use each provider's native variable name (`OPENAI_API_KEY`,
  etc.), which the underlying `any-llm` SDK reads directly.

## Deploy

1. Click **Deploy on Railway** above.
2. Fill in at least one provider key. The master key is generated for you.
3. Deploy. Railway provisions Postgres, pulls the Otari image, runs migrations on
   startup, and bootstraps a first-use API key.
4. Generate a public domain for the otari service (Settings → Networking) if you
   want to call it from outside Railway.

## Verify

Once both services are healthy:

```bash
# Replace with your service's public domain.
export OTARI_URL=https://your-otari.up.railway.app

curl "$OTARI_URL/health"
```

Grab the bootstrapped API key from the otari service's deploy logs (printed once
on first startup), then make a real request:

```bash
curl "$OTARI_URL/v1/chat/completions" \
  -H "Authorization: Bearer <bootstrapped-or-generated-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}]
  }'
```

Use the provider that matches the key you supplied (for example `anthropic:...`,
`mistral:...`, or `gemini:...`).

## Publishing the template

A Railway multi-service template (the Postgres service, the env-var input form,
and the `${{Postgres.DATABASE_URL}}` reference wiring) is a Railway-hosted object
and cannot be fully round-tripped from a file in this repo. This directory is the
human source of truth plus a reviewable snapshot; publishing is a one-time manual
step on the mozilla-ai Railway account.

1. Stand up the two services described in [`template.json`](template.json) in a
   throwaway Railway project and confirm a real `/v1/chat/completions`
   round-trip plus that the bootstrapped key works.
2. From that project, create and publish the template on the mozilla-ai account.
3. Capture the published template URL and replace `REPLACE_ME` in the
   **Deploy on Railway** button above (and in the project root `README.md`).
4. If the published config drifts from this snapshot, update
   [`template.json`](template.json) in the same change so the two stay in sync.
