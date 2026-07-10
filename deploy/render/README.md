# Deploy Otari on Render

[`render.yaml`](../../render.yaml) at the repo root is the Blueprint. This page
is how to use it and what to check after Apply.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mozilla-ai/otari)

## Resources

| Name | Plan / kind | Notes |
| --- | --- | --- |
| `otari` | Web, Starter | `docker.io/mzdotai/otari:0.2.0`, `/health`, Oregon |
| `otari-db` | Postgres 16, basic-256mb | User/db `otari`, `ipAllowList: []` (private only) |

Render does not build from this tree. It pulls the published image. On first
boot, `OTARI_AUTO_MIGRATE` runs Alembic and `OTARI_BOOTSTRAP_API_KEY` prints a
`gw-…` client key once in the service logs.

Postgres 16 matches the `postgres:16-alpine` service in
[`docker-compose.yml`](../../docker-compose.yml). Otari is not locked to that
major; change `postgresMajorVersion` in the Blueprint if you prefer another.

## Env vars

Otari’s own settings are `OTARI_<FIELD>`. Upstream provider SDKs (through
[any-llm](https://github.com/mozilla-ai/any-llm)) read their usual key names.

| Variable | Set how | Notes |
| --- | --- | --- |
| `PORT`, `OTARI_PORT` | both `8000` | Otari listens on `OTARI_PORT`. Render’s health check and routing use `PORT`. They have to agree. |
| `OTARI_HOST` | `0.0.0.0` | Without this, the process only binds localhost inside the container. |
| `OTARI_DATABASE_URL` | from `otari-db` | Keys, budgets, usage. |
| `OTARI_MASTER_KEY` | generated | Management APIs. Copy it from the Environment tab after deploy. |
| `OTARI_REQUIRE_PRICING` | `false` | Image default is `true`. With no pricing table yet, that rejects every model. |
| `OTARI_AUTO_MIGRATE` | `true` | |
| `OTARI_BOOTSTRAP_API_KEY` | `true` | |

Apply also prompts for four provider keys, all optional in the form:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`

Fill the ones you will call; leave the others empty. Chat completions need at
least one. Anything else from [`docs/models.md`](../../docs/models.md) can be
added later on the service Environment tab under its native name.

When you do want priced models, put pricing (and the rest of structured config)
in `OTARI_CONFIG_YAML` or `OTARI_CONFIG_B64`. See
[Full config via environment](../../docs/configuration.md#full-config-via-environment).

Render’s `postgresql://` URL is fine as-is. Otari rewrites it to the async
driver on its own.

## Deploy

1. Use the button above (or the same link in the root README).
2. On Apply, enter whatever provider keys you need.
3. Wait until `otari` and `otari-db` are live, then grab the `*.onrender.com` URL
   from the Dashboard.

## Test it

```bash
export OTARI_URL=https://<your-service>.onrender.com

curl "$OTARI_URL/health"
curl "$OTARI_URL/health/readiness"
```

Readiness should show the database connected. Pull the bootstrap `gw-…` key out
of the **otari** logs (first boot only):

```bash
curl "$OTARI_URL/v1/chat/completions" \
  -H "Authorization: Bearer <gw-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

Use a `provider:model` string that matches a key you actually set.

Quick checks:

- [ ] Web + Postgres both live
- [ ] `/health` OK
- [ ] `/health/readiness` → database connected
- [ ] `gw-…` key in logs
- [ ] Chat completion returns 200
- [ ] Clients talk to `$OTARI_URL/v1` — the image’s sample page still hardcodes `http://localhost:8000/v1`

## Editing the Blueprint

Change [`render.yaml`](../../render.yaml), keep the image tag pinned unless you
mean to float, then apply it on a throwaway Render project and run the checks
above. If the Deploy button’s repo URL moves, update this README, the root
README, and `docs/deployment.md` together.
