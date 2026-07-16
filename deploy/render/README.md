# Deploy Otari on Render

Deploy Otari and Render Postgres from a Blueprint ([`render.yaml`](./render.yaml)). Render pulls the published Otari image rather than building this repository. For hybrid mode instead, connected to otari.ai, see [Hybrid mode](#hybrid-mode) below.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mozilla-ai/otari&path=deploy/render/render.yaml)

## What gets deployed

| Resource | Plan | Details |
| --- | --- | --- |
| `otari` | Free web service | `docker.io/mzdotai/otari:0.2.0`, Oregon, health check at `/health/readiness` |
| `otari-db` | Free Render Postgres 16 | Database and user `otari`, private connections only |

The web service is stateless. Postgres stores users, API key hashes, budgets, pricing, and usage history. On first boot, Otari runs its database migrations and creates a bootstrap API key, which is printed once in the service logs.

This Blueprint deploys Otari in standalone mode. For hybrid mode instead, connected to otari.ai without a local database, see [Hybrid mode](#hybrid-mode) below.

## Free instance limits

The free configuration is intended for evaluation and hobby use:

- Render spins down the web service after 15 minutes without inbound traffic. Waking it can take about a minute.
- Each workspace receives 750 free web-service hours per month, shared across its free services.
- Free Postgres is limited to 1 GB, expires after 30 days, and does not support backups.

Upgrade to paid instances before production use. See Render's full list of [Free instance limitations](https://render.com/docs/free).

## Configuration

The Blueprint wires the web service and database together. Otari settings use the `OTARI_<FIELD>` convention:

| Variable | Set how | Notes |
| --- | --- | --- |
| `PORT`, `OTARI_PORT` | `8000` | Keeps Render's detected port aligned with Otari's listening port. |
| `OTARI_HOST` | `0.0.0.0` | Binds Otari on the container network. |
| `OTARI_DATABASE_URL` | from `otari-db` | Uses the database's internal connection string. |
| `OTARI_MASTER_KEY` | generated | Protects management APIs. Retrieve it from the Environment tab. |
| `OTARI_REQUIRE_PRICING` | `true` | Fail closed when a model has no pricing entry. Set explicitly so image upgrades cannot weaken this policy. |
| `OTARI_DEFAULT_PRICING` | `true` | Uses bundled prices for common models while fail-closed pricing stays enabled. |
| `OTARI_AUTO_MIGRATE` | `true` | Runs Alembic migrations during startup. |
| `OTARI_BOOTSTRAP_API_KEY` | `true` | Creates a first-use API key when the database has no keys. |

Render's `postgresql://` connection string works without modification. Otari selects the async database driver automatically.

### Provider credentials

During initial setup, Render prompts for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, and `GEMINI_API_KEY`. Each field is optional, but at least one provider credential is required before the gateway can serve requests. Leave unused fields blank.

The underlying [any-llm](https://github.com/mozilla-ai/any-llm) SDK reads each provider's native environment variables. To use any other provider from [`docs/models.md`](../../docs/models.md), add its variable on the service's Environment tab. Render prompts for variables marked `sync: false` only during initial creation, so add or rotate credentials for an existing service from that tab.

### Pricing

The Blueprint sets `OTARI_REQUIRE_PRICING=true` (fail closed) and
`OTARI_DEFAULT_PRICING=true` (bundled fallback prices). Database pricing always takes precedence. For a custom model that is not covered by the bundled data, add pricing through the `/v1/pricing` API, `OTARI_CONFIG_YAML`, or `OTARI_CONFIG_B64`. See [Full config via environment](../../docs/configuration.md#full-config-via-environment).

## Deploy

1. Click **Deploy to Render** above. The button passes `path=deploy/render/render.yaml` so Render loads this Blueprint instead of looking for a root `render.yaml`.
2. If you create the Blueprint from the Dashboard instead (**New → Blueprint**), set **Blueprint Path** to:

   ```text
   deploy/render/render.yaml
   ```

3. Enter the provider credentials you need and leave unused fields blank.
4. Review the two free resources, then apply the Blueprint.
5. Wait for `otari` and `otari-db` to become live. Copy the web service's
   `*.onrender.com` URL from the Dashboard.

## Verify

```bash
export OTARI_URL=https://<your-service>.onrender.com

curl "$OTARI_URL/health"
curl "$OTARI_URL/health/readiness"
```

Readiness should report that the database is connected. Find the bootstrap
`gw-…` key in the `otari` service logs. It is printed in full only once, during the first successful startup.

```bash
curl "$OTARI_URL/v1/chat/completions" \
  -H "Authorization: Bearer <gw-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

Use a `provider:model` value that matches a credential you supplied. Clients should use `$OTARI_URL/v1` as their OpenAI-compatible base URL.

For a longer-lived deployment, use `OTARI_MASTER_KEY` to create a named API key, then revoke the bootstrap key through the key-management API.

## Upgrade for production

Before the free database expires, change the web-service plan to `starter` and the database plan to `basic-256mb` or higher. Paid web services do not spin down when idle, and paid Render Postgres adds continuous backups and point-in-time recovery.

The Blueprint pins the Otari image to a release tag, and image-backed services do not redeploy when a new image is published to that tag. To upgrade, change `image.url` to the desired release and sync the Blueprint. To keep every Blueprint update manual, turn off Auto Sync in the Blueprint settings.

## Hybrid mode

The Blueprint above deploys Otari in standalone mode with its own database. Otari also supports hybrid mode, delegating provider routing, auth, and usage tracking to [otari.ai](https://otari.ai) instead — see [Modes](../../docs/modes.md) for the concept.

A separate Blueprint, [`render.hybrid.yaml`](./render.hybrid.yaml), deploys hybrid mode:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/mozilla-ai/otari&path=deploy/render/render.hybrid.yaml)

### What gets deployed

| Resource | Plan | Details |
| --- | --- | --- |
| `otari-hybrid` | Free web service | `docker.io/mzdotai/otari:0.2.0`, Oregon, health check at `/health/readiness` |

No database is created. Otari keeps no local state in hybrid mode: users, budgets, and usage are managed by otari.ai instead.

### Configuration

| Variable | Set how | Notes |
| --- | --- | --- |
| `PORT`, `OTARI_PORT` | `8000` | Keeps Render's detected port aligned with Otari's listening port. |
| `OTARI_HOST` | `0.0.0.0` | Binds Otari on the container network. |
| `OTARI_AI_TOKEN` | you provide | The gateway token (`gw-...`) for this Otari instance. Create it in otari.ai under **Organisation > Gateways > Create token**. Setting this alone switches Otari into hybrid mode; no `OTARI_MODE` is needed. |

`OTARI_MASTER_KEY`, `OTARI_DATABASE_URL`, the pricing flags, and the migration/bootstrap flags from the standalone Blueprint don't apply here: hybrid mode has no local database or management endpoints to protect. Only `/health`, `/health/liveness`, `/health/readiness`, `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` are exposed. Chat requests use `Authorization: Bearer <otari-user-token>` issued by otari.ai, not a locally minted API key.

### Deploy

1. Click **Deploy to Render** above. The button passes `path=deploy/render/render.hybrid.yaml`.
2. Enter your `OTARI_AI_TOKEN`.
3. Review the single free web service, then apply the Blueprint.
4. Wait for `otari-hybrid` to become live, then copy its `*.onrender.com` URL from the Dashboard.

### Verify

```bash
export OTARI_URL=https://<your-service>.onrender.com

curl "$OTARI_URL/health"
curl "$OTARI_URL/health/readiness"
```

The `/health` response includes `"mode": "hybrid"` and platform reachability. Then verify a chat request using an otari.ai user token:

```bash
curl "$OTARI_URL/v1/chat/completions" \
  -H "Authorization: Bearer <otari-user-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "ping"}]
  }'
```

## Maintaining the Blueprint

When changing either Blueprint:

1. Validate it against Render's current schema (requires [Render CLI](https://render.com/docs/cli) v2.7.0 or newer):

   ```bash
   render blueprints validate deploy/render/render.yaml
   render blueprints validate deploy/render/render.hybrid.yaml
   ```
2. If a deploy link or Blueprint path changes, update this README, the project root [`README.md`](../../README.md), and `docs/deployment.md` together.
