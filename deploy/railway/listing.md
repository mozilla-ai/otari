# Deploy and Host Otari on Railway

[Otari](https://github.com/mozilla-ai/otari) is an open-source, OpenAI-compatible LLM gateway you run yourself. Put one endpoint in front of providers like OpenAI, Anthropic, Mistral, and Gemini, then issue virtual API keys, enforce budgets, and track usage and cost in one place. It speaks the OpenAI Chat Completions and Responses APIs plus the Anthropic Messages API.

## About Hosting Otari

This template runs two services and a storage bucket: the Otari gateway from its published Docker image (`mzdotai/otari:0.15.0`, pinned to a release), a managed PostgreSQL database, and a Railway bucket for uploaded files. Otari keeps no state on its own disk. Postgres holds your keys, users, budgets, and usage; the bucket holds the files you upload through the Files API, so they survive a redeploy and every replica reads the same files.

On first boot Otari runs its database migrations and prints a first-use API key in the deploy logs, so the gateway is usable right away. The template generates the master key, the key that encrypts stored provider credentials, and the provider account pepper for you. The gateway listens on port `8000`, with a healthcheck at `/api/v1/health/readiness`, so a deploy that cannot reach its database does not go live.

After the deploy:

1. Open the otari service's public domain.
2. Sign in with the master key (`OTARI_MASTER_KEY` on the Variables tab).
3. Add at least one provider on the Providers page.

Then make a request:

```bash
curl "https://YOUR_DOMAIN/api/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "Say hello."}]}'
```

### What the template sets

| Variable | Value | Notes |
| --- | --- | --- |
| `OTARI_DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Pre-wired; leave as-is. |
| `OTARI_MASTER_KEY` | auto-generated | Signs you in to the dashboard and the management API. |
| `OTARI_SECRET_KEY` | auto-generated | Encrypts the provider credentials you add. Keep it: losing it makes them unrecoverable. |
| `OTARI_PROVIDER_ACCOUNT_PEPPER` | auto-generated | Required at startup while provider file copies are on. |
| `OTARI_REQUIRE_PRICING` | `false` | Serves models that have no configured pricing. |
| `OTARI_DEFAULT_PRICING` | `true` | Meters common models from community-maintained rates. Prices you set on the Models page always win. |
| `OTARI_FORWARDED_ALLOW_IPS` | `*` | Lets per-IP sign-in limits see the real client, not Railway's ingress. |
| `OTARI_PUBLIC_BASE_URL` | `https://${{RAILWAY_PUBLIC_DOMAIN}}` | Your service's public URL, for sign-in redirects, passkeys and email links. Change it if you attach a custom domain. |
| `OTARI_FILES_BACKEND` | `s3` | Keeps uploaded files in the bucket, not on the container's disk. |
| `OTARI_FILES_S3_BUCKET` | `${{otari-files.BUCKET}}` | Pre-wired to the bucket's S3 name. |
| `OTARI_FILES_S3_ENDPOINT_URL` | `${{otari-files.ENDPOINT}}` | Pre-wired to the bucket's S3 endpoint. |
| `OTARI_FILES_S3_REGION` | `${{otari-files.REGION}}` | Pre-wired to the bucket's region. |
| `AWS_ACCESS_KEY_ID` | `${{otari-files.ACCESS_KEY_ID}}` | Pre-wired to the bucket's access key. |
| `AWS_SECRET_ACCESS_KEY` | `${{otari-files.SECRET_ACCESS_KEY}}` | Pre-wired to the bucket's secret key. |
| `PORT` | `8000` | The port Railway's healthcheck probes. Keep it equal to the target port. |

To keep provider keys in Railway variables instead of the dashboard, declare the providers through `OTARI_CONFIG_YAML`; see [Full config via environment](https://github.com/mozilla-ai/otari/blob/main/docs/configuration.md#full-config-via-environment).

### Upgrading

The image is pinned to a release, so a redeploy never migrates your schema by surprise. To upgrade, back up Postgres, read the [release notes](https://github.com/mozilla-ai/otari/releases), add any variable they say a release now requires (an existing deploy never gets new template variables), then change the image tag under Settings → Source. To take patch releases automatically, turn on Image Auto Updates with **Patches only**.

## Common Use Cases

- Give a team one OpenAI-compatible endpoint in front of several providers
- Issue and revoke virtual API keys per app or user without sharing raw provider keys
- Enforce budgets and track spend and token usage across models

## Dependencies for Otari Hosting

- A PostgreSQL database (provisioned by this template)
- A storage bucket for uploaded files (provisioned by this template; billed per stored GB)
- At least one provider API key (OpenAI, Anthropic, Mistral, Gemini, or any [any-llm provider](https://docs.mozilla.ai/any-llm/providers/)), added on the dashboard after deploy

### Deployment Dependencies

- Otari repository: https://github.com/mozilla-ai/otari
- Otari documentation: https://github.com/mozilla-ai/otari/tree/main/docs
- Railway deploy guide: https://github.com/mozilla-ai/otari/blob/main/deploy/railway/README.md
- Docker image: https://hub.docker.com/r/mzdotai/otari

## Why Deploy Otari on Railway?

Railway is a singular platform to deploy your infrastructure stack. Railway will host your infrastructure so you don't have to deal with configuration, while allowing you to vertically and horizontally scale it.

By deploying Otari on Railway, you are one step closer to supporting a complete full-stack application with minimal burden. Host your servers, databases, AI agents, and more on Railway.
