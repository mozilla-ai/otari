# Deploy Otari on Railway

One-click deploy of a self-hosted [Otari](https://github.com/mozilla-ai/otari)
gateway in front of key-only providers (OpenAI, Anthropic, Mistral, Gemini),
backed by a managed Postgres database and a bucket for uploaded files. No local
setup: deploy, then add your provider keys on the dashboard.

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/otari)

## What you get

The template stands up two services and a bucket:

| Service | Source | Notes |
| --- | --- | --- |
| **otari** | `mzdotai/otari:0.14.1` (Docker Hub) | Target port `8000` with a public domain, healthcheck `/api/v1/health/readiness`, which fails while the database is unreachable. Pulls the published image, pinned to a release; builds nothing. See [Upgrade](#upgrade). |
| **Postgres** | Railway managed | Durable storage for keys, users, budgets, and usage. |
| **otari-files** | Railway bucket | S3-compatible object storage for the bytes behind the Files API: uploads, attachments, and files a sandbox produced. They survive a redeploy and every replica reads the same bucket. Billed per stored GB; see [Files](#files). |

Otari is a good fit for a one-click deploy: the app keeps no state on its own
disk, its stateful dependencies are Postgres and the bucket, the image is
published, and `auto_migrate` plus `bootstrap_api_key` are on by default, so the
schema is created and a first-use API key is minted on startup with no extra
steps.

## Configuration

The template wires the two services together and generates the keys it needs. All of
Otari's scalar config is reachable through `OTARI_<FIELD>` environment variables;
the snapshot of what the template sets lives in [`template.json`](template.json).

| Variable | Value | Notes |
| --- | --- | --- |
| `OTARI_DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Pre-wired; leave as-is. |
| `OTARI_MASTER_KEY` | auto-generated (`${{secret(48)}}`) | Auto-set; read it from the otari service's Variables tab. |
| `OTARI_SECRET_KEY` | auto-generated Fernet key | Encrypts the provider credentials you add on the Providers page. Keep it: losing it makes them unrecoverable. |
| `OTARI_PROVIDER_ACCOUNT_PEPPER` | auto-generated (`${{secret(43)}}`) | Names the provider account a copy of an attached file is in. Otari refuses to start without it while provider copies are on. |
| `OTARI_REQUIRE_PRICING` | `false` | Pre-set, so a fresh deploy serves models that have no configured pricing. |
| `OTARI_DEFAULT_PRICING` | `true` | Pre-set, so common models are metered from the bundled genai-prices dataset without configuring each one. Prices you set in the dashboard or via `/api/v1/pricing` always override it. |
| `OTARI_FORWARDED_ALLOW_IPS` | `*` | Pre-set, so the per-IP sign-in and public-catalog limits see the real client, not Railway's ingress. Safe here because the ingress is the only path to the container; see [Behind a reverse proxy](../../docs/deployment.md#behind-a-reverse-proxy). |
| `OTARI_PUBLIC_BASE_URL` | `https://${{RAILWAY_PUBLIC_DOMAIN}}` | Pre-wired to the service's Railway domain. OAuth redirect URIs, the passkey relying-party ID and email links are built from it. Change it if you attach a custom domain. |
| `OTARI_FILES_BACKEND` | `s3` | Pre-set, so uploaded files go to the bucket rather than the container's disk; see [Files](#files). |
| `OTARI_FILES_S3_BUCKET` | `${{otari-files.BUCKET}}` | Pre-wired to the bucket's S3 name; leave as-is. |
| `OTARI_FILES_S3_ENDPOINT_URL` | `${{otari-files.ENDPOINT}}` | Pre-wired to the bucket's S3 endpoint; leave as-is. |
| `OTARI_FILES_S3_REGION` | `${{otari-files.REGION}}` | Pre-wired to the bucket's region, which Railway reports as `auto`; leave as-is. |
| `AWS_ACCESS_KEY_ID` | `${{otari-files.ACCESS_KEY_ID}}` | Pre-wired to the bucket's access key. Otari's S3 backend takes credentials from boto3's default chain, so they travel under the AWS names; see the Bedrock note under [Files](#files). |
| `AWS_SECRET_ACCESS_KEY` | `${{otari-files.SECRET_ACCESS_KEY}}` | Pre-wired to the bucket's secret key; leave as-is. |
| `PORT` | `8000` | Pre-set for Railway's deploy-time healthcheck, which probes the port in `PORT`, not the target port. Otari listens on `OTARI_PORT` (pinned to `8000` in the image) and never reads `PORT`, so keep the two equal. |

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

## Files

The [Files API](../../docs/files.md) is on by default, and its bytes need a
home that outlives the container. The image default is the `local` backend
under `/app/otari-files`, which on Railway is the container's own disk: every
redeploy, including an image auto-update, starts from a fresh disk, and each
replica has a disk of its own. A file's row would stay in Postgres while its
bytes vanish, so every later read of it answers 404 with no error at upload
time. The template therefore sets the `s3` backend and wires it to the
`otari-files` bucket, a Railway-managed S3-compatible store. File metadata stays
in Postgres and the bytes live in the bucket, so file storage no longer pins the
otari service to one replica (an in-memory rate limit still counts per replica;
see `rate_limit_store`). See [Storage backends](../../docs/files.md#storage-backends).

Three things to know:

- The bucket's keys reach the gateway as `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY`, which every boto3 client in the process reads, not
  only the files backend. A Bedrock provider entry with no credentials of its
  own would sign its requests with the bucket's keys, which AWS rejects, so give
  Bedrock its own keys on the Providers page or in `OTARI_CONFIG_YAML`.
- Railway bills a bucket per stored GB and keeps no backups of it. A deleted
  bucket stays restorable for 52 hours, after which its objects are gone. The
  file sweep (`files_retention_hours`, off by default) is how to bound what
  accumulates.
- To run without the bucket, do it in this order. First set
  `OTARI_FILES_ENABLED=false` on the otari service and let it redeploy. Then
  delete the bucket and the six `otari-files` variables. The order matters:
  with files on, the gateway refuses to start while its bucket settings point
  at nothing; with files off it starts and logs that stored file references no
  longer resolve. Files uploaded before stop being served, and
  `OTARI_PROVIDER_ACCOUNT_PEPPER` goes unused. A template deploy cannot leave
  the bucket out: Railway creates every service and bucket a template holds,
  and only its variables are open to the deployer.

## Deploy

1. Click **Deploy on Railway** above.
2. Deploy. The keys are generated for you. Railway provisions Postgres, pulls
   the Otari image, gives the otari service a public `*.up.railway.app` domain,
   runs migrations on startup, and bootstraps a first-use API key.
3. Open that domain (Settings → Networking), sign in with the master key (from the Variables tab), and
   add at least one provider on the Providers page.

## Verify

Once both services are healthy:

```bash
# Replace with your service's public domain.
export OTARI_URL=https://your-otari.up.railway.app

curl "$OTARI_URL/api/v1/health/readiness"
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

## Upgrade

The template pins the Otari image to a release tag, not `latest`. Otari runs
its migrations on startup, and Railway redeploys on its own (a crash restart, a
host move, a variable change). On a moving tag, any of those could pull a newer
release and migrate your schema without you choosing to upgrade. Migrations
only go forward, so going back needs a database restore.

To upgrade:

1. Back up Postgres. Railway's Postgres service has a **Backups** tab; or run
   `pg_dump` against its public URL.
2. Read the [release notes](https://github.com/mozilla-ai/otari/releases) for
   every release between yours and the target.
3. Add any variable those releases newly require to the otari service. The
   template only reaches new deploys, so an existing one never gets a variable
   added to it later. The release notes name such variables; [`template.json`](template.json) at the target
   release's tag shows what a new deploy would get.
4. On the otari service, open Settings → Source and change the image tag to the
   target release (for example `0.15.0`). Railway
   redeploys, and Otari migrates the schema on startup.
5. Check `/api/v1/health/readiness` and make one real request, as in [Verify](#verify).

To take patch releases without doing this by hand, turn on Railway's
[Image Auto Updates](https://docs.railway.com/deployments/image-auto-updates):
on the otari service, open Settings → Source → **Configure Auto Updates**,
choose **Patches only**, and pick a maintenance window. Railway then moves the
pinned tag within its minor line (`0.14.1` to `0.14.2`) and notifies workspace
admins. Avoid **Minor updates and patches**: a minor release can migrate the
schema, and the backup Railway takes before an update covers the otari
service's volumes, not the Postgres service.

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
   matches the live config (services, the bucket, variables with their
   descriptions, defaults, target port, public domain).
   The README on Railway is [`listing.md`](listing.md), not this file: Railway
   requires its own fixed sections and absolute links, and it drops anything in
   angle brackets as HTML. Edit `listing.md`, then paste it into the template
   editor's README field.
3. To move a new deploy to a newer release, change the image tag on the live
   template and in `template.json`. This does not touch existing deploys: each
   keeps the tag it was deployed with until its operator upgrades.
4. The deploy link is `https://railway.com/deploy/<code>`, where the code is
   set in the template editor (it is `otari`). If it changes, update every
   **Deploy on Railway** button (`grep -rn railway.com/deploy`) and the
   `code` in `template.json`.
5. Run `make railway-template-check`. It reads the live template from
   Railway's public API (no token) and lists every difference from
   `template.json` and `listing.md`: images, bucket names, healthcheck path,
   domain port, and each variable's default, optional flag and description. Fix either side
   until it passes. CI runs the same check on every PR that touches this
   directory and once a week, so an edit made in the Railway editor without a
   PR still shows up. After a release that changes required config, do steps
   1 to 5 again.

Listing the template in Railway's public marketplace is optional: the deploy
link works without it. Publishing only adds marketplace discoverability and
usage-kickback eligibility.
