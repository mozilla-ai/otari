# Deployment

Otari publishes `mzdotai/otari` on Docker Hub. For local setup, use the
[Self-hosted quickstart](self-hosted-quickstart.md); this page covers production.

## Production checklist

- Use PostgreSQL and back it up. The default SQLite database is for evaluation
  and single-node local use.
- Keep a strong master key in a secret store.
- Set and back up `OTARI_SECRET_KEY`, which encrypts stored provider credentials.
  Only a SQLite deployment generates one for itself, in a file beside the
  database (see
  [Runtime provider management](configuration.md#runtime-provider-management)).
- Set `OTARI_PROVIDER_ACCOUNT_PEPPER` to its own random value. Without it,
  provider copies of attached files are off (see
  [Provider copies](configuration.md#provider-copies)).
- Configure explicit pricing, or turn on default pricing deliberately.
- Terminate TLS in front of the gateway.
- Restrict database and management access.
- Monitor `/api/v1/health/readiness` and, with `enable_metrics` (see
  [Common settings](configuration.md#common-settings)), `/metrics`.
- Pin an image version, [verify it](#verify-the-image), and test migrations
  before upgrading.

### Behind a reverse proxy

Set `forwarded_allow_ips` (`OTARI_FORWARDED_ALLOW_IPS`) to the proxy's
addresses or networks, comma-separated, so Otari reads the client address from
`X-Forwarded-For`.

Without it, every request seems to come from the proxy. The dashboard sign-in
limit and the public-catalog limit count failures per client address, so ten
wrong passwords from one visitor lock everyone out of the dashboard for a
minute. A [country file](#the-country-file) also looks up the client address.

Use `*` only where the proxy is the only path to the container, as on Railway:
any peer in the list can set its own client address. When the setting is unset,
uvicorn's `FORWARDED_ALLOW_IPS` applies, and without that only `127.0.0.1` is
trusted.

### Watch the connection pool

On PostgreSQL, requests share a fixed pool of database connections, and when
it runs out every request fails at once. Alert when checked-out connections
stay close to capacity. These `/metrics` series carry a `pool` label (`request`
for request traffic, `log` for the usage-log writer):

| Metric | Meaning |
| --- | --- |
| `gateway_db_pool_connections_checked_out` | Connections in use right now |
| `gateway_db_pool_connections_idle` | Connections available to hand out |
| `gateway_db_pool_overflow_connections` | Connections open beyond the pool's base size |
| `gateway_db_pool_capacity` | Most connections the pool hands out at once: `db_pool_size` plus `db_max_overflow` for `request`, `db_log_pool_size` for `log` (no overflow) |

SQLite has no pool, so these series do not appear there. The pool settings are
in [Database connections](configuration.md#database-connections).

### Watch providers and refusals

`/metrics` reports each upstream attempt by provider instance and model, so a
dashboard can follow one provider's errors, latency and spend:

| Metric | Labels | Meaning |
| --- | --- | --- |
| `gateway_upstream_attempts_total` | `provider`, `model`, `outcome` (`success` or `error`), `status_code`, `fallback` | One per attempt on the completion, embedding, image, audio, moderation and rerank routes, including a routing policy's failed candidates. `status_code` is `200` for a success, the HTTP status of a failure (a timeout is `504`, an unclassified connection failure `502`), or `none` for a failure with no status, such as a stream that ended without reporting usage. `fallback` is `true` for any candidate after the first |
| `gateway_attempt_duration_seconds` | `provider`, `model`, `outcome` | From the start of the request to the end of the attempt, for the first candidate only |
| `gateway_time_to_first_token_seconds` | `provider`, `model` | From the start of the request to a streamed attempt's first chunk, for the first candidate only |
| `gateway_tokens_total` | `provider`, `model`, `type` | Input and output tokens |
| `gateway_request_cost_dollars` | `provider`, `model` | Each request's cost in USD |
| `gateway_abandoned_attempts_total` | `provider`, `model`, `reason`, `position` | Attempts given up before their first chunk |
| `gateway_refusals_total` | `code`, `budget_scope` | Responses refused with an [`Otari-Error-Code`](https://docs.otari.ai/reference#error-codes), by code and, for `budget_exceeded`, the refusing scope (`user`, `api_token`, ...; empty otherwise) |

`model` is the model name sent to the provider. Each distinct `provider:model`
a caller names adds a series, so where callers are not trusted to pick, restrict
keys to the models they may use.

### Verify the image

Each published image carries a signed build provenance attestation tying it to
the `mozilla-ai/otari` repository and the workflow that built it. Check it
before you deploy a tag:

```sh
gh attestation verify oci://docker.io/mzdotai/otari:<tag> --repo mozilla-ai/otari
```

The command fails for an image this repository's workflow did not build, such
as one pushed to Docker Hub with stolen credentials.

## Docker Compose

The repository's Compose stack runs Otari and PostgreSQL. The
[Self-hosted quickstart](self-hosted-quickstart.md) covers its first start and
the `.env` secrets it needs.

To test source changes, add `docker-compose.build.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up --build
```

## Managed deployment templates

- [Railway](https://github.com/mozilla-ai/otari/blob/db422245a21a835223ae174613da498f3b2b3f5c/deploy/railway/README.md): that page owns its image tag,
  variables, platform limits and maintenance steps.

## Optional services

Compose profiles start the bundled service backends:

```bash
docker compose --profile code-exec up -d
docker compose --profile web-search up -d
docker compose --profile object-storage up -d
```

- `object-storage` runs an S3-compatible store for uploaded files; see
  [Object storage with Docker Compose](https://docs.otari.ai/reference/gateway/files#object-storage-with-docker-compose).
- Web search needs no container when `web_search_provider` names a licensed API
  (`tavily` or `brave`) and `web_search_provider_api_key` holds its key.

A request that needs a backend you have not configured fails; ordinary
inference is unaffected. See [Built-in tools](https://docs.otari.ai/reference/gateway/tools) and
[Guardrails](https://docs.otari.ai/reference/gateway/guardrails-reference).

## Upgrading from hosted or hybrid mode

From hosted mode:

- Set `OTARI_PROVIDER_ACCOUNT_PEPPER` to keep provider copies of attached files
  on. Without it, startup turns them off and logs a warning.
- `web_search_workspace_default` and `code_execution_workspace_default` default
  to `true`. Set them to `false` to keep web access and code execution off for
  workspaces with no policy of their own.

From hybrid mode, the gateway now serves from its own database. Leftover
`OTARI_MODE`, `OTARI_AI_TOKEN` and `OTARI_DATA_PLANE_URL` are ignored; declare
providers and create keys and budgets as for a new deployment.

## The country file

A provider that serves only some countries needs each request's country
([Supported countries](configuration.md#supported-countries)). Where no proxy
sets a country header, Otari looks up the client address in a MaxMind DB
country file, such as GeoLite2 Country or DB-IP Lite Country. The Docker image
has the reader but not the file.

The simplest way to provide it: upload the file in the **Country file** card on
the dashboard's **Settings** page. Otari stores it in the database, so every
replica reads it. Upload each new release to keep it current (DB-IP Lite
monthly, GeoLite2 twice a week).

DB-IP Lite is under CC BY 4.0: credit DB-IP where your service names its data
sources, for example with a link to `https://db-ip.com`.

### A mounted country file

Where you control the containers, mount the folder that holds the file and set
its path. In Compose, with the file at `./geoip/countries.mmdb`, add to the
`otari` service:

```yaml
    volumes:
      - ./geoip:/app/geoip:ro
    environment:
      - OTARI_CLIENT_COUNTRY_DATABASE=/app/geoip/countries.mmdb
```

- Otari rereads a replaced file without a restart, so a scheduled job such as
  [geoipupdate](https://github.com/maxmind/geoipupdate) keeps it current.
- Replace the file atomically (write a new file, then rename it over the old
  one).
- Mount the folder, not the file: a single-file mount keeps the old file after
  a rename.
- Every replica needs its own copy.
- While `client_country_database` is set, an uploaded file is not read.

## Configuration

Otari accepts a mounted YAML file, scalar environment variables, or a complete
YAML document in `OTARI_CONFIG_YAML` or `OTARI_CONFIG_B64`. See
[Configuration](configuration.md) for precedence and security notes.
