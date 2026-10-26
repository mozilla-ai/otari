# Deployment

Otari publishes `mzdotai/otari` on Docker Hub. Use the
[Self-hosted quickstart](self-hosted-quickstart.md) for local setup; this page covers production choices.

## Production checklist

A durable deployment should:

- use PostgreSQL and back it up
- set a strong master key in a secret store
- set and back up `OTARI_SECRET_KEY`, which encrypts the provider credentials it stores
- set `OTARI_PROVIDER_ACCOUNT_PEPPER` to its own random value to turn on provider
  copies of attached files, which are off without it (see
  [Provider copies](configuration.md#provider-copies))
- configure explicit pricing or deliberately enable default pricing
- terminate TLS in front of the gateway
- restrict database and management access
- monitor `/api/v1/health/readiness` and, with `enable_metrics` (see
  [Common settings](configuration.md#common-settings)), `/metrics`
- pin an image version, [verify it](#verify-the-image), and test migrations before upgrading

The default SQLite database is intended for evaluation and single-node local use.

### Behind a reverse proxy

When TLS ends at a proxy or a platform ingress, each request reaches Otari from
the proxy's address. The dashboard sign-in limit and the public-catalog limit
count failures per client address, so all visitors then share one budget. Ten
wrong passwords from one visitor lock everyone out of the dashboard for a
minute.

Set `forwarded_allow_ips` (`OTARI_FORWARDED_ALLOW_IPS`) to the proxy's
addresses or networks, comma-separated, so Otari reads the client address from
`X-Forwarded-For`. Use `*` only where the proxy is the sole path to the
container, as on Railway: any peer in the list can set its own client address.
When it is unset, uvicorn's `FORWARDED_ALLOW_IPS` applies, and without that
only `127.0.0.1` is trusted.

### Watch the connection pool

On PostgreSQL the gateway serves requests from a fixed pool of database
connections, and running out of them makes every request fail at once. Alert
when checked-out connections stay close to capacity for a sustained period,
from these `/metrics` series, labeled by `pool` (`request` for request traffic,
`log` for the usage-log writer):

| Metric | Meaning |
| --- | --- |
| `gateway_db_pool_connections_checked_out` | Connections in use right now |
| `gateway_db_pool_connections_idle` | Connections available to hand out |
| `gateway_db_pool_overflow_connections` | Connections open beyond the pool's base size |
| `gateway_db_pool_capacity` | Ceiling on connections the pool hands out at once: `db_pool_size` plus `db_max_overflow` for `request`, `db_log_pool_size` for `log`, which has no overflow |

SQLite keeps no pool, so these series do not appear there. The pool settings
are in [Database connections](configuration.md#database-connections).

### Watch providers and refusals

`/metrics` also reports each upstream attempt, labeled by the provider instance
and the model a usage row names, so a dashboard can follow one provider's
errors, latency and spend without reading the usage log:

| Metric | Labels | Meaning |
| --- | --- | --- |
| `gateway_upstream_attempts_total` | `provider`, `model`, `outcome` (`success` or `error`), `status_code`, `fallback` | One per attempt on the completion, embedding, image, audio, moderation and rerank routes, including a routing policy's failed candidates. `status_code` is `200` for a success, the HTTP status that classified a failure (a timeout is `504`, an unclassified connection failure `502`), or `none` for a failure that carried no status, such as a stream that ended without reporting its usage. `fallback` is `true` for a candidate after the plan's first |
| `gateway_attempt_duration_seconds` | `provider`, `model`, `outcome` | From the start of the request to the end of the attempt, observed for a plan's first candidate only, since a fallback's would include its predecessors' time |
| `gateway_time_to_first_token_seconds` | `provider`, `model` | From the start of the request to a streamed attempt's first chunk, for a plan's first candidate only |
| `gateway_tokens_total` | `provider`, `model`, `type` | Input and output tokens |
| `gateway_request_cost_dollars` | `provider`, `model` | Each request's cost in USD |
| `gateway_abandoned_attempts_total` | `provider`, `model`, `reason`, `position` | Attempts given up before their first chunk |
| `gateway_refusals_total` | `code`, `budget_scope` | Responses refused with an [`Otari-Error-Code`](api-reference.md#error-codes), by that code and, for `budget_exceeded`, the scope that refused (`user`, `api_token`, ...; empty otherwise) |

`model` is the model as the provider was sent it. A request that names its own
`provider:model` adds a series per distinct model, so restrict keys to the
models they may use where callers are not trusted to pick.

### Verify the image

Each published image carries a signed build provenance attestation, which ties it
to the `mozilla-ai/otari` repository and the workflow that built it. Check it with
the GitHub CLI before you deploy a tag:

```sh
gh attestation verify oci://docker.io/mzdotai/otari:<tag> --repo mozilla-ai/otari
```

The command fails for an image that this repository's publishing workflow did not
build, such as one pushed to Docker Hub with stolen credentials.

## Docker Compose

The repository Compose stack runs Otari and PostgreSQL. The
[Self-hosted quickstart](self-hosted-quickstart.md) walks through its first start, including the
`.env` secrets it needs.

Use `docker-compose.build.yml` when testing source changes:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up --build
```

## Managed deployment templates

- [Railway](../deploy/railway/README.md)

That page owns its current image tag, variables, platform limits, and
maintenance instructions.

## Optional services

Compose profiles start the bundled service backends:

```bash
docker compose --profile code-exec up -d
docker compose --profile web-search up -d
docker compose --profile guardrails up -d
docker compose --profile object-storage up -d
```

The `object-storage` profile runs an S3-compatible store for uploaded files;
[Object storage with Docker Compose](files.md#object-storage-with-docker-compose)
has the settings that use it.

Web search needs no container when `web_search_provider` names a licensed API
(`tavily` or `brave`) and `web_search_provider_api_key` carries its key. These
services are optional; requests that require an unconfigured backend fail
without affecting ordinary inference.

See [Built-in tools](tools.md) and [Guardrails](guardrails.md).

## Upgrading from hosted or hybrid mode

A deployment that ran in hosted mode serves inference and the Playground as
well as management. Set `OTARI_PROVIDER_ACCOUNT_PEPPER` to keep provider copies
of attached files on: without it, startup turns them off and logs a warning.
`web_search_workspace_default` and `code_execution_workspace_default` default
to `true`; set them to `false` to keep web access and code execution off for a
workspace with no policy of its own.

A gateway that ran in hybrid mode serves from its own database too: leftover
`OTARI_MODE`, `OTARI_AI_TOKEN` and `OTARI_DATA_PLANE_URL` are ignored, so
declare its providers and create its keys and budgets as a new deployment does.

## Configuration

Otari accepts a mounted YAML file, scalar environment variables, or a complete
YAML document in `OTARI_CONFIG_YAML` or `OTARI_CONFIG_B64`. See
[Configuration](configuration.md) for precedence and security notes.
