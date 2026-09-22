# Deployment

Otari publishes `mzdotai/otari` on Docker Hub. Use the
[Quickstart](quickstart.md) for local setup; this page covers production choices
and hybrid gateways.

## Standalone checklist

A durable standalone deployment should:

- use PostgreSQL and back it up
- set a strong master key in a secret store
- set and back up `OTARI_SECRET_KEY` when storing provider credentials
- configure explicit pricing or deliberately enable default pricing
- terminate TLS in front of the gateway
- restrict database and management access
- monitor `/api/v1/health/readiness` and, when enabled, `/metrics`
- pin an image version and test migrations before upgrading

The default SQLite database is intended for evaluation and single-node local use.

### Watch the connection pool

On PostgreSQL the gateway serves requests from a fixed pool of database
connections, and running out of them makes every request fail at once. `/metrics`
reports the pool directly, labeled by `pool` (`request` for request traffic,
`log` for the usage-log writer):

| Metric | Meaning |
| --- | --- |
| `gateway_db_pool_connections_checked_out` | Connections in use right now |
| `gateway_db_pool_connections_idle` | Connections available to hand out |
| `gateway_db_pool_overflow_connections` | Connections open beyond the pool's base size |
| `gateway_db_pool_capacity` | Ceiling on connections the pool hands out at once |

The `request` pool is sized by `db_pool_size` and may open `db_max_overflow`
connections beyond it, so its capacity is the two added together. The `log`
pool is sized by `db_log_pool_size` and has no overflow, so its capacity is
that value.

Alert on checked-out connections approaching capacity for a sustained period.
These metrics do not appear on SQLite, which opens a connection per use and
keeps no pool.

## Docker Compose

The repository Compose stack runs Otari and PostgreSQL:

```bash
cp config.example.yml config.yml
docker compose pull
docker compose up -d
```

Use `docker-compose.build.yml` when testing source changes:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up --build
```

## Managed deployment templates

- [Render](../deploy/render/README.md)
- [Railway](../deploy/railway/README.md)

Those pages own their current image tags, variables, platform limits, and
maintenance instructions.

## Connect a gateway to otari.ai

Create a gateway in otari.ai and copy its gateway token, then start Otari in
hybrid mode. Put the token in a private environment file:

```dotenv
OTARI_AI_TOKEN=gw_your_gateway_token
```

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  mzdotai/otari:0.4.0 \
  otari serve
```

Hybrid mode does not initialize a local management database or use local
provider credentials. Clients send an otari.ai user token to the gateway:

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer tk_your_user_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Say hello."}]
  }'
```

Verify both liveness and control-plane reachability:

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/readiness
```

See [Runtime modes](modes.md) for the trust and credential model.

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

## Configuration

Otari accepts a mounted YAML file, scalar environment variables, or a complete
YAML document in `OTARI_CONFIG_YAML` or `OTARI_CONFIG_B64`. See
[Configuration](configuration.md) for precedence and security notes.
