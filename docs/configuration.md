# Configuration

The gateway is configured through a YAML file and environment variables.

## Config file

The config file is passed at startup with `--config`:

```bash
gateway serve --config config.yml
```

### Full example

```yaml
# Database
database_url: "postgresql://gateway:gateway@postgres:5432/gateway"

# Server
host: "0.0.0.0"
port: 8000

# Auth
master_key: "your-secret-master-key"

# Rate limiting (requests per minute per user, omit to disable)
# rate_limit_rpm: 60

# Prometheus metrics at /metrics
# enable_metrics: true

# Providers
providers:
  openai:
    api_key: "sk-..."
  anthropic:
    api_key: "sk-ant-..."
  mistral:
    api_key: "..."
  vertexai:
    credentials: "/app/service_account.json"
    project: "my-gcp-project"
    location: "us-central1"

# Pricing (USD per million tokens)
pricing:
  openai:gpt-4o:
    input_price_per_million: 2.50
    output_price_per_million: 10.00
```

### Config reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `database_url` | string | `sqlite:///./otari-gateway.db` | Database connection URL (SQLite or PostgreSQL) |
| `auto_migrate` | bool | `true` | Run Alembic migrations automatically on startup |
| `db_pool_size` | int | `10` | Persistent DB connections per worker |
| `db_max_overflow` | int | `20` | Extra burst connections above pool size |
| `db_pool_timeout` | float | `30.0` | Seconds to wait for a connection |
| `db_pool_recycle` | int | `-1` | Recycle connections older than N seconds (-1 = disabled) |
| `host` | string | `0.0.0.0` | Server bind host |
| `port` | int | `8000` | Server bind port |
| `master_key` | string | none | Master key for management endpoints |
| `rate_limit_rpm` | int | none | Max requests per minute per user (none = disabled) |
| `cors_allow_origins` | list | `[]` | Allowed CORS origins (empty = disabled) |
| `providers` | dict | `{}` | Provider credentials (see below) |
| `pricing` | dict | `{}` | Model pricing entries |
| `enable_metrics` | bool | `false` | Enable Prometheus `/metrics` endpoint |
| `enable_docs` | bool | `true` | Enable `/docs`, `/redoc`, `/openapi.json` |
| `bootstrap_api_key` | bool | `true` | Create a first-use API key on startup when none exist |
| `log_writer_strategy` | string | `"single"` | Usage log writing: `"single"` (inline) or `"batch"` (background) |
| `budget_strategy` | string | `"for_update"` | Budget validation: `"for_update"`, `"cas"`, or `"disabled"` |
| `mode` | string | `"standalone"` | Configured mode (`"standalone"` or `"platform"`). Effective behavior is driven by presence of `OTARI_PLATFORM_TOKEN`. |
| `platform` | dict | `{}` | otari.ai integration settings (`base_url`, timeouts, retries) |

## Environment variables

Environment variables with the `GATEWAY_` prefix override config file values. For example, `GATEWAY_PORT=9000` overrides `port: 8000` in the YAML.

The config file also supports `${ENV_VAR}` interpolation:

```yaml
master_key: "${MY_SECRET_KEY}"
```

### Common variables

| Variable | Description |
|----------|-------------|
| `GATEWAY_MASTER_KEY` | Master key for management endpoints |
| `GATEWAY_DATABASE_URL` | Database connection URL |
| `GATEWAY_HOST` | Server bind host |
| `GATEWAY_PORT` | Server bind port |
| `GATEWAY_AUTO_MIGRATE` | Auto-run migrations on startup |
| `GATEWAY_BOOTSTRAP_API_KEY` | Create first-use API key |

### Provider credentials

Provider API keys can be set as environment variables instead of in the config file. These are picked up directly by the underlying SDK.

These credentials are used for standalone deployments. When connected to otari.ai, local provider credentials are not used.

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `MISTRAL_API_KEY` | Mistral |
| `GEMINI_API_KEY` | Google Gemini |

### otari.ai variables

These are only relevant when running connected to [otari.ai](https://otari.ai). See [Modes](modes.md) for details.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTARI_PLATFORM_TOKEN` | -- | Gateway token from otari.ai (enables platform connection) |
| `ANY_LLM_PLATFORM_TOKEN` | -- | Legacy alias for `OTARI_PLATFORM_TOKEN` |
| `PLATFORM_BASE_URL` | -- | Base URL for the otari.ai API |
| `PLATFORM_RESOLVE_TIMEOUT_MS` | `5000` | Timeout for provider resolution calls |
| `PLATFORM_USAGE_TIMEOUT_MS` | `5000` | Timeout for usage reporting calls |
| `PLATFORM_USAGE_MAX_RETRIES` | `3` | Max retries for transient usage reporting failures |
| `STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS` | `2000` | Per-attempt timeout waiting for first streamed chunk |

`OTARI_PLATFORM_TOKEN` takes precedence. `ANY_LLM_PLATFORM_TOKEN` is supported for backward compatibility.

## Provider configuration

Each provider entry in the config file supports:

| Field | Description |
|-------|-------------|
| `api_key` | Provider API key |
| `api_base` | Custom API base URL (optional) |
| `client_args` | Extra client options: `custom_headers`, `timeout` (optional) |

### Vertex AI

Vertex AI requires additional fields instead of a simple API key:

```yaml
providers:
  vertexai:
    credentials: "/app/service_account.json"
    project: "my-gcp-project"
    location: "us-central1"
```

The `credentials` field points to a Google Cloud service account JSON file.

## Pricing

Model pricing is configured per model key (`provider:model_name`) in USD per million tokens:

```yaml
pricing:
  openai:gpt-4o:
    input_price_per_million: 2.50
    output_price_per_million: 10.00
  anthropic:claude-sonnet-4-6:
    input_price_per_million: 3.00
    output_price_per_million: 15.00
```

Config pricing sets initial values. Pricing set via the `/v1/pricing` API takes precedence.
