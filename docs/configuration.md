# Configuration

Otari is configured through a YAML file and environment variables.

## Config file

The config file is passed at startup with `--config`:

```bash
otari serve --config config.yml
```

### Full example

```yaml
# Database
database_url: "postgresql://otari:otari@postgres:5432/otari"

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
| `database_url` | string | `sqlite:///./otari.db` | Database connection URL (SQLite or PostgreSQL) |
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
| `require_pricing` | bool | `true` | Reject requests for models with no configured pricing (HTTP 402, fail-closed). When `false`, unpriced models are served and logged without cost. Audio and moderation endpoints are always exempt. |
| `default_pricing` | bool | `false` | When a model has no pricing in the database, fall back to community-maintained defaults from the bundled genai-prices dataset. Off by default (opt-in). Database pricing always wins. See [Default pricing](#default-pricing). |
| `reject_user_mismatch` | bool | `true` | When `true`, a non-master key whose request names a `user` other than its own is rejected (HTTP 403). When `false`, the client `user` is still forwarded to the provider but spend is always bound to the key's own user. The master key may always bill an arbitrary user. |
| `stream_missing_usage_policy` | string | `"estimate"` | How to bill a streamed response that completes with no provider usage data: `"estimate"` (charge the up-front estimate), `"fail"` (charge estimate and mark errored), or `"allow_free"` (don't bill). |
| `budget_estimate_default_output_tokens` | int | `1024` | Output-token count assumed when reserving budget for a request with no declared max output; reconciled to actual usage on completion. |
| `mode` | string | `"standalone"` | Configured mode (`"standalone"` or `"hybrid"`; the legacy value `"platform"` is still accepted). Effective behavior is driven by presence of `OTARI_AI_TOKEN`. |
| `platform` | dict | `{}` | otari.ai integration settings (`base_url`, timeouts, retries) |

## Environment variables

The following `OTARI_` variables override config file values for their matching fields. For example, `OTARI_PORT=9000` overrides `port: 8000` in the YAML.

`OTARI_` overrides apply to scalar fields (strings, numbers, booleans). List and dict fields (`cors_allow_origins`, `providers`, `pricing`, and the `platform` block) are not read from individual `OTARI_` variables; set them in the YAML file, or supply the whole config through the environment with `OTARI_CONFIG_YAML` / `OTARI_CONFIG_B64` (see [Full config via environment](#full-config-via-environment)). The `platform` block also has dedicated `PLATFORM_*` variables (see the otari.ai variables below).

The config file also supports `${ENV_VAR}` interpolation:

```yaml
master_key: "${MY_SECRET_KEY}"
```

### Full config via environment

On PaaS platforms (Railway, Render, Fly.io, Kubernetes) where mounting a `config.yml` is awkward, you can supply the entire config, including the non-scalar `providers` and `pricing` fields, through the environment. This reaches the full schema with no file mount and no custom image.

| Variable | Description |
|----------|-------------|
| `OTARI_CONFIG_YAML` | The full config as raw YAML, parsed exactly like a `config.yml`. |
| `OTARI_CONFIG_B64` | The same YAML, base64-encoded, for env-var UIs that mangle multiline values. |

The YAML supports the same `${ENV_VAR}` interpolation as a config file, so you can keep secrets in separate variables:

```yaml
# value of OTARI_CONFIG_YAML
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    api_base: https://my-proxy.example/v1
pricing:
  openai:gpt-4o:
    input_price_per_million: 2.5
    output_price_per_million: 10
```

Precedence, lowest to highest: the config file, then the env-structured config (`OTARI_CONFIG_YAML` or `OTARI_CONFIG_B64`, with raw YAML winning if both are set), then scalar `OTARI_<FIELD>` overrides. Env-structured keys replace the matching top-level keys from the file. Invalid base64, invalid YAML, or a non-mapping top level fails fast at startup with a clear error.

Note the `require_pricing` interaction: it defaults to `true` (fail-closed), so the gateway rejects any model without configured pricing. An env-only deploy therefore needs either `pricing` entries (as above) or `OTARI_REQUIRE_PRICING=false` to serve unpriced models.

### Common variables

| Variable | Description |
|----------|-------------|
| `OTARI_MASTER_KEY` | Master key for management endpoints |
| `OTARI_DATABASE_URL` | Database connection URL |
| `OTARI_HOST` | Server bind host |
| `OTARI_PORT` | Server bind port |
| `OTARI_AUTO_MIGRATE` | Auto-run migrations on startup |
| `OTARI_BOOTSTRAP_API_KEY` | Create first-use API key |

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
| `OTARI_AI_TOKEN` | -- | Otari token from otari.ai (enables platform connection) |
| `PLATFORM_RESOLVE_TIMEOUT_MS` | `5000` | Timeout for provider resolution calls |
| `PLATFORM_USAGE_TIMEOUT_MS` | `5000` | Timeout for usage reporting calls |
| `PLATFORM_USAGE_MAX_RETRIES` | `3` | Max retries for transient usage reporting failures |
| `STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS` | `2000` | Per-attempt timeout waiting for first streamed chunk |

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

A pricing entry whose provider is not listed in the `providers` section is skipped at startup with a warning, not treated as a fatal error: the provider may still be reachable through environment credentials (any-llm reads keys like `OPENAI_API_KEY`), so a pricing/provider mismatch should not abort the gateway. To have such an entry take effect, add the provider to the `providers` section.

### Default pricing

Default pricing is **off by default**. When you enable it (`default_pricing: true` in `config.yml`, or
`OTARI_DEFAULT_PRICING=true`) and a model has no price in the database (neither config nor `/v1/pricing`),
Otari falls back to community-maintained default pricing from the
[genai-prices](https://github.com/pydantic/genai-prices) dataset, which bundles per-million rates for
hundreds of models across the major providers. With it on, common models (for example `openai:gpt-4o`,
`anthropic:claude-sonnet-4-6`) are priced without any configuration, so `require_pricing` does not reject
them. The defaults are bundled with the installed package; no network access is used.

It is opt-in because a billing gateway should generally charge on rates you control: community estimates can
lag or differ from real provider rates, and turning this on changes what `require_pricing: true` guarantees
(unpriced-but-known models are auto-priced rather than rejected).

Resolution order is always database first, defaults last, so any price you set in config or via
`/v1/pricing` overrides the community default. Defaults are used only as a lookup fallback; they are never
written to the database.

Limitations when enabled:

- **Tiered pricing** is flattened to the base rate, so a request that crosses a context tier is billed at
  the base (a small under or over charge for very large requests).
- A **provider-agnostic match** is attempted when the exact provider is not in the dataset; an ambiguous
  model *name* could resolve to a different provider's rate. Prefer configuring such models explicitly.
- **HuggingFace** is modeled per inference backend, so a model is priced only when you pin a backend with
  the `huggingface:<model>:<backend>` selector (see the model reference in `models.md`). Auto routing and
  the policy suffixes (`:cheapest`, `:fastest`, ...) cannot be priced from the id alone and fall through to
  `require_pricing`.

> **Fail-closed by default.** With `require_pricing: true` (the default), a request for a model
> that has no pricing entry is rejected with HTTP 402 rather than served free and unmetered — an
> unpriced model would otherwise bypass the budget cap. To run genuinely free or self-hosted
> models, add an explicit `$0` pricing entry, or set `require_pricing: false`. Audio and moderation
> endpoints are exempt. A startup warning is logged if `require_pricing` is on with no pricing configured.
