# Configuration

Otari reads a YAML file, environment variables, and selected settings stored by
the management API. Start with [`config.example.yml`](../config.example.yml);
the running dashboard's Settings page shows the effective non-secret scalar
configuration and which values can be changed without a restart.

## Config file

Pass a file explicitly:

```bash
otari serve --config config.yml
```

A small standalone configuration looks like this:

```yaml
database_url: "postgresql://otari:otari@postgres:5432/otari"
master_key: ${OTARI_MASTER_KEY}
default_pricing: true

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

String values support `${ENV_VAR}` interpolation. Keep credentials in the
environment or a secret store rather than committing them to YAML.

## Environment variables

Every scalar `GatewayConfig` field can be overridden as
`OTARI_<UPPERCASE_FIELD>`, for example:

```bash
export OTARI_DATABASE_URL="postgresql://otari:otari@localhost:5432/otari"
export OTARI_MASTER_KEY="..."
export OTARI_DEFAULT_PRICING=true
```

Provider SDKs also read their native credential variables, including
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`, and
`GEMINI_API_KEY`.

Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, `on`, and
`off`, without regard to case.

### Full config via environment

Container platforms can supply the whole YAML document through
`OTARI_CONFIG_YAML`, or its base64-encoded form through `OTARI_CONFIG_B64`.
Use only one; raw YAML wins when both are present.

```bash
export OTARI_CONFIG_YAML='
default_pricing: true
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
'
```

Precedence from lowest to highest is the config file, structured environment
config, then scalar `OTARI_<FIELD>` values. Stored runtime settings override
the corresponding startup value after the database is available.

## Common settings

| Setting | Purpose |
| --- | --- |
| `database_url` | SQLite or PostgreSQL connection. PostgreSQL is recommended for production. |
| `master_key` | Deployment-wide management credential. |
| `host`, `port` | Server bind address. |
| `auto_migrate` | Apply Alembic migrations at startup. |
| `require_pricing` | Reject unpriced, budgeted traffic. Defaults to `true`. |
| `default_pricing` | Use the bundled genai-prices catalog when no stored price exists. |
| `rate_limit_rpm` | Per-user request limit. Unset disables it. |
| `enable_metrics` | Serve Prometheus metrics at `/metrics`. |
| `enable_docs` | Serve OpenAPI, Swagger UI, and ReDoc. |
| `mode` | `standalone`, `hosted`, or `hybrid`. See [Modes](modes.md). |

For every field, its current default, validation, and description live on
`GatewayConfig` in `src/gateway/core/config.py`. Operators can read the
non-secret effective set through `GET /v1/settings`.

## Provider configuration

The `providers` map is keyed by provider instance. A standard provider needs
only its credential:

```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

A custom or self-hosted endpoint can use a named instance:

```yaml
providers:
  home_lab:
    provider_type: openai
    api_base: "https://models.example.com/v1"
    api_key: ${HOME_LAB_TOKEN}
    models: [qwen3-32b]
```

Call it as `home_lab:qwen3-32b`. The optional `models` list supplies discovery
for backends without a model-listing endpoint. See [Models](models.md).

### Runtime provider management

Standalone operators can store provider credentials through the Providers page
or `/v1/provider-credentials`. Stored entries override config-file entries with
the same instance name. Config-file entries remain read-only in the dashboard.

Stored credentials require `OTARI_SECRET_KEY`, a Fernet key generated with
`otari gen-secret-key`. To rotate it, configure the new key before the old key,
run the provider and search-tool re-encryption endpoints, then remove the old
key. Losing every configured encryption key makes stored credentials
unrecoverable.

## Pricing

Pricing keys use `provider:model` or `instance:model`:

```yaml
pricing:
  openai:gpt-5:
    input_price_per_million: 1.25
    output_price_per_million: 10.00
```

Config-file prices seed the database. Prices stored through `/v1/pricing` take
precedence. Rates and settled costs use decimal arithmetic and costs are rounded
once to a micro-dollar. Use PostgreSQL for durable accounting.

### Default pricing

`default_pricing: true` enables a bundled genai-prices snapshot when no stored
price exists. Explicit database or config pricing always wins. The dashboard can
review and accept newer snapshots.

Default pricing is off because provider catalogs and reseller rates change.
With `require_pricing: true`, a budgeted request with no effective price is
rejected instead of bypassing the budget.

### Cache and tiered pricing

Optional cache fields reprice cached input when a provider reports it:

- `cache_read_price_per_million`
- `cache_write_price_per_million`
- `cache_write_1h_price_per_million`

Use `pricing_tiers` for a rate that applies to an entire request after an input
token threshold. The OpenAPI pricing schemas and dashboard editor show the
accepted shape.

### Per-request pricing (audio and moderations)

Audio, moderations, and direct search do not use token pricing. They reuse
`input_price_per_million` as USD per million requests. An unpriced request on
these endpoints is served at zero cost.

### Per-image pricing (image generation)

Image generation uses `input_price_per_million` as raw USD per image, without
million-unit scaling. Image generation is subject to `require_pricing` and
reserves the requested image count before dispatch.

These overloaded units are retained for schema compatibility. Do not apply a
token price to a request-priced or image-priced endpoint.

## Search tools

`search_tools` configures direct `POST /v1/search` calls. The same entries can
be managed at runtime from Tools or `/v1/search-tools`.

```yaml
search_tools:
  local:
    provider: searxng
    api_base: "http://searxng:8080"
```

`GET /v1/search-tools/providers` publishes the supported providers and whether
each requires an `api_key` or `api_base`. Provider options and request filters
are covered in [Built-in tools](tools.md). A tool carrying an `api_key` must use
an HTTPS `api_base`; a keyless local SearXNG endpoint may use HTTP.

## Mail

Mail is optional. Invitations still return an accept link when no transport is
configured.

SMTP needs the deployment's public URL, a host, and a sender:

```yaml
public_base_url: "https://otari.example.com"
mail_transport: smtp
smtp_host: "smtp.example.com"
smtp_port: 587
smtp_tls: true
mail_from_email: "otari@example.com"
smtp_user: ${SMTP_USER}
smtp_password: ${SMTP_PASSWORD}
```

`mail_transport: console` writes complete messages to logs for local testing.
Those messages can contain invitation or password-reset tokens, so never use it
where logs are shared. Test delivery from Settings or
`POST /v1/settings/mail/test`.

## Built-in tools and guardrails variables

The Tools pages and `GET /v1/tool-settings` show effective sandbox, web-search,
and guardrail configuration. Common startup settings are:

- `sandbox_url`
- `web_search_url`
- `guardrails_url`
- `mcp_allow_loopback` and `mcp_allow_private_hosts`
- `web_search_allow_private_hosts`
- `provider_allow_private_hosts`

See [Built-in tools](tools.md), [MCP](mcp.md), and
[Guardrails](guardrails.md) for behavior and security boundaries.

## Documentation links

By default the dashboard's Documentation link opens the bundled guide at
`/#/docs`. Set `docs_url` or `OTARI_DOCS_URL` to point it at an absolute
HTTP or HTTPS URL. The bundled guide remains available.

## Legal pages

The account menu carries a Terms of service row and a Data & Privacy row for
whichever of them this deployment has published. Set `terms_url` or
`OTARI_TERMS_URL`, and `privacy_url` or `OTARI_PRIVACY_URL`, to absolute HTTP or
HTTPS URLs:

```yaml
terms_url: "https://example.com/terms"
privacy_url: "https://example.com/privacy"
```

Each is independent. Unset, the Terms of service row is absent and the Data &
Privacy row stays disabled. A deployment whose dashboard sits beside a site that
owns the documents points at that site. `GET /v1/bootstrap` publishes both
addresses unauthenticated, so a credential in either is refused at startup, the
way `data_plane_url` refuses one. The same check covers `docs_url`.

## The data-plane address

A hosted control plane does not serve inference. Set `data_plane_url` or
`OTARI_DATA_PLANE_URL` to the gateway origin used by client snippets:

```yaml
data_plane_url: "https://gateway.example.com"
```

Supply the origin or path prefix without a trailing slash or `/v1`. Credentials,
query strings, and fragments are refused because `GET /v1/bootstrap` publishes
this value without authentication.

## otari.ai variables

Hybrid mode requires `OTARI_AI_TOKEN`. Optional platform settings control the
platform API URL, management URL, resolution timeout, usage-report timeout and
retries, and first-chunk fallback timeout. See [Modes](modes.md) and the
normative [hybrid-mode protocol](hybrid-mode-protocol.md).

## Extending Otari with a bootstrap module

`bootstrap` or `OTARI_BOOTSTRAP` names a trusted `module:callable` loaded
inside the gateway process. The callable can rebind extension ports and
contribute capability-gated routers. Most deployments should leave it unset.

This is executable code, not a feature flag. Install the module in the gateway
environment, pin it to a compatible Otari release, and authenticate every
contributed route. See [Architecture](../ARCHITECTURE.md) for the extension
boundary.
