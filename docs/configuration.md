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
| `public_base_url` | string | none | This deployment's own externally-reachable URL, no trailing slash. Needed to put an absolute link in an invitation email; see [Mail](#mail). |
| `smtp_host` | string | none | SMTP server host for outgoing mail. Unset disables mail entirely. |
| `smtp_port` | int | `587` | SMTP server port |
| `smtp_user` | string | none | SMTP username, if the server requires auth |
| `smtp_password` | string | none | SMTP password, if the server requires auth |
| `smtp_tls` | bool | `true` | Use STARTTLS when connecting to the SMTP server |
| `mail_from_email` | string | none | The `From` address on outgoing mail. Required, alongside `smtp_host`, for mail to actually send. |
| `mail_from_name` | string | `Otari` | The `From` display name on outgoing mail |
| `invitation_expiry_hours` | int | `168` | How long an organization invitation stays acceptable (default 7 days) |
| `rate_limit_rpm` | int | none | Max requests per minute per user (none = disabled) |
| `cors_allow_origins` | list | `[]` | Allowed CORS origins (empty = disabled) |
| `providers` | dict | `{}` | Provider credentials (see below) |
| `routing` | dict | `{}` | Named routing policies: which real model serves a request, what is tried after a retryable failure, and which guardrails always run. An alias is the one-target case. See [Routing policies](routing.md). Standalone mode only. |
| `router_alpha` | float | `0.3` | Learned router's cost-vs-quality dial: `score = predicted_quality - alpha * normalized_cost`. 0 ignores cost; higher prefers cheaper candidates. |
| `router_k` | int | `5` | Neighbors per routing decision. A pool with fewer comparable examples serves the policy's default target. |
| `router_seed_count` | int | `20` | Examples a pool needs before the router routes at all. |
| `router_confidence_floor` | float | `0.0` | Share of the `k` neighbors that must agree on the winner. Below it, the default target leads and the ranking becomes the failover chain. |
| `router_granularity` | string | `trace_sticky` | `trace_sticky` decides once per conversation and reuses it; `step` re-decides on every call. |
| `router_embedding_model` | string | `openai:text-embedding-3-small` | Model used to embed the task signal. Changing it invalidates stored examples rather than mixing vector spaces. |
| `router_max_records_per_user` | int | `5000` | Cap on stored routing-memory records per user; oldest are evicted past it, and it also bounds how many one decision loads. `0` disables eviction rather than storing nothing, so the store grows without limit while each decision stays bounded. |
| `aliases` | dict | `{}` | Model name aliases (display name to target selector `instance:model` or `provider:model`). The alias is what users see in `GET /v1/models` and in response `model` fields; pricing, budgets, and usage key on the resolved target. See [Model aliases](models.md#model-aliases) and, for more than one target, [Routing policies](routing.md). Standalone mode only. |
| `pricing` | dict | `{}` | Model pricing entries |
| `search_tools` | dict | `{}` | Search tools served by `POST /v1/search` (see below). Standalone mode only. |
| `enable_metrics` | bool | `false` | Enable Prometheus `/metrics` endpoint |
| `enable_docs` | bool | `true` | Enable `/docs`, `/redoc`, `/openapi.json` |
| `bootstrap_api_key` | bool | `true` | Create a first-use API key on startup when none exist |
| `log_writer_strategy` | string | `"single"` | Usage log writing: `"single"` (inline) or `"batch"` (background). Prefer `"batch"` for streaming clients: with `"single"`, a client that disconnects at the SSE `[DONE]` marker can leave the usage row uncommitted and the budget reservation unreconciled. `"batch"` queues in memory and flushes on a 1s interval or 100-row batch, so it removes that race but is not crash-durable. |
| `budget_strategy` | string | `"for_update"` | Budget validation: `"for_update"`, `"cas"`, or `"disabled"` |
| `require_pricing` | bool | `true` | Reject requests for models with no configured pricing (HTTP 402, fail-closed). When `false`, unpriced models are served and logged without cost. Audio and moderation endpoints are always exempt. |
| `default_pricing` | bool | `false` | When a model has no pricing in the database, fall back to community-maintained defaults from the bundled genai-prices dataset. Off by default (opt-in). Database pricing always wins. See [Default pricing](#default-pricing). |
| `reject_user_mismatch` | bool | `true` | When `true`, a non-master key whose request names a `user` other than its own is rejected (HTTP 403). When `false`, the client `user` is still forwarded to the provider but spend is always bound to the key's own user. This is the deployment-wide default; an individual key overrides it in either direction with its own `reject_user_mismatch` (`null` inherits). The master key may always bill an arbitrary user. |
| `capture_agent_telemetry` | bool | `true` | When `true`, content-free coding-agent telemetry is stored as `agent_telemetry` rows: behavioral log events (`tool_result`, `tool_decision`, `user_prompt`, `api_error`) received at `POST /v1/logs`, and outcome-metric data points (lines of code, commits, pull requests, active time) received at `POST /v1/metrics`. When `false`, both are discarded before storage. Usage capture and billing are unaffected either way. This is the deployment-wide default; an individual key overrides it in either direction with its own `capture_agent_telemetry` (`null` inherits). See [Use with Claude Code](use-with-claude-code.md). |
| `stream_missing_usage_policy` | string | `"estimate"` | How to bill a streamed response that completes with no provider usage data: `"estimate"` (charge the up-front estimate), `"fail"` (charge estimate and mark errored), or `"allow_free"` (don't bill). |
| `budget_estimate_default_output_tokens` | int | `1024` | Output-token count assumed when reserving budget for a request with no declared max output; reconciled to actual usage on completion. |
| `streaming_keepalive_interval_ms` | int | `15000` | Idle interval after which a streaming response emits a transport keepalive while it waits on the provider: a `ping` event on `/v1/messages`, an SSE comment line (`: keepalive`) on `/v1/chat/completions` and `/v1/responses`. Keeps an intermediary with a read timeout (Cloudflare's default Proxy Read Timeout is 125s) from severing a connection during a long time-to-first-token. Keepalives start once the provider stream is open and never touch usage accounting or extend a first-chunk/failover deadline. `0` disables. |
| `model_discovery` | bool | `true` | Auto-discover models for `GET /v1/models` from the configured providers: the `providers` block plus anything added at runtime on the Providers page. A provider that is callable through its credential environment variable alone is not discovered until it has an entry. Setting this to `false` also stops the background refresher, so the gateway makes no unattended `list_models` calls; the operator-facing `/v1/models/discoverable` and `/v1/providers/health` still dial when asked. |
| `model_cache_ttl_seconds` | int | `300` | TTL for the in-memory model-discovery cache, and the interval at which a background task re-dials every configured provider to refill it (floored at 30s; a round that saw a failure comes back on `model_discovery_negative_ttl_seconds` instead). While it is above `0`, `GET /v1/models`, `/v1/models/discoverable` and `/v1/providers/health` answer from that cache rather than dialing on the request path, and so does `GET /v1/models/{model_id}`, which never dials at all. The one read that still dials is the first one to ask about a provider that has never been dialed (a freshly started worker whose background refresh has not landed yet), so a cold worker reports what a provider actually serves instead of claiming it has no models. Setting this to `0` disables caching: reads then dial for themselves, and no background refresher runs. Force a live re-dial with `?refresh=true` on `/v1/models/discoverable` or `/v1/providers/health`; a health check within a few seconds of the last dial reuses it rather than starting another. |
| `model_discovery_timeout_seconds` | float | `10.0` | Per-provider timeout for a live model-discovery (`list_models`) call. Bounds how long an unreachable or slow provider can stall discovery before it is treated as failed. |
| `model_discovery_negative_ttl_seconds` | float | `30.0` | How long a failed model-discovery result is remembered before that provider is dialed again (`0` disables negative caching). This governs a read that dials: a cold provider, or any read while `model_cache_ttl_seconds` is `0`. Once the background refresher owns the dialing, how quickly a recovered provider reappears is bounded by `model_cache_ttl_seconds` (the refresh interval) rather than by this. |
| `models_dev_metadata` | bool | `true` | Enrich the dashboard's model detail with metadata (modalities, capabilities, knowledge cutoff) fetched from the public models.dev catalog. Set `false` to disable the outbound call; the gateway then falls back to the bundled genai-prices data. |
| `models_dev_cache_ttl_seconds` | int | `86400` | TTL for the cached models.dev catalog, and the interval at which a background task refetches it (floored at 5 minutes). While it is above `0`, `GET /v1/models/metadata` answers from that cache rather than waiting on the fetch. A failed fetch is held for one minute, not for the refresh interval, so a transient models.dev outage costs a minute of enrichment rather than a day. `0` disables caching, which means every read fetches instead. |
| `files_enabled` | bool | `true` | Enable the `/v1/files` upload/storage endpoints (standalone mode). |
| `files_backend` | string | `"local"` | Blob backend for uploaded file bytes (`"local"` filesystem for now). |
| `files_local_dir` | string | `"./otari-files"` | Directory the `local` files backend writes uploaded bytes to. |
| `files_max_bytes` | int | `536870912` | Maximum size in bytes for a single uploaded file (512 MiB default). |
| `files_retention_hours` | int | none | Stop serving files older than N hours (they return 404); none keeps files indefinitely. Stored bytes are not auto-reclaimed. |
| `file_understanding_enabled` | bool | `true` | Normalize file/image content blocks before the provider call (pass through for capable models, extract to text otherwise). When `false`, blocks are forwarded unchanged. |
| `vision_strategy` | string | `"describe"` | How image blocks are handled for text-only models: `"describe"` (vision side-call), `"ocr"` (extract text only), or `"off"` (drop with a log line). |
| `vision_describe_model` | string | none | `provider/model` used to caption images for text-only targets when `vision_strategy="describe"`. When unset, `describe` falls back to a logged drop. |
| `vision_describe_max_tokens` | int | `1024` | Cap on the describe model's output tokens per image; bounds cost and latency of the vision side-call. |
| `model_capabilities` | dict | `{}` | Per-model multimodal capability overrides (`provider/model` to `{supports_image, supports_pdf}`). Authoritative over any-llm's provider-level flags; needed for text-only local models behind OpenAI-compatible servers. |
| `sandbox_url` | string | none | Base URL of the code-execution sandbox backend for `otari_code_execution` tools. When unset, such requests are rejected with HTTP 400. Also settable via `OTARI_SANDBOX_URL`. |
| `guardrails_url` | string | none | Default input-guardrails service URL, used when a request does not pass its own guardrail `url`. Also settable via `OTARI_GUARDRAILS_URL`. |
| `web_search_url` | string | none | Base URL of the web-search backend (SearXNG instance or a search adapter) for `otari_web_search` tools. When unset, such requests are rejected with HTTP 400. docker-compose sets this to the bundled SearXNG container. Also settable via `OTARI_WEB_SEARCH_URL`. |
| `tools_header` | string | none | Override for the purpose-hint preamble header injected ahead of gateway-managed tool hints. When unset, a built-in default header is used. Also settable via `OTARI_TOOLS_HEADER`. |
| `sandbox_purpose_hint` | string | none | Default purpose hint forwarded to the sandbox backend when a tool entry supplies none. Also settable via `OTARI_SANDBOX_PURPOSE_HINT`. |
| `web_search_purpose_hint` | string | none | Default purpose hint for the web-search backend when a tool entry supplies none. Also settable via `OTARI_WEB_SEARCH_PURPOSE_HINT`. |
| `web_search_engines` | string | none | Comma-separated SearXNG engine list for the web-search backend. Also settable via `OTARI_WEB_SEARCH_ENGINES`. |
| `web_search_max_results` | int | none | Default cap on web-search hits (a per-tool `max_results` still overrides it). Also settable via `OTARI_WEB_SEARCH_MAX_RESULTS`. |
| `web_search_extract` | bool | none | Whether the web-search backend extracts page content in-process (`true`) or returns snippet-only results (`false`). When unset, extraction is on. Also settable via `OTARI_WEB_SEARCH_EXTRACT`. |
| `web_search_intercept` | bool | none | Whether a provider-named web-search declaration (bare `web_search`, `web_search_<date>`) is run against the gateway's backend instead of forwarded to the provider. Off when unset; `otari_web_search` is always run by the gateway. Requires `web_search_url`. See [Built-in tools](tools.md#web-search-interception). Also settable via `OTARI_WEB_SEARCH_INTERCEPT`. |
| `web_search_allow_private_hosts` | bool | `false` | SSRF gate: allow the web-search backend to fetch private/loopback/reserved hosts. Also settable via `OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS`. |
| `mcp_allow_loopback` | bool | `true` | SSRF gate: allow MCP server URLs that resolve to loopback (same-host sidecars). Also settable via `OTARI_MCP_ALLOW_LOOPBACK`. |
| `mcp_allow_private_hosts` | bool | `false` | SSRF gate: allow MCP server URLs that resolve to private/reserved hosts (and accept hostnames that fail to resolve at validation time). Also settable via `OTARI_MCP_ALLOW_PRIVATE_HOSTS`. |
| `provider_allow_private_hosts` | bool | `true` | SSRF gate: allow a provider `api_base` that resolves to private/loopback/reserved hosts. On by default (unlike the other gates), because `api_base` is master-key gated and the home-lab use case depends on private endpoints. Set to `false` to make provider connection tests, model discovery, and the credential write path (`POST /v1/provider-credentials` and `PATCH /v1/provider-credentials/{instance}`) refuse an internal `api_base`, so a blocked endpoint cannot be persisted. With the gate on, the write path enforces the same URL shape as the report path: the `api_base` must be an `http(s)` URL whose hostname resolves, so a non-`http(s)` scheme, a value with no hostname (e.g. `myproxy:8080`), or a host that cannot be resolved (including a temporarily-unreachable public one, a deliberate anti-DNS-rebinding stance) is refused with HTTP 400 before the private-range check. Chat dispatch (which dials the endpoint on every request) is not gated, so it is not a general egress control. Also settable via `OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS`. |
| `mode` | string | none | Operating mode (`"standalone"` or `"hybrid"`; the legacy value `"platform"` means hybrid): when unset, the mode is derived from the presence of `OTARI_AI_TOKEN`; when set explicitly it is enforced at startup, so `"hybrid"` without a token and `"standalone"` with a token both fail as conflicting configuration. |
| `platform` | dict | `{}` | otari.ai integration settings (`base_url`, timeouts, retries) |

## Environment variables

The following `OTARI_` variables override config file values for their matching fields. For example, `OTARI_PORT=9000` overrides `port: 8000` in the YAML.

`OTARI_` overrides apply to scalar fields (strings, numbers, booleans). List and dict fields (`cors_allow_origins`, `providers`, `pricing`, `aliases`, `routing`, and the `platform` block) are not read from individual `OTARI_` variables; set them in the YAML file, or supply the whole config through the environment with `OTARI_CONFIG_YAML` / `OTARI_CONFIG_B64` (see [Full config via environment](#full-config-via-environment)). The `platform` block also has dedicated `PLATFORM_*` variables (see the otari.ai variables below).

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
| `OTARI_MASTER_KEY` | Master key for management endpoints. When unset in standalone mode, one is generated on first run and printed to the logs (see [Runtime provider management](#runtime-provider-management)). |
| `OTARI_SECRET_KEY` | Fernet key that encrypts provider credentials and search-tool keys added through the dashboard. Required to store either in the UI. Generate one with `otari gen-secret-key`. |
| `OTARI_DATABASE_URL` | Database connection URL |
| `OTARI_HOST` | Server bind host |
| `OTARI_PORT` | Server bind port |
| `OTARI_AUTO_MIGRATE` | Auto-run migrations on startup |
| `OTARI_BOOTSTRAP_API_KEY` | Create first-use API key |

### Mail

Outgoing mail is what delivers an organization invitation's accept link (see [Access control](access-control.md#invitations)). It is optional: with none of this configured, invitations still work, just without an email sent, so an operator shares the accept link another way.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTARI_PUBLIC_BASE_URL` | none | This deployment's own externally-reachable URL, no trailing slash (e.g. `https://otari.example.com`). Needed to put an absolute link in an email; without it, an invitation is still created but not emailed even if SMTP is configured below. |
| `OTARI_SMTP_HOST` | none | SMTP server host. Unset disables mail entirely. |
| `OTARI_SMTP_PORT` | `587` | SMTP server port. |
| `OTARI_SMTP_USER` | none | SMTP username, if the server requires auth. |
| `OTARI_SMTP_PASSWORD` | none | SMTP password, if the server requires auth. |
| `OTARI_SMTP_TLS` | `true` | Use STARTTLS when connecting. |
| `OTARI_MAIL_FROM_EMAIL` | none | The `From` address on outgoing mail. Required, alongside `OTARI_SMTP_HOST`, for mail to actually send. |
| `OTARI_MAIL_FROM_NAME` | `Otari` | The `From` display name on outgoing mail. |
| `OTARI_INVITATION_EXPIRY_HOURS` | `168` | How long an invitation stays acceptable (default 7 days). |

### Built-in tools and guardrails variables

These operator-facing settings configure the gateway-managed tools (`otari_code_execution`, `otari_web_search`), the MCP tool loop, and input guardrails. Each is validated at startup and, where it maps to a scalar `GatewayConfig` field, can also be set in the config file (see the config reference above).

| Variable | Default | Description |
|----------|---------|-------------|
| `OTARI_SANDBOX_URL` | none | Base URL of the code-execution sandbox backend for `otari_code_execution` tools. Required to use that tool. |
| `OTARI_WEB_SEARCH_URL` | none | Base URL of the web-search backend for `otari_web_search` tools. Required to use that tool. |
| `OTARI_GUARDRAILS_URL` | none | Default input-guardrails service URL, used when a request does not pass its own guardrail `url`. |
| `OTARI_TOOLS_HEADER` | built-in | Override for the purpose-hint preamble header injected ahead of gateway-managed tool hints. |
| `OTARI_SANDBOX_PURPOSE_HINT` | none | Default purpose hint forwarded to the sandbox backend when a tool entry supplies none. |
| `OTARI_WEB_SEARCH_PURPOSE_HINT` | none | Default purpose hint for the web-search backend when a tool entry supplies none. |
| `OTARI_WEB_SEARCH_ENGINES` | backend default | Comma-separated SearXNG engine list (e.g. `google,bing`). |
| `OTARI_WEB_SEARCH_MAX_RESULTS` | backend default | Default cap on returned hits (a per-tool `max_results` still overrides it). Must be `>= 1`. |
| `OTARI_WEB_SEARCH_EXTRACT` | `true` | `0`/`false` disables in-process content extraction (snippet-only mode). |
| `OTARI_WEB_SEARCH_INTERCEPT` | `false` | Run provider-named web-search declarations (`web_search`, `web_search_<date>`) against the gateway's backend instead of forwarding them. Requires `OTARI_WEB_SEARCH_URL`. |
| `OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS` | `false` | SSRF gate: allow the web-search backend to fetch private/loopback/reserved hosts. |
| `OTARI_MCP_ALLOW_LOOPBACK` | `true` | SSRF gate: allow MCP server URLs that resolve to loopback (same-host sidecars). |
| `OTARI_MCP_ALLOW_PRIVATE_HOSTS` | `false` | SSRF gate: allow MCP server URLs that resolve to private/reserved hosts, and accept hostnames that fail to resolve at validation time. |
| `OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS` | `true` | SSRF gate: allow a provider `api_base` that resolves to private/loopback/reserved hosts. On by default (unlike the other gates). Set to `false` to make provider connection tests, model discovery, and the credential write path refuse an internal `api_base`; with the gate on, a non-`http(s)`, hostname-less, or currently-unresolvable `api_base` is also refused. Does not gate chat dispatch, so it is not a general egress control. |

### Provider credentials

Provider API keys can be set as environment variables instead of in the config file. These are picked up directly by the underlying SDK.

These credentials are used for standalone deployments. When connected to otari.ai, local provider credentials are not used.

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `MISTRAL_API_KEY` | Mistral |
| `GEMINI_API_KEY` | Google Gemini |

### Runtime provider management

Instead of setting providers at launch, you can add them from the admin dashboard
after startup (standalone mode only). This lets you launch Otari with almost
nothing set and finish configuration in the browser.

- **First-run master key.** If `OTARI_MASTER_KEY` is unset, Otari generates a
  master key on first startup, stores only its hash, and prints the plaintext
  once to the logs (look for `Your master key:`). Use it to sign in. An
  operator-set `OTARI_MASTER_KEY` always takes precedence and is never generated
  over.
- **Encryption at rest.** Provider keys added in the dashboard are stored
  encrypted with `OTARI_SECRET_KEY` (a Fernet key), as are the API keys of
  [search tools added at runtime](#adding-a-search-tool-at-runtime). Set it
  before adding a key; generate one with `otari gen-secret-key`. Keep it safe and
  separate from the database: losing it makes every stored key undecryptable, and
  a database dump alone cannot decrypt them. Rotate by prepending a new key
  (comma- or whitespace-separated); both old and new are tried on decrypt, then
  run `POST /v1/provider-credentials/reencrypt` and
  `POST /v1/search-tools/reencrypt` before retiring the old key.
- **Precedence.** Dashboard-stored providers merge over `config.yml` providers.
  A stored provider with the same instance name as a config one takes precedence
  (the shadowing is logged at startup). In the dashboard, config providers are
  marked `config` and are read-only; stored providers are marked `stored` and can
  be edited, tested, and deleted.
- **Scope.** Providers that authenticate with an API key (OpenAI, Anthropic,
  Mistral, Gemini, and OpenAI-compatible backends) are supported. Providers that
  use ADC/IAM (Vertex AI, Bedrock) remain config-file only.

### otari.ai variables

These are only relevant when running connected to [otari.ai](https://otari.ai). See [Modes](modes.md) for details.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTARI_AI_TOKEN` | none | Otari token from otari.ai (enables platform connection) |
| `PLATFORM_MANAGEMENT_URL` | `https://otari.ai` | Control plane a connected gateway points its operator at, published by `GET /v1/bootstrap`. Must be an absolute `http(s)` URL; the gateway refuses to start otherwise |
| `PLATFORM_RESOLVE_TIMEOUT_MS` | `5000` | Timeout for provider resolution calls |
| `PLATFORM_USAGE_TIMEOUT_MS` | `5000` | Timeout for usage reporting calls |
| `PLATFORM_USAGE_INLINE_TIMEOUT_MS` | `1500` | Budget for the successful usage report awaited to attach inline cost. Expiry returns the response without cost while the report continues |
| `PLATFORM_USAGE_MAX_RETRIES` | `3` | Max retries for transient usage reporting failures |
| `STREAMING_FALLBACK_FIRST_CHUNK_TIMEOUT_MS` | `2000` | Per-attempt timeout waiting for first streamed chunk |
| `STREAMING_FALLBACK_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS` | `0` | Extra first-chunk grace for the sole/final attempt, added on top of the per-attempt budget. `0` = no grace (unchanged) |

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

## Search tools

`search_tools` declares what [`POST /v1/search`](api-reference.md#search) can
run against, keyed by the name callers pass as `search_tool_name` (or in the
`/v1/search/{tool}` path). A tool can also be added at runtime from the
dashboard's Tools page (or `POST /v1/search-tools`), which needs no
config file at all; see [Adding a search tool at runtime](#adding-a-search-tool-at-runtime).
The two sources are merged, and a stored tool wins over a config-file entry of
the same name.

```yaml
search_tools:
  exa-search:
    provider: exa
    api_key: ${EXA_API_KEY}
    # api_base: "https://api.exa.ai"   # optional; override to route via a proxy
    # timeout: 30                      # optional; seconds
    options:                           # optional provider-native defaults
      type: fast
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | string | the tool name | Search provider to dispatch to. Supported: `exa`, `searxng`. |
| `api_key` | string | required for `exa` | Credential for the provider. A `searxng` backend is normally keyless; see below. |
| `api_base` | string | provider default | Override the provider's base URL. Required for `searxng` unless `web_search_url` is set. |
| `timeout` | number | `30` | Request timeout in seconds; must be greater than 0. |
| `options` | dict | `{}` | Provider-native request fields used as defaults. |

`options` is the escape hatch for knobs the LiteLLM-shaped request has no field
for, such as Exa's `type`, `category`, or `moderation`. Fields derived from the
request win over it, and `query` always comes from the caller.

Page content is requested by default, because the `snippet` on each result is
that text. Providers charge per page for it, so a deployment that only wants
ranked URLs (or highlights) can opt out by pinning `contents.text` to null:

```yaml
search_tools:
  exa-urls-only:
    provider: exa
    api_key: ${EXA_API_KEY}
    options:
      contents:
        text: null        # no page content, so no per-page content charge
        highlights: true  # keep the cheaper highlight sentences as the snippet
```

The opt-out is the operator's, so it holds even for a request that carries
`max_tokens_per_page`.

### Serving the web-search backend you already run

The `searxng` provider speaks the same
`GET {api_base}/search?format=json` contract as the
[`otari_web_search`](tools.md#web-search) tool's backend, so the bundled SearXNG
container and the Brave and Tavily fronting adapters are reachable from
`POST /v1/search` too, with no commercial key:

```yaml
web_search_url: "http://searxng:8080"

search_tools:
  local:
    provider: searxng   # api_base defaults to web_search_url
```

Set `api_base` to point a tool at a different backend than the in-loop tool
uses, and `api_key` only when the backend authenticates the gateway (it is sent
as `X-Gateway-Token`, the header the in-loop backend already sends); a
self-hosted SearXNG ignores it. `options` become extra `/search` query params,
so backend-native knobs (`language`, `time_range`) go there, and `q`, `format`
and `max_results` stay the gateway's. A tool that pins no `engines` option
inherits `web_search_engines`, so both surfaces query the same engines on the
same backend.

### Adding a search tool at runtime

Everything above is also settable without a config file. The dashboard's
Tools page has a Search tools card that adds, edits, and removes
tools, backed by `/v1/search-tools` (master-key gated, standalone mode only):

```bash
curl -X POST http://localhost:8000/v1/search-tools \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "local", "provider": "searxng", "api_base": "http://searxng:8080"}'
```

This works the same way runtime provider credentials do. The tool is a row in
`search_tool_credentials`, the API key is encrypted at rest with
`OTARI_SECRET_KEY` (so storing one requires that variable to be set) and is
never returned, and the change reaches every worker and replica within 30
seconds. `GET /v1/search-tools` lists the stored tools alongside the config-file
ones, which stay honored and are read-only there; a config-file tool is edited
where the file is defined, not through the API. Rotating `OTARI_SECRET_KEY`
needs `POST /v1/search-tools/reencrypt` alongside the provider-credential call
described in [Encryption at rest](#encryption-at-rest).

An entry saved here is held to the same rules startup validation applies to the
config file, so a tool that would refuse to boot cannot be stored.

Two request fields are honored by the gateway rather than upstream, because
SearXNG has no equivalent param: `search_domain_filter` and
`max_tokens_per_page` are applied to the returned hits. A plain SearXNG hit
carries the engine's short snippet, so the per-page cap usually does not bite; an
adapter that returns whole extracted pages (the bundled Tavily one does) is where
it does. `max_results` and `country` are forwarded, for a backend that reads them
(Tavily's adapter caps its page at 5 without the former), and a plain SearXNG
ignores an unknown param. SearXNG reports no cost either, so a search is billed at
the flat rate configured for the tool (below).

Every entry is validated at startup, so an unsupported provider or a missing
`api_key` fails before the first request. A `searxng` tool with no backend URL
anywhere is a startup warning instead of a failure, since the web-search URL can
also come from the dashboard's Tools page, which is read after the config loads;
until one is set, `POST /v1/search` refuses that tool with a 400 and the rest of
the gateway serves. Search tools are standalone mode only: in hybrid mode, search
is the platform's to configure.

To price search, add a flat per-request rate under the model key
`<provider>:<tool>` (for example `exa:exa-search`), following the same
convention as moderations: the stored `input_price_per_million` is USD per
million requests, so `5000.0` charges $0.005 per search. For what a search is
billed at, this is only a fallback; when the provider reports its own charge
(Exa returns `costDollars`), that is what gets billed.

Configuring the rate anyway is strongly recommended, because it is also what
the gateway reserves against the caller's budget *before* the search runs. With
no rate configured, that reservation is $0: a user already over their cap is
still refused, but one just under it can overshoot by a search's cost, and
concurrent searches cannot see each other's holds. Spend stays truthful either
way (the provider's charge is reconciled onto the usage row afterwards), so this
is about how tightly the cap holds, not about billing accuracy. Set the rate to
a typical search cost for the tool and the pre-flight hold does its job. Otari
logs a warning at startup for any configured search tool with no rate; an
explicit `0` counts as configured, for a tool you mean to be free.

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

#### How amounts are stored and rounded

Rates and settled costs are exact decimals, not binary floating point. A rate is
stored to 8 decimal places, far finer than any published price, so a rate you
type is the rate that prices your traffic. Costs are computed exactly and
rounded **once, half-up, to the micro-dollar** (0.000001 USD), which is the scale
of the `cost` column on a usage row. That is the only rounding in the cost path:
a charge below half a micro-dollar settles at zero, and one at or above it
settles at one micro-dollar.

Exactness at rest needs PostgreSQL. SQLite, the default for a single-node
deployment, has no decimal storage type and keeps the value as a float; the
gateway still computes and rounds identically on both, and a rounded amount
survives the round trip, but a deployment doing real accounting should run
PostgreSQL.

A pricing entry whose provider is not listed in the `providers` section is skipped at startup with a warning, not treated as a fatal error: the provider may still be reachable through environment credentials (any-llm reads keys like `OPENAI_API_KEY`), so a pricing/provider mismatch should not abort the gateway. To have such an entry take effect, add the provider to the `providers` section.

#### Cache token pricing

Providers that support prompt caching (OpenAI, Gemini, Anthropic) report cached-token counts that are captured in `cache_read_tokens` and `cache_write_tokens` on the usage log. Without a configured cache rate these tokens are still billed, at `input_price_per_million` (they are ordinary input tokens), just not at a discounted cache rate. To bill them at the provider's cache price instead, add the optional `cache_read_price_per_million` and `cache_write_price_per_million` rates (USD per million tokens):

```yaml
pricing:
  openai:gpt-4o:
    input_price_per_million: 2.50
    output_price_per_million: 10.00
    cache_read_price_per_million: 1.25  # discounted rate for cached input
  anthropic:claude-sonnet-4-6:
    input_price_per_million: 3.00
    output_price_per_million: 15.00
    cache_read_price_per_million: 0.30  # cache-read rate
    cache_write_price_per_million: 3.75  # cache-creation (write) rate
    cache_write_1h_price_per_million: 6.00  # Anthropic 1-hour cache creation
```

Every physical input token is charged once. The input/prompt count is treated as the grand total that *includes* cache reads and writes; the uncached remainder is billed at `input_price_per_million`, and cache tokens are re-priced at their own rate when one is configured. Providers report cache counts two ways, and the gateway normalizes both onto that single model:

- **OpenAI / Gemini**: `cache_read_tokens` is already a subset of `prompt_tokens`, so setting `cache_read_price_per_million` discounts the cached portion (re-priced at the cache rate instead of the full `input_price_per_million`) rather than double-counting it. `cache_write_tokens` is always 0 for these providers.
- **Anthropic**: `cache_read_tokens` and `cache_write_tokens` are reported separately from `prompt_tokens`, so they are added on top and billed at `cache_read_price_per_million` / `cache_write_price_per_million`. `cache_write_1h_tokens` records the 1-hour subset of cache creation and uses `cache_write_1h_price_per_million` when set, otherwise the standard cache-write rate. This holds on every turn, including warm-cache reads that create no new cache (`cache_write_tokens` is 0).

When a cache rate is left unset (null), those cache tokens are not dropped or billed at $0: they remain part of the input total and are billed at `input_price_per_million`, since they are still real input tokens. Set the cache rate to bill them at the provider's discounted cache price. The same fields are available on the `/v1/pricing` API (`SetPricingRequest` and `PricingResponse`).

#### Long-context pricing tiers

Some models charge a different rate once a request reaches a context threshold. Set `pricing_tiers` when that rate applies to the whole request, as it does for the tiered catalog entries Otari imports from `genai-prices`:

```yaml
pricing:
  openai:gpt-5:
    input_price_per_million: 1.25
    output_price_per_million: 10.00
    pricing_tiers:
      - min_input_tokens: 272000
        input_price_per_million: 2.50
        output_price_per_million: 15.00
```

At `min_input_tokens` and above, an entry replaces only the rates it lists for the entire request. Omitted rates inherit the base price. The dashboard's model detail editor exposes the same controls, and each usage row records the resulting billable meters and rate breakdown for auditability.

#### Per-request pricing (audio and moderations)

Audio (`/v1/audio/transcriptions`, `/v1/audio/speech`) and moderations report no token usage, so they bill a flat rate per request instead. `ModelPricing` only has per-million-token columns, so these routes reuse `input_price_per_million` with a different unit: **USD per million requests**. Divide by a million to read it as a per-request charge, so `6000.0` charges $0.006 per request.

```yaml
pricing:
  openai:whisper-1:
    input_price_per_million: 6000.0   # $0.006 per transcription
  openai:tts-1:
    input_price_per_million: 15000.0  # $0.015 per speech request
  openai:omni-moderation-latest:
    input_price_per_million: 0.0      # free (same effect as leaving it unset)
```

The key is the resolved `provider:model`, the same key every other route uses. [Search tools](#search-tools) follow the same convention under `<provider>:<tool>`.

Two behaviors are worth knowing before you rely on this:

- **Leaving the rate unset is free by design.** These endpoints are exempt from `require_pricing`, so an unpriced model is served, the usage row records a $0 cost, and no charge line is written. That is deliberate: a $0 charge line would render in Activity as a billed meter explaining a charge that never happened. Set a nonzero rate to get the charge line; a rate of `0.0` is treated the same as no rate at all.
- **[Default pricing](#default-pricing) never applies here.** The genai-prices dataset quotes rates in USD per million *tokens*, and some audio models are in it (`openai:gpt-4o-transcribe`, `openai:gpt-4o-mini-tts`). Honoring one would charge a token rate as a request rate, so per-request routes resolve their rate from what you configured and nothing else, whether or not `default_pricing` is on. Search works the same way.

One dashboard consequence of recording a $0 cost rather than no cost: these rows read as priced, so they do not appear under Activity's `Priced?` filter or in the Usage page's "unpriced requests" count. That count is for rows whose cost is unknown; an audio row with no rate configured is knowingly free, not unknown.

Audio differs from moderations and search in one respect: it reserves $0 against the budget before the call rather than reserving the configured rate. The per-user gate still applies (an unknown, blocked, or already-over-budget user is refused), but the cost lands on the budget at reconciliation instead of being held up front, so concurrent audio requests cannot see each other's holds. That makes the cap soft for audio: requests already in flight can settle past `max_budget`, by at most the configured rate times the number of concurrent audio requests. Spend stays truthful either way, and the next request is refused. A duration-based meter that would give audio a real pre-call estimate is tracked in [#376](https://github.com/mozilla-ai/otari/issues/376).

#### Per-image pricing (image generation)

Image generation (`/v1/images/generations`) bills per generated image, and it overloads `input_price_per_million` with a *third* unit: **raw USD per image**, unscaled. Unlike the per-request rate above, there is no division by a million.

```yaml
pricing:
  openai:dall-e-3:
    input_price_per_million: 0.04   # $0.04 per image
  openai:gpt-image-1:
    input_price_per_million: 0.19   # $0.19 per image
```

The cost is the rate times the number of images returned (falling back to the requested `n` when a provider omits the count), and each usage row records an `images` meter with the matching charge line.

Unlike audio and moderations, image generation is **not** exempt from `require_pricing`: an unpriced image model is rejected with HTTP 402 under the default `require_pricing: true`. It also reserves its estimate before the call, so the rate is held against the budget for the requested image count and reconciled to the delivered count. [Default pricing](#default-pricing) is not consulted here either, for the same unit reason: the dataset carries `openai:gpt-image-1` at 5.0 USD per million tokens, which read as a per-image rate would bill $5.00 for one image.

### Default pricing

Default pricing is **off by default**. When you enable it (`default_pricing: true` in `config.yml`, or
`OTARI_DEFAULT_PRICING=true`) and a model has no price in the database (neither config nor `/v1/pricing`),
Otari falls back to community-maintained default pricing from the
[genai-prices](https://github.com/pydantic/genai-prices) dataset, which bundles per-million rates for
hundreds of models across the major providers. With it on, common models (for example `openai:gpt-4o`,
`anthropic:claude-sonnet-4-6`) are priced without any configuration, so `require_pricing` does not reject
them. The defaults are bundled with the installed package; no network access is used by default. A master-key
operator can use **Check for price updates** in dashboard **Settings** to fetch the latest upstream snapshot, review
the added, changed, and removed rates, and explicitly accept or reject it. Pending reviews and accepted snapshots are
stored in the database, so the decision survives a standalone restart. The accepted snapshot uses source
`genai-prices`. Stored custom prices always take precedence and are not changed by a refresh.

It is opt-in because a billing gateway should generally charge on rates you control: community estimates can
lag or differ from real provider rates, and turning this on changes what `require_pricing: true` guarantees
(unpriced-but-known models are auto-priced rather than rejected).

Resolution order is always database first, defaults last, so any price you set in config or via
`/v1/pricing` overrides the community default. Defaults are used only as a lookup fallback; they are never
written to the database.

Limitations when enabled:

- **Tiered pricing** is retained from the dataset and applied when a request crosses a configured context threshold.
- Lookups key on the **provider instance name** first, then on the `provider_type` backing it, so an
  instance you named yourself (`aws-prod` over `provider_type: bedrock`) is priced at its real provider's
  rates rather than falling through. A `provider_type` of `openai-compatible` or `anthropic-compatible` is
  excluded here: it names the protocol an endpoint speaks, not who serves the request, so a self-hosted
  backend is not billed at OpenAI's or Anthropic's rates for reusing their model names. Set an explicit
  price for those.
- A **provider-agnostic match** is attempted when the exact provider is not in the dataset; an ambiguous
  model *name* could resolve to a different provider's rate. Prefer configuring such models explicitly.
- A **vendor-prefixed model id** (`anthropic.claude-sonnet-5`, `us.anthropic.claude-sonnet-5-v1:0`,
  `openai.gpt-oss-120b`) is a last resort, tried only once every lookup above it has missed, and is priced
  under the vendor named in the prefix. That is the vendor's own list price, not the reseller's, and the gap
  is not always small (Bedrock lists Sonnet about 10% above Anthropic, but a reseller's rate for an
  open-weights model can differ by half again), so configure it explicitly where the difference matters.
- **HuggingFace** is modeled per inference backend, so a model is priced only when you pin a backend with
  the `huggingface:<model>:<backend>` selector (see the model reference in `models.md`). Auto routing and
  the policy suffixes (`:cheapest`, `:fastest`, ...) cannot be priced from the id alone and fall through to
  `require_pricing`.
- **Routes that do not bill by the token are excluded.** Audio, moderations, and search bill a flat rate per
  request; image generation bills per image. Every dataset rate is per million tokens, so defaults are not
  consulted for any of them: honoring one would charge a token rate under a unit that is not a token. For
  audio, moderations, and search that means an unpriced model stays free (they are exempt from
  `require_pricing`); for images it means an image model priced only in the dataset now falls through to
  `require_pricing` and is rejected with 402 rather than billed at roughly a million times its real rate. See
  [per-request pricing](#per-request-pricing-audio-and-moderations) and
  [per-image pricing](#per-image-pricing-image-generation).

> **Fail-closed by default.** With `require_pricing: true` (the default), a request for a model
> that has no pricing entry is rejected with HTTP 402 rather than served free and unmetered; an
> unpriced model would otherwise bypass the budget cap. To run genuinely free or self-hosted
> models, add an explicit `$0` pricing entry, or set `require_pricing: false`. Audio and moderation
> endpoints are exempt. A startup warning is logged if `require_pricing` is on with no pricing configured.
