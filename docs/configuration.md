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

A small configuration looks like this:

```yaml
database_url: "postgresql://otari:otari@postgres:5432/otari"
master_key: ${OTARI_MASTER_KEY}
default_pricing: true
```

Providers are added as [hosted providers](hosted-providers.md), on the Hosted
providers page or under `hosted_providers:`.

String values support `${ENV_VAR}` interpolation; keep credentials in the
environment or a secret store, not in YAML. A reference to an unset variable
fails the load. A set but empty variable leaves a `providers` entry with no
credential: Otari ignores the entry with a warning naming the variable, so a
credential stored for the same provider through the dashboard serves instead.

## Environment variables

Every scalar `GatewayConfig` field can be overridden as
`OTARI_<UPPERCASE_FIELD>`, for example:

```bash
export OTARI_DATABASE_URL="postgresql://otari:otari@localhost:5432/otari"
export OTARI_MASTER_KEY="..."
export OTARI_DEFAULT_PRICING=true
```

Provider credentials come from the environment through `${VAR}` references
in the `providers` map (see [Provider configuration](#provider-configuration)).
A provider's native variable, such as `OPENAI_API_KEY`, is never read on its
own: name it in the entry (`api_key: ${OPENAI_API_KEY}`); see
[Models](models.md#configuring-a-provider).

Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, `on`, and
`off`, without regard to case.

### Full config via environment

Container platforms can supply the whole YAML document through
`OTARI_CONFIG_YAML`, or its base64-encoded form through `OTARI_CONFIG_B64`.
Use only one; raw YAML wins when both are present.

```bash
export OTARI_CONFIG_YAML='
master_key: ${OTARI_MASTER_KEY}
default_pricing: true
'
```

Precedence from lowest to highest is the config file, structured environment
config, then `OTARI_<FIELD>` values: the scalar fields, and
`OTARI_HOSTED_PROVIDERS` and `OTARI_PROVIDER_SUPPORTED_COUNTRIES` as JSON. A
top-level key of the structured config, and those two variables, replace the
lower layer's value whole rather than merging into it. Stored runtime settings override
the corresponding startup value after the database is available.

## Common settings

| Setting | Purpose |
| --- | --- |
| `database_url` | SQLite or PostgreSQL connection. PostgreSQL is recommended for production. Unset, it is `sqlite:///./otari.db`, and in the container image `sqlite:////data/otari.db`, on the image's `/data` volume. |
| `master_key` | Deployment-wide management credential. |
| `host`, `port` | Server bind address. |
| `auto_migrate` | Apply Alembic migrations at startup. |
| `require_pricing` | Reject unpriced, budgeted traffic. Defaults to `true`. |
| `default_pricing` | Use the models.dev price snapshot when no stored price exists. |
| `models_dev_base_url` | Where the models.dev catalog is fetched from. Defaults to [Mozilla.ai's fork of models.dev](https://github.com/mozilla-ai/models.dev), which serves models.dev's catalog plus the models it does not list yet; set `https://models.dev` to read models.dev itself. |
| `pricing_refresh` | Whether the catalog updates itself on a schedule, and whether its price changes wait for review: `manual`, `review`, or `auto`. |
| `feedback_enabled` | Offer **Contact**, which sends a message to the Otari team. Defaults to `false`; startup setting. See [Contacting the Otari team](#contacting-the-otari-team). |
| `platform_url` | The otari.ai address **Connect otari.ai** asks for a key. Defaults to `https://api.otari.ai`; an empty value removes the option. Startup setting. See [Connecting otari.ai](hosted-providers.md#connecting-otariai). |
| `update_check` | Check otari.ai daily for new releases and security fixes, sending the version and a random install id. Unset (the default) asks an operator in the dashboard; `true` or `false` answers for them. See [Update check](update-check.md). |
| `health_checks_enabled` | Offer the Health page and its scheduled checks of the hosted providers' models. Defaults to `true`; startup setting. See [Health checks](health-checks.md). |
| `public_catalog` | Serve the model catalog to visitors without a session. Defaults to `false`. |
| `public_catalog_rate_limit_per_minute` | Anonymous catalog reads per client address per minute. Defaults to 60. |
| `rate_limit_rpm` | Per-user request limit. Unset disables it. |
| `rate_limit_store` | Where `rate_limit_rpm` and the `rate_limits` rules are counted: `memory` (the default) or `redis`. See [Rate limits across replicas](#rate-limits-across-replicas). |
| `rate_limit_redis_url` | The Redis that the `redis` store counts in. |
| `rate_limits` | Requests per minute, tokens per minute and requests in flight, per deployment, API key, user or model. Also managed from the dashboard. See [Rate limit rules](#rate-limit-rules). |
| `budgets` | Budgets declared by id and written to the database at every start. See [Budgets and API keys in config](#budgets-and-api-keys-in-config). |
| `api_keys` | API keys, service keys among them, declared with a secret you choose and written to the database at every start. See [Budgets and API keys in config](#budgets-and-api-keys-in-config). |
| `federation_issuers` | OpenID Connect issuers whose identity tokens may authenticate as an API key in place of its secret. Empty, the default, turns it off. Setting it requires `OTARI_SECRET_KEY` and `federation_audiences`, or the gateway refuses to start. See [Workload identity](workload-identity.md). |
| `federation_audiences` | The audiences this deployment answers to, one of which an identity token must name in its `aud` claim. Required once `federation_issuers` or `federation_member_rules` is set. See [How a token is checked](workload-identity.md#how-a-token-is-checked). |
| `federation_member_rules` | Let an organization's owners and admins give their own API keys federation rules naming any public https issuer, matched only in the token exchange that names them. Defaults to `false`. Setting it requires `OTARI_SECRET_KEY` and `federation_audiences`. See [Rules on a member's own key](workload-identity.md#rules-on-a-members-own-key). |
| `federation_allow_private_hosts` | Let a federation issuer's discovery document and signing keys be fetched from a private address. Defaults to `false`. |
| `federation_access_token_ttl_sec` | How long an access token from the workload identity token exchange lasts, in seconds. Defaults to 600; between 60 and 3600. See [Claude Code and the Anthropic SDKs](workload-identity.md#claude-code-and-the-anthropic-sdks). |
| `automatic_routing` | Request-aware ordering of providers for unpinned catalog IDs. Enabled by default. See [Automatic routing](#automatic-routing). |
| `provider_max_retries` | How many times a provider SDK retries a failed call before Otari sees the failure. Unset keeps each SDK's default. See [Provider retries](#provider-retries). |
| `provider_supported_countries` | The countries each provider type serves, such as `openai: [US, DE]`. A request from another country skips that provider. Empty, the default, turns it off. See [Supported countries](#supported-countries). |
| `client_country_header` | A header a trusted proxy sets to the client's country code, such as `CF-IPCountry`. See [Supported countries](#supported-countries). |
| `client_country_database` | Path to a MaxMind DB country file the client address is looked up in. When set, a country file stored from the Settings page is not read. Needs the `geoip` extra. See [Supported countries](#supported-countries). |
| `closed_model_providers` | Providers whose models are closed, so a plan for open-weight models does not cover them, except a model in `closed_model_open_families`. A model on any other provider is open-weight unless its name is in `closed_model_families`. Defaults to Anthropic, Gemini, Vertex AI, xAI and OpenAI, with their Azure and Vertex variants. |
| `closed_model_families` | Model name prefixes that are closed on any provider, matched after the last `/` and any vendor prefix (OpenRouter's `anthropic/claude-sonnet-4` and Bedrock's `us.anthropic.claude-sonnet-4` are both `claude`). A prefix matches a whole word unless it ends in `-`: `o3` matches `o3-mini` but not `o3x`, and `gpt-` matches any name it starts. Defaults to `claude`, `gemini`, `grok`, `gpt-`, `chatgpt` and `o1` to `o9`. |
| `closed_model_open_families` | Model name prefixes that are open-weight even on a closed provider or in a closed family, matched the same way. Defaults to `gpt-oss` and `gemma`. |
| `streaming_keepalive_interval_ms` | Idle interval after which a streaming response sends a keepalive, so an intermediary with a read timeout does not cut a slow stream. Defaults to 15000; `0` disables keepalives. |
| `streaming_first_chunk_hold_ms` | How long the gateway holds a streaming response's status line for the provider's first chunk, so a provider failure before any output gets its own HTTP status, and falls over in a routing policy, rather than ending a `200` in an error event. A slower provider still streams. Defaults to 15000; `0` commits the response as soon as the stream opens. See [Error codes](api-reference.md#error-codes). |
| `idempotency_retention_sec` | How long a completion sent with an `Idempotency-Key` is kept for a retry to replay. Defaults to a day; `0` ignores the header. Needs `OTARI_SECRET_KEY`, which encrypts the stored response. See [Retrying safely](api-reference.md#retrying-safely). |
| `provider_request_timeout_seconds` | How long a non-streaming provider call may take, in seconds. Defaults to 600. Connecting stays capped at 5 seconds for providers whose SDK separates the two (Anthropic, OpenAI and OpenAI-compatible, Groq, Cerebras, Together); Bedrock, Gemini, Vertex AI, Mistral and Cohere apply the value as a whole. Without it, Anthropic's SDK refuses a non-streaming request with a large `max_tokens`. Providers with no per-request timeout, and a provider whose `client_args` sets its own `timeout`, keep their client's timeout. |
| `enable_metrics` | Serve Prometheus metrics at `/metrics`. Needs the `metrics` extra (`pip install gateway[metrics]`), which the Docker image installs; setting this without it refuses to start. |
| `trace_content_enabled` | Keep what an agent session's spans were about: prompts, commands, file paths, tool call arguments. Defaults to `false`. See [Keeping what a span was about](sessions.md#keep-what-a-session-was-about). |
| `trace_content_tool_output` | Also keep what each tool returned, on a request routed through Otari that keeps content. A workspace can still turn it off for itself. Defaults to `false`. |
| `trace_platform_break_glass_enabled` | Let the deployment's operator open any organization's session content, recorded with a reason and a reference. Defaults to `false`. See [Who reads content](sessions-reference.md#who-reads-content). |
| `trace_content_key_backend` | What seals that content: `secret_key` (the default, `OTARI_SECRET_KEY`), `aws_kms`, or `aws_kms_only`, which opens nothing `OTARI_SECRET_KEY` sealed. See [Sealing content with AWS KMS](trace-storage.md#sealing-content-with-aws-kms). |
| `accept_incoming_trace_context` | Join spans the gateway creates to the caller's trace. Defaults to `false`. See [Trace context propagation](#trace-context-propagation). |
| `enable_docs` | Serve OpenAPI, Swagger UI, and ReDoc. |

For every field, its current default, validation, and description live on
`GatewayConfig` in `src/gateway/core/config.py`, or in the per-domain module
under `src/gateway/core/settings/` that it inherits. Operators can read the
non-secret effective set through `GET /api/v1/settings`.

### Database connections

| Setting | Purpose |
| --- | --- |
| `db_pool_size`, `db_max_overflow` | Connections available for serving requests. |
| `db_pool_timeout` | Seconds a request waits for a free connection before it is refused. |
| `db_pool_recycle` | Retire a connection after this many seconds. `-1` disables. |
| `db_connect_timeout` | Seconds to wait for a new connection to be established. |
| `db_command_timeout` | Client-side ceiling on one statement. `0` disables. |
| `db_statement_timeout_ms` | Server-side ceiling on one statement, the backstop for the setting above. Must exceed it; `0` disables. |
| `db_log_pool_size` | Connections reserved for usage logging, separate from the pool above. |

The pool sizes bound concurrent database work, not provider calls: a request
returns its connection before the upstream call. Raise them for heavy
dashboard and management traffic, not for inference.

Set recycling and both statement timeouts behind a managed database or a NAT,
which drop idle connections without closing them. Without them, a silently
dropped connection makes a request hang for minutes (the pool's idle ping
blocks on the dead socket too).

`db_statement_timeout_ms` (server-side) still ends a statement when the client
is the stuck half. It must be above `db_command_timeout` (client-side); the
defaults are, and startup validation requires it.

### Rate limits across replicas

`rate_limit_rpm` is counted in this process by default, so a deployment
running N replicas (or N workers) admits up to N times the limit. To hold the
limit for the deployment as a whole, count it in Redis:

```yaml
rate_limit_rpm: 600
rate_limit_store: redis
rate_limit_redis_url: redis://redis:6379/0
```

This needs the `redis` extra (`pip install gateway[redis]`), which the Docker
image installs, and Redis 5 or later. Startup refuses `redis` without a URL or
without the extra.

Each request is checked and counted atomically against Redis's clock, so
replica clock skew does not matter. If Redis cannot be reached, each replica
counts on its own and retries Redis a few seconds later; the log says when that
starts and stops.

### Rate limit rules

`rate_limits` sets limits beyond `rate_limit_rpm`. Each rule names what one
count is shared by (`per`) and sets any of three limits:

```yaml
rate_limits:
  - name: keys          # each API key on its own
    per: key
    rpm: 600
    tpm: 200000
  - name: end-users     # each user, including a service key's end users
    per: user
    rpm: 30
  - name: everyone      # one count for the whole deployment
    per: deployment
    max_concurrent: 200
  - name: flash-cap     # each model listed, however a request reaches it
    per: model
    models: ["vertex:gemini-2.5-flash"]
    rpm: 100
  - name: feature       # every user on one budget together, each budget listed on its own
    per: budget
    budgets: ["<budget_id>", "<budget_id>"]
    rpm: 600
    mode: report        # served anyway, and named in Otari-Rate-Limit-Exceeded
```

- `rpm`: requests per minute.
- `tpm`: tokens per minute, counted on what each request used. A request is
  admitted while the minute's tokens are under the limit and charged what it
  used once it completes, so the last request admitted in a minute can take
  the count past the limit. A failed request is charged the tokens its
  provider reported (usually none); one refused after admission (by its
  budget, say) is charged nothing.
- `tpm_admission: estimate` counts the way providers count their own quotas:
  a request holds an estimate (its prompt plus `max_tokens`, or
  `budget_estimate_default_output_tokens` when it sets none) while it runs, and
  is refused when that does not fit. Use it to stay under a provider's quota,
  such as a `per: model` rule that should move traffic to the next model before
  the provider refuses it. A request whose estimate alone exceeds the limit is
  always refused, with no `Retry-After`.
- `max_concurrent`: requests in flight at once. A slot is given back when the
  response ends, streamed or not. `lease_sec` (15 minutes by default) bounds how
  long a slot outlives a process that dies holding it.

A request has to fit every rule that applies to it. One that does not is
refused with a 429 naming the rule and limit, is counted by none of them, and
holds no budget.

- Rules apply to chat completions, messages, responses and search, after
  `rate_limit_rpm`.
- They count in `rate_limit_store`, so with Redis they hold across replicas.
- `per: key` does not limit a request without an API key (the master key or a
  dashboard session), and `per: user` does not limit one billed to no user.

A build that bills organizations can also limit how many of an organization's
calls on the deployment's own provider credentials are in flight at once, which
no rule here sets. A chat completion, message or response over that limit waits
up to 10 seconds for a slot, then is refused with a 429 naming the rule
`organization_concurrency`. Each server process lets at most that many requests
wait at once; one more is refused straight away with the same 429. It counts in the same store.

`per: model` limits each model in `models`, written as `instance:model`. One
count is kept per model and shared by every policy, alias and direct call that
reaches it, so it can stand for a provider's quota.

- It is checked when an attempt is about to call the model, not at admission.
  A [routing policy](routing.md#spill-over-when-a-model-is-full-priority-routing)
  skips a full model and tries its next candidate; the request gets a 429 only
  when no candidate has room. A direct call to a full model is refused.
- A candidate that fails before responding gives back its tokens and slot but
  keeps its request counted.
- A search tool is limited the same way, named as `provider:tool`
  (`exa:exa-search`).

`per: budget` keeps one count for every user on the same budget, such as all of
a service key's end users on one feature's budget, so it limits a feature as a
whole.

- `budgets` narrows it to the listed `budget_id`s (as `GET /api/v1/budgets`
  returns them), each counted on its own; unset, it counts every budget.
- Not counted: a user on no budget, a key with `exclude_from_budget`, and any
  request while `budget_strategy` is `disabled`.
- Like `per: deployment`, it is counted after the rules that count the caller
  alone, so a request over its own limit takes no place in a shared count.

`mode: report` counts a rule but refuses nothing; use it to watch a limit
before enforcing it. `mode: enforce` is the default.

- A request over the rule is served, and routing picks the same candidate.
- The response carries `Otari-Rate-Limit-Exceeded` naming every report rule it
  was over (comma-separated), and the gateway logs it and counts it in
  `gateway_rate_limit_reported_total` by rule.
- A reported request is not counted in the window it was over, so the count
  matches what an enforced rule would keep. A request something else refuses
  is not reported.

Rules can also be managed from the dashboard (**Settings**, **Rate limit
rules**) or through `/api/v1/rate-limits`:

- A change applies at once on the replica that served it and within 30 seconds
  on the others; a changed rule keeps what it already counted.
- "Each model" adds a `per: model` rule. `per: budget`, `budgets` and `mode`
  are set through the API.
- Rules from `config.yml` are listed read-only, and a stored rule cannot take
  one's name. A stored rule whose name `config.yml` later declares is skipped
  with a warning at startup; `DELETE /api/v1/rate-limits/{name}` still removes
  it.

### Trace context propagation

With `accept_incoming_trace_context: true`, Otari extracts incoming context with
OpenTelemetry's configured propagators (`OTEL_PROPAGATORS`; W3C `traceparent`,
`tracestate` and `baggage` by default) and makes it current for the request, so
the spans the gateway creates join the caller's trace. The context is detached
when the response, including a streamed body, finishes. Missing or invalid
headers never reject a request; the gateway starts a new trace instead. Otari
does not inject propagation headers into provider or platform requests.

It is off by default because the headers are unauthenticated: the middleware
runs before route auth, so any caller can choose the trace ID and sampling flag
your collector ingests, and a malformed `tracestate` makes OpenTelemetry log a
warning per offending member (`opentelemetry.trace.span`). Enable it for trusted
service-to-service callers, ideally behind a proxy that strips these headers at
the edge. Browsers cannot send them cross-origin: they are not in the CORS
allow-list.

## Budgets and API keys in config

A deployment can declare its budgets and API keys in config.yml instead of
creating them through the management API. Each start creates the ones missing
from the database and writes back the ones whose entry changed, so a service that calls Otari needs only this file and one secret
that both sides read from the environment.

```yaml
budgets:
  team-monthly:
    max_budget: 500
    reset_alignment: calendar_month

api_keys:
  backend:
    secret: ${BACKEND_API_KEY}
    user_id: backend
    ceiling: team-monthly
```

A service key with end-user budgets is declared the same way; see
[Or declare it in config.yml](operator-service-keys.md#or-declare-it-in-configyml).

A budget is keyed by its id and takes the fields `PUT /api/v1/budgets/{budget_id}`
takes: `name`, `max_budget`, `token_limit`, `request_limit`, `rpm_limit`,
`tpm_limit`, and either `budget_duration_sec` or `reset_alignment`, not both.
An id is up to 128 letters, digits, `.`, `_` and `-`, and starts with a letter
or digit.

A key is keyed by a name of the same form, which is how Otari finds it again at
the next start. It takes:

| Field | Meaning |
| --- | --- |
| `secret` | Required. The secret requests present, interpolated from the environment. |
| `key_name` | Display name. Defaults to the key's name in config. |
| `user_id` | The user the key bills to, created when missing. Unset uses the shared `default` user. |
| `is_service_key` | Lets requests name end users. See [Track spend per end user](end-users.md). |
| `end_user_budget_ids` | Budgets a request may start a new end user on with `Otari-End-User-Budget`. |
| `max_end_users` | Most live end users the key may have on the budgets it assigns, or on any budget when it assigns none. See [Capping a key's end users](end-users.md#capping-a-keys-end-users). |
| `end_user_budget_id` | Budget a new end user starts on when the request names none. Must be on `end_user_budget_ids` when both are set. |
| `ceiling` | Budget that caps the key as a whole, across all of its end users: a scoped budget with `scope_type: api_token` on the key. |
| `exclude_from_budget` | Log the key's cost without enforcing any budget. |
| `reject_user_mismatch` | Per-key override of the deployment setting. |

Unknown fields are refused, so a misspelled limit stops the start rather than
leaving a budget uncapped. A key may name a budget declared here or one that
already exists in the database.

### The secret

The secret is yours to choose, so the same value works across restarts and
replicas, and rotating it is a matter of changing the variable on both sides.
It must look like a key Otari mints: `tk-` followed by at least 47 letters,
digits, `-` and `_`, random enough not to read as a placeholder. Generate one
with:

```bash
python -c "import secrets; print('tk-' + secrets.token_urlsafe(48))"
```

Otari refuses to start when a secret is empty, is a placeholder, does not have
that form, equals the master key, or is shared by two keys. Only its hash is
stored, and no log line, error or API response carries it: the keys API shows
its usual prefix and last four characters.

### What config.yml owns

Config is the source of truth for what it declares, and a start writes an
entry back whenever the entry changed:

- A missing budget or key is created. An existing one gets the declared values
  only when its entry differs from the one a start last wrote. So a change made
  through the API or dashboard (lowering a budget, deactivating a key in an
  emergency) lasts across restarts until `config.yml` changes that entry.
- Changing any field writes the whole entry back: a deactivated key becomes
  active again. Fields a key does not declare (expiry, allowed models,
  metadata) are left alone.
- Changing a key's secret in config, including through the environment
  variable it reads, changes its secret at the next start, and the old one
  stops working. Regenerating a declared key from the dashboard or the API
  lasts until config.yml changes the key's entry or its secret, which restores
  the secret config.yml declares.
- `end_user_budget_ids`, `end_user_budget_id` and `ceiling` are reconciled
  with the key's entry. Existing end users keep the budget they started on. A
  ceiling moved to another budget restarts its period and keeps its recorded
  spend; one already on the declared budget keeps its period. A key with no
  `ceiling` keeps any ceiling it has, and an unchanged entry leaves a ceiling
  changed through the API alone.
- Otari keeps a digest of each entry it wrote beside the budget or key, never
  the secret itself (the secret enters as its hash), to tell whether the entry
  changed.
- Nothing is deleted. A budget or key removed from config stays as it is and
  stops being marked as declared, so it can then be changed or deleted through
  the API. Declaring it again writes the entry back over it at the next start,
  even an entry unchanged from before. A key deleted through the API while config still declares it is
  created again at the next start.

`GET /api/v1/budgets` reports `origin: config` on a declared budget, and
`GET /api/v1/keys` reports the key's `config_name`. A key whose secret is
already on a key minted through the API takes that key over, keeping its id,
its end users and its ceiling, which is how an existing key moves into config.

Every replica applies the declarations when it starts. They are written in one
transaction, and each insert yields to a row another replica placed first, so
replicas starting together create one budget, one key and one ceiling. A
declaration that cannot be applied, such as a key naming a budget that does not
exist, stops the start and nothing is written.

Declarations are applied before the first-run bootstrap key, so a fresh
deployment that declares a key does not mint and print one. Neither does a
deployment that generated its own master key: it starts from the setup link,
and its operator creates the first key in the dashboard. Set
`bootstrap_api_key: true` (`OTARI_BOOTSTRAP_API_KEY=true`) to have that
deployment mint and print one anyway.

### Mismatched `user` fields

`reject_user_mismatch` (default `true`) is the deployment-wide answer to a request
whose `user` field names someone other than an ordinary key's user (service
keys and the master key name users by design); a key's own value
overrides it, as [Mismatched `user` fields](api-keys.md#mismatched-user-fields)
describes. It is a runtime setting, changed on the Settings page or with
`PATCH /api/v1/settings`.

### Search tools on a key's model list

When neither a key nor its user has an `allowed_tools` list, the key still
reaches search tools through its `allowed_models`; when either has one, even an
empty one, that list decides. The model-list route is a deprecated form, described in
[Model and search tool access](api-keys.md#model-and-search-tool-access). Startup
counts the keys and users that still rely on it, and names the tools some reached
only through a wildcard, which no longer grants them. Give those keys and users an
`allowed_tools` list before a later release removes the form.

## Budget enforcement

These settings govern how [budgets](budgets.md) are enforced:

| Setting | Purpose |
| --- | --- |
| `budget_strategy` | `for_update` (the default), `cas` (lock-free), or `disabled`, which bypasses budget enforcement for the whole deployment. |
| `budget_reservation_ttl_sec` | How long a request's budget hold may stay in flight before it is treated as leaked and returned, as for a crashed request. Defaults to 900 (15 minutes); it must comfortably exceed the slowest request the deployment serves. |
| `budget_estimate_default_output_tokens` | The output tokens a hold assumes for a request that sets no output limit. Defaults to 1024. |

A budget's per-minute limits, `rpm_limit` and `tpm_limit`, are counted in
`rate_limit_store`, so with Redis they hold across replicas; see
[Rate limits across replicas](#rate-limits-across-replicas).

## Provider configuration

The `providers:` block is deprecated in favor of
[hosted providers](hosted-providers.md). At startup, each entry a hosted
provider can hold is copied onto one and is no longer read; see
[Copies of `providers:` entries](hosted-providers.md#copies-of-providers-entries)
and [Upgrading](hosted-providers.md#upgrading).

The `providers` map is keyed by provider instance. Only a provider it declares
(or a hosted provider, or an organization's key) serves requests. A standard
provider needs only its credential, which the entry names:

```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

| Field | Meaning |
| --- | --- |
| `api_key` | The provider's credential, normally a `${VAR}` reference. |
| `api_base` | An endpoint other than the provider's default. |
| `provider_type` | The implementation of a [named instance](models.md#named-provider-instances). Defaults to the instance name. |
| `models` | Model ids offered for discovery, for a backend with no model-listing endpoint. |
| `client_args` | Options for the provider's SDK client. See [Bedrock read timeout](#bedrock-read-timeout). |
| `max_retries` | This instance's SDK retry count. See [Provider retries](#provider-retries). |
| `messages_api` | `completions` (default) or `responses`. See [Messages through the Responses API](#messages-through-the-responses-api). |
| `credentials`, `project`, `location` | Vertex AI only: configure its client. |
| Any other key | Passed to the provider's any-llm call as it is. |

Keyless local backends, providers authenticating from ambient cloud
credentials, and named instances for custom or self-hosted endpoints are
described in [Models](models.md#configuring-a-provider). The instance names
`otari` and `hosted` are reserved for a managed platform's own offerings and
are refused in `providers:`.

`import_configured_providers` (`OTARI_IMPORT_CONFIGURED_PROVIDERS`), default
`true`, copies the entries onto hosted providers at startup; `false` keeps
serving every entry from `config.yml`.

### Automatic routing

```yaml
automatic_routing:
  enabled: true
  session_ttl_seconds: 3600
  # Set only behind an edge that overwrites this header and blocks direct access.
  country_header: CF-IPCountry
  offerings:
    nebius-us: {region: us, capacity: 10}
    nebius-eu: {region: eu, capacity: 10}
    # An exact instance:model entry overrides the instance-wide hint.
    "bedrock:eu.anthropic.claude-sonnet-4-6": {region: eu, capacity: 5}
```

Offering keys name configured provider instances or exact `instance:model`
selectors. Declare a region only when that endpoint actually serves it; Otari
never guesses a provider's location from its brand. `region` accepts `us` or
`eu`; omitted means unknown. `capacity` is a positive relative concurrency
weight, default 1, not a limit. Use rate-limit rules for enforced limits, including
`max_concurrent`. A sticky session falls back when those limits refuse its provider.

Send `Otari-Conversation-Id` to reuse the provider and credential that served the
conversation. `session_ttl_seconds` sets the inactivity expiry (1–86400 seconds,
default 3600), refreshed by successful responses. Requests without a conversation
ID are ranked independently.

Both `country_header` and `default_region` are unset by default. With a trusted
country header, US visitors prefer US endpoints and visitors in EU member
countries prefer EU endpoints. Server-to-server traffic reflects the location of
the calling server; applications that know the end user's preference can send
`Otari-Region: us` or `eu`, which takes precedence. Other locations use
`default_region` if set, otherwise there is no region preference. These settings
allow cross-region fallback and do not enforce data residency.

This is startup configuration, also accepted as JSON in
`OTARI_AUTOMATIC_ROUTING`. Set `enabled: false` to use the baseline catalog order.
The [model guide](models.md#automatic-provider-selection) explains cost, load,
latency, sticky sessions and the limits of local feedback.

### Provider retries

Most provider SDKs retry a failed call themselves before Otari sees the failure.
The OpenAI and Anthropic SDKs, and every OpenAI-compatible provider built on
them, retry a 429, a 5xx or a dropped connection twice by default and wait out
the provider's `Retry-After`, so one request to a throttled provider can make
three upstream calls and take many seconds to fail. Google's SDK (Gemini,
Vertex AI) does not retry by default. A client that retries on top of that
multiplies the calls again.

`provider_max_retries` sets the retry count for every provider instance, and an
instance's own `max_retries` overrides it:

```yaml
provider_max_retries: 0   # OTARI_PROVIDER_MAX_RETRIES

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    max_retries: 1
```

When neither is set, each SDK keeps its own default, except in a routing policy
that can fall over to another candidate (an `on_failure` chain or a weighted or
router pool): there every candidate makes one upstream call, so a failure moves
to the next candidate at once instead of after the SDK's retries. Setting either
value restores retries inside such a policy too.

The setting reaches the OpenAI-compatible, Anthropic (including Azure and
Vertex AI), Groq, Cerebras, Together, Gemini and Vertex AI clients. Other
providers (Bedrock, Mistral, Cohere, Ollama and the rest) keep their SDK's own
retry behavior, which `client_args` can still configure where the SDK allows
it. A retry setting in an instance's `client_args` (`max_retries`, or
`http_options.retry_options` for Google) wins over both values. Neither value
reaches an [owned provider endpoint](provider-endpoints.md), which keeps its
SDK's default outside a policy that can fall over.

### Provider-specific request fields

A chat completion request field that Otari does not know is dropped before the
call, so a parameter only one provider takes never reaches it. An instance can
let named fields through with `request_fields`. Exa's chat endpoint, for
example, takes `text` to return the full text of its sources:

```yaml
providers:
  exa:
    provider_type: openai
    api_base: https://api.exa.ai
    api_key: ${EXA_API_KEY}
    request_fields: [text]
```

A request reaching that instance with `"text": true` sends it to Exa in
`extra_body`, which the OpenAI SDK, and the OpenAI-compatible providers built
on it, merge into the request body. A field the instance does not list is
dropped as before, and so is every listed field on a request another instance
serves, including a routing policy's other candidates. A name the request
already carries (`temperature`, `user`, `safety_identifier`, ...), that shapes
it (`model`, `messages`, `stream`) or that Otari sets itself (a credential,
`api_base`, `extra_body`) cannot be listed, so a caller can never override the
gateway's own values through it. Chat completions only.

### Messages through the Responses API

A request to `/v1/messages` for a model with no Anthropic Messages API of its
own is served as a chat completion by default. An OpenAI instance (or any
instance whose provider has a Responses API, such as Azure OpenAI) can serve it
through the Responses API instead:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    messages_api: responses   # default: completions
```

This reaches models OpenAI serves only on the Responses API (the codex and pro
variants), and keeps a reasoning model's reasoning across turns:

- Each reasoning item comes back as a `thinking` block whose signature carries
  its encrypted content. When a client such as Claude Code sends the block back,
  Otari hands the reasoning back to the provider.
- The signature names the instance that served it; only that instance gets the
  reasoning back, for any model it serves. Such a block is dropped before a
  call to any other instance or provider, and other providers' thinking blocks
  are dropped before a Responses call.
- When a dropped block opened an assistant turn whose tool calls the request
  answers, `thinking` is left off that call (Anthropic refuses it otherwise).
  A fallback from such an instance to an Anthropic model mid-loop therefore
  answers that turn without extended thinking.

Reasoning fields are sent only to a model whose name looks like a reasoning
model (`o1`, `o3`, `o4-mini` and the rest of the `o` series, `gpt-5*` except its
`-chat` variants, and anything with `codex` in it), whether or not the client
asked for thinking, because any other model fails the call over them. An Azure
OpenAI deployment is called by its deployment name, so name a reasoning
deployment after its model (`gpt-5-mini`, not `prod-reasoner`) or its reasoning
is neither requested nor carried across turns.

How the request is translated:

- Every call is sent with `store: false`, so nothing is kept upstream.
- A Messages `thinking` budget becomes `reasoning.effort`: up to 2,048 tokens
  is `low`, up to 16,384 `medium`, more `high`. `output_config.effort` takes
  precedence (`max` becomes `high`). Reasoning summaries are requested unless
  thinking is disabled.
- `stop_sequences` is refused with a 400 (no Responses equivalent); `top_k`,
  `cache_control` and `metadata` are dropped.
- With no `prompt_cache_key`, the request's session (`Otari-Conversation-Id`,
  or the session id Claude Code sends) keys the provider's prompt cache, scoped
  to the caller, so the session id itself never reaches the provider.
- Anthropic's server tools are not forwarded: Otari's own web search, web fetch
  and code execution serve them where configured (see [Built-in tools](tools.md)),
  and one Otari does not serve is dropped.

The setting covers `/v1/messages` traffic only, and only where the instance's
provider has a Responses API; an instance whose provider has none keeps the
default and logs a warning once. Any other value fails at startup. It is read
from the `providers:` block of the gateway's own config.

### Bedrock read timeout

A Bedrock client waits up to 600 seconds for the next part of a response.
botocore's own default is 60 seconds, which fails a streamed response whose
model goes quiet for a minute, as a model writing a long tool call can. A
`config` in the instance's `client_args` takes the botocore `Config` options and
wins over the default:

```yaml
providers:
  bedrock:
    client_args:
      config:
        read_timeout: 120
```

### Runtime provider management

Operators add provider credentials at runtime as
[hosted providers](hosted-providers.md), on the Hosted providers page or through
`/api/v1/hosted-providers`. Config-file entries remain read-only in the
dashboard.

Stored credentials are encrypted with a Fernet key, and where that key comes
from depends on the database:

- **PostgreSQL**: set `OTARI_SECRET_KEY` to a key generated with
  `otari gen-secret-key`. Nothing is stored until it is set.
- **SQLite**: when `OTARI_SECRET_KEY` is unset, Otari generates a key on first
  start and keeps it in a file beside the database file, named after it with
  `.secret-key` added (`/data/otari.db.secret-key` in the container image),
  readable only by its owner. Later starts load it from there. A set
  `OTARI_SECRET_KEY` always wins, and the file is then neither read nor written.

A key beside the database protects against a leaked database file or dump. It
does not protect against a stolen disk or volume, which holds both; set
`OTARI_SECRET_KEY` from a secret store where that matters. Back the key file up
with the database. If the database holds encrypted credentials and neither
`OTARI_SECRET_KEY` nor the key file is present, Otari refuses to start rather
than generate a key that could not decrypt them.

To rotate `OTARI_SECRET_KEY`, configure the new key before the old key,
run the hosted provider and search-tool re-encryption endpoints
(`POST /api/v1/hosted-providers/reencrypt` and
`POST /api/v1/search-tools/reencrypt`), then remove the old key. A subagent
router's key has no re-encryption endpoint yet: re-enter it on the Subagent
router page after the rotation. Losing every configured encryption key makes
stored credentials unrecoverable.

### Hosted providers in config.yml

The `hosted_providers:` block declares [hosted providers](hosted-providers.md),
keyed by name, with the fields the `/api/v1/hosted-providers` body takes
(`provider`, `api_key`, `api_base`, `client_args`, `enabled`,
`restricted_to_organization_id`, `models`, `supported_countries`). `OTARI_HOSTED_PROVIDERS` sets the
whole block as JSON and replaces the file's block whole. The fields, validation
and how a configured entry behaves beside a stored one are in
[The `hosted_providers:` block](hosted-providers.md#the-hosted_providers-block).

### Provider copies

A request that asks a provider's own code execution to run over an attached file
gets a short-lived copy of that file in the provider's account (see
[Files](files.md#a-file-the-providers-own-code-execution-reads)). The account a
copy is in is named by a keyed digest of the credential that made it, so the
digest needs a key of its own: `OTARI_PROVIDER_ACCOUNT_PEPPER`.

While the pepper is unset, Otari turns `files_provider_upload_enabled` off at
startup and logs a warning, so a request that asks a provider's own code
execution to run over an attached file is refused. Otari refuses to start when
the pepper is shorter than 32 characters or equal to the master key or an
`OTARI_SECRET_KEY` key. Generate one with
`otari gen-provider-account-pepper` or `openssl rand -base64 32`, and keep it in
your secret store. Rotating it costs nothing but a fresh copy of each file the
next time a request uses it.

### End-user identifiers

Every request reaches a provider under the account of the credential that
serves it, which many users usually share. A provider
that enforces its usage policy per end user needs an identifier for each, or a
block meant for one person lands on the whole account.

Otari sends each provider call a digest of the workspace, the resolved user,
and the user tag the caller sent, if any:

- The tag is `safety_identifier` (else `user`) on chat completions and
  Responses, and `metadata.user_id` on Messages. A tag equal to the resolved
  user counts as none.
- The provider never sees the tag or an Otari id.
- The same tag from the same user in the same workspace always gets the same
  identifier. Since the resolved user is always in the digest, tagging a
  request with another user's id never gives it that user's identifier.
- The Playground ignores `user` and `safety_identifier` in its request body, so
  its calls carry the signed-in member's identifier.

`user` also names the billed end user on a service key, and on any other key
must match the key's user unless `reject_user_mismatch` is off.
`safety_identifier` names nobody billed and nothing checks it. Send it when one
key serves several end users that `user` does not tell apart; otherwise they
all share the key holder's identifier. A caller that sends a new tag on each
request evades provider blocks, which Otari cannot detect; block that customer.

| Provider | Field |
| --- | --- |
| OpenAI, Azure OpenAI | `safety_identifier` |
| Cerebras, DeepSeek, Fireworks, Groq, OpenRouter, xAI | `user` on chat completions (DeepSeek's `user_id`) |
| Anthropic, including Azure and Vertex AI | `metadata.user_id` |
| Bedrock | `requestMetadata.otari_user`, except `metadata.user_id` on Messages to a Claude model that sets no `output_format` |
| Gemini API | the `safety_identifier` label, cut to the 63 characters a label value allows, on chat completions and Messages |
| Meta | `safety_identifier` on chat completions |
| Any other provider with a native Messages API, on Messages | `metadata.user_id` |

Bedrock's abuse detection reads no identifier, but `requestMetadata` lands in
the account's invocation logs, where an AWS abuse report's prompt ids are
traced back. Every other provider gets nothing (Vertex AI and Mistral have no
such field; for others an unknown field can fail the call). A Responses or
Messages request served as a chat completion gets the provider's chat field.
The `vision_describe_model` call carries the identifier too.

The identifier cannot be turned off; OpenAI's partner program for gateways
requires one on every request.

Set `OTARI_PROVIDER_USER_IDENTIFIER_KEY` to a random value of at least 32
characters (`openssl rand -base64 32`) to key the digest. Without it the digest
is unkeyed (the same ids give the same identifier on any deployment) and Otari
logs a warning at startup. Otari refuses to start when the key equals the
master key or an `OTARI_SECRET_KEY` key. Rotating it gives every user a new
identifier and resets provider blocks and history, so rotate only after a leak.

#### Tracing an identifier back

A provider that reports abuse names the identifier, and it is a one-way digest.
Each usage row stores the identifier its request's provider calls carry, in the
indexed `usage_logs.safety_identifier` column, so the database answers who it
was:

```sql
SELECT workspace_id, user_id, api_key_id,
       MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen,
       COUNT(DISTINCT request_group_id) AS requests
FROM usage_logs
WHERE safety_identifier = '<identifier>'
GROUP BY workspace_id, user_id, api_key_id;
```

The rows keep an identifier after a key rotation, so an old one still traces.
For a Gemini label, which is cut to 63 characters, match the prefix:
`WHERE safety_identifier LIKE '<label>%'`.

The rows hold the user behind the key, not the caller's tag; when the
identifier came from a tag, the workspace's owner maps it to their own end
user. Rows have no identifier for imported usage, the pass-through endpoints
(embeddings, images, audio, moderations, batches), requests refused before the
user was resolved, and rows written before Otari stored it.

### Supported countries

A provider may serve only some countries. OpenAI's partner program for
gateways, for example, requires a gateway to refuse OpenAI traffic from outside
OpenAI's [supported countries](https://platform.openai.com/docs/supported-countries).
`provider_supported_countries` lists, per provider type, the ISO 3166-1 alpha-2
codes that may reach it:

```yaml
provider_supported_countries:
  openai: [US, CA, GB, DE, FR]  # copy the provider's own full list
client_country_header: CF-IPCountry
```

OpenAI's list, as 208 codes taken from that page on 2026-10-23, is in
[openai-supported-countries.json](public/openai-supported-countries.json), a
JSON list the dashboard's Load from JSON file button reads for a hosted
provider. OpenAI changes the page, so check the file against it before relying
on it. The page allows Ukraine "with certain exceptions", which a country code
cannot express, so `UA` allows all of it.

A provider type with no entry serves every country, and an empty list closes
the provider to all of them. Quote Norway's code, `"NO"`: YAML reads a bare
`NO` as false, which is refused as no country code. As an environment variable,
the block is a JSON object:
`OTARI_PROVIDER_SUPPORTED_COUNTRIES='{"openai": ["US", "DE"]}'`.

A request from a country not on the list, or from one no source knows, is
treated this way:

- On chat completions, Messages and Responses, a routing policy or a catalog
  id drops that provider's candidates and falls over to the next one. A plain
  model is refused with a `403` and the `country_not_supported` code, and so
  is a plan with no candidate left, without the code.
- The call that describes an image for a text-only model is skipped, so the
  image falls back to OCR or is dropped, as when no `vision_describe_model` is
  set.
- Embeddings, images, audio, moderations, rerank, decisions and batches are
  refused with a `403` and the `country_not_supported` code. A batch's status,
  cancel, list and results calls are checked as its creation is, since they
  reach the same provider.

Every refusal except a batch's leaves an error row in usage.

The list keys on who serves the call: an instance named for the provider or
declaring its `provider_type`, a hosted provider of that type, and an
organization's own key for it. An instance declared `openai-compatible` is a
server that speaks OpenAI's protocol, such as vLLM or Ollama, so the `openai`
list does not apply to it.

A hosted provider can also have its own `supported_countries`, set in the
dashboard's Add and Edit dialogs, through `/api/v1/hosted-providers`, or in its
`hosted_providers:` entry:

- It applies only on that provider's credential, never to an organization's
  own key for the same provider.
- Both lists apply: with `openai: [US, DE]` above and
  `supported_countries: [DE]` on a hosted `openai`, only Germany is served on
  that hosted provider.
- A routing policy or catalog id skips a hosted candidate the list leaves out;
  a plain model, pass-through call or batch call is refused with `403`
  (`country_not_supported`).
- Through the API, null serves every country and an empty list none; in the
  dashboard, a blank field serves every country.
- It also needs a country source (below); without one, every request on that
  provider is refused as from an unknown country.

The country comes from a header or from a country file. With both, the header
is read first and the file only when the header is absent.

- `client_country_header` names a header that a proxy in front of Otari sets,
  such as Cloudflare's `CF-IPCountry` or CloudFront's
  `CloudFront-Viewer-Country`. A header that is present decides alone, so a
  value that is no country code, such as Cloudflare's `T1` for Tor or `XX`,
  counts as unknown. Set it only when every request reaches Otari through that proxy and
  the proxy replaces any value a client sends. Otherwise a client picks its own
  country.
- A country file is a MaxMind DB file in which Otari looks up the client
  address. Behind a proxy that address is the proxy's own unless
  `forwarded_allow_ips` trusts it (see
  [Behind a reverse proxy](deployment.md#behind-a-reverse-proxy)). Two files
  are free:
  [GeoLite2 Country](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data),
  which needs a free MaxMind account and is kept current by
  [geoipupdate](https://github.com/maxmind/geoipupdate), and
  [DB-IP Lite Country](https://db-ip.com/db/download/ip-to-country-lite), a
  monthly file that needs no account. DB-IP Lite is under CC BY 4.0, which asks
  a service that uses it to credit DB-IP, for example with a link to
  `https://db-ip.com` where the service names its data sources. Reading a file
  needs the `geoip` extra (`pip install gateway[geoip]`), which the Docker
  image installs. A file reaches Otari in one of two ways:
  - Upload it in the **Country file** card on the dashboard's Settings page, or
    with `PUT /api/v1/settings/country-database` (a country database, up to
    32 MiB). Otari stores it in the database; the uploading replica reads it at
    once and the others within 30 seconds. A rejected file, or one a replica
    cannot open, leaves the previous file in use. The card shows the file's
    type and build date (DB-IP publishes monthly, MaxMind twice a week).
  - Mount it into the container and set `client_country_database` to its path,
    as [Deployment](deployment.md#a-mounted-country-file) shows. A replaced file
    is read again without a restart, and Otari refuses to start on a file it
    cannot read. While it is set, a stored file is not read.

Otari refuses to start with a `provider_supported_countries` list and no
source: no header, no `client_country_database` and no stored file. For the
same reason the stored file cannot be removed while it is the only source for
that list or for the supported countries of an enabled hosted provider. A
private address, or one the file does not hold, has no country, and
a listed provider refuses it.

A VPN changes the country a request appears to come from, so this check does
not replace the account and billing checks a provider may also ask for, which
need billing data a self-hosted Otari does not hold.

## Pricing

Pricing keys use `provider:model` or `instance:model`:

```yaml
pricing:
  openai:gpt-5:
    input_price_per_million: 1.25
    output_price_per_million: 10.00
```

Each start writes the config file's prices to the database. An entry with no
`effective_at` adds a version only when its rates differ from the version in
force, whoever set that version, so an unchanged file adds nothing, and a start
puts the file's rate back over one set since through `/api/v1/pricing` or the
dashboard. To keep a rate set there, remove the model's entry from this file or
give it the same rate. An entry with `effective_at` adds its version once, and a
version already stored at that instant is kept. An entry for a model of a
[hosted provider](hosted-providers.md) is skipped with a warning, because this
file prices only the `providers:` and `decision_providers:` instances and the
tools it declares: price a hosted provider's model through the pricing API or
the Hosted providers page.
Rates and settled costs use decimal arithmetic and costs are rounded
once to a micro-dollar. Use PostgreSQL for durable accounting.

A stored price change, made through the API, the dashboard or a catalog update,
applies at once in the gateway process that wrote it. Each other worker or
replica keeps the rates it has read for up to 30 seconds, so it applies the
change within that window, as it does a rate scheduled to start later.

### Default pricing

`default_pricing: true` prices from a models.dev snapshot when no stored price
exists. Explicit database or config pricing always wins. A snapshot ships inside
the package, so defaults work offline and a fresh deployment needs no network;
once an operator accepts a newer one (below), that one serves instead. The
`models_dev_metadata` setting only controls whether the gateway fetches the
descriptions and capabilities shown beside a model; with it off, they come from
the bundled snapshot. Prices come from the snapshot whether or not it is on.

A model is matched on its provider and id, including a pinned HuggingFace
backend and Bedrock geo-prefixed ids. A bare id that providers list at different
prices stays unpriced rather than guessing, so set a price for it or name the
provider.

Default pricing is off by default because provider catalogs and reseller rates
change.

With `require_pricing: true`, a budgeted request with no effective price is
refused with a 402 instead of bypassing the budget. With `default_pricing` on,
the 402 points at `GET /api/v1/models` (an unpriced model is most often a
misspelled name) and at adding a price; with it off, it offers adding a price
or turning the fallback on. Two other refusals take precedence:

- a model whose provider has no credential: `provider_not_configured`
- a selector nothing resolves (no provider, alias, policy or catalog id):
  `invalid_model`

With `require_pricing: false`, such a request is served with no cost on its
model tokens and no inline `cost_usd`; priced gateway tools it ran are still
charged. On an organization's own provider keys, an unpriced model is then
switched on and served free, where `require_pricing: true` keeps it off until
priced (see [Who is shown which models](models.md#who-is-shown-which-models)).

The gateway logs a warning per unpriced model at most once an hour per
process. Operators see a dashboard banner naming the models that served
unpriced traffic in the selected workspace in the last 24 hours, linked to
those requests on the Overview, where each can be priced.

### Keeping the defaults current

A catalog update fetches the latest models.dev prices and applies them, then
asks every enabled hosted provider what it lists, offers what is new, and
prices each model at the prices now in force. A model waiting for a price is
served once it has one, so a model released after the bundled snapshot arrives
priced and serving in one update.

With `models_dev_metadata` on, an update also stores the catalog of model
facts it downloaded, which gives each model its kind and details, so the
dashboard does not show a price for a model it still calls unknown. Storing it
is best effort: a database error there is logged and the price update goes
on, and a v1 catalog, read when v2 cannot be, lists every model as chat.

**Update catalog** on the Hosted providers page runs it at once, after a preview
of the price changes and of each provider's new models
(`POST /api/v1/hosted-providers/catalog-update/preview`, then
`POST /api/v1/hosted-providers/catalog-update` with the preview's `digest`).
Its optional cleanup also removes the prices of models no hosted provider
serves, with their history; the cleanup never runs on a schedule.

`pricing_refresh` decides whether the update also runs on its own, every
`pricing_refresh_interval_seconds` (default one day, minimum five minutes):

- `manual` (the default) never runs it on a schedule: only an operator does.
- `review` runs it, but holds changed prices for an operator to approve on the
  Hosted providers page. The model lists are still refreshed, at the prices
  already in force, so a new model that only the new prices cover waits for the
  approval. `GET /api/v1/pricing/refresh/pending` reads what is held; nothing is
  metered differently until it is approved. On a deployment with nobody to make
  that call, prefer `auto`.
- `auto` runs it and applies changed prices without review, unless they look wrong: an
  update that would lose or remove more than a tenth of the priced models is
  held for review as under `review`.

`POST /api/v1/pricing/refresh/reject` discards held prices.

An update fetches models.dev's v2 catalog, and its v1 catalog when v2 fails,
from `models_dev_base_url`. It carries every rate the catalog states, per
second, image, character and minute as well as per token, so a video, image,
speech or transcription model is priced by an update as a chat model is. The default is Mozilla.ai's fork of models.dev,
which publishes the same documents at the same paths, built from models.dev's
catalog plus the models it does not list yet; each of those is also offered to
models.dev, and leaves the fork once it lands there. `https://models.dev` reads
models.dev itself, and a private mirror works when it serves the same paths.
An update refuses models.dev data the gateway cannot read: a document that lists
no models, or one with more than a quarter of its models damaged. When neither
catalog can be read, the prices in force stay, and the update reports an error.

Each accepted snapshot is kept, and a request or report priced for a past
instant (`as_of`, usage backfills) uses the snapshot in force then, or the oldest
one for a time before the first. History therefore starts at the first accepted
snapshot; models.dev publishes no dated price changes of its own.

The bundled snapshot is the whole models.dev catalog, so it also carries the
model facts a listing shows. To refresh it, as a maintainer does at release, run
`python scripts/update_models_dev_snapshot.py` (or pass `--input FILE` with a
saved v2 `api.json`, adding `--input-version v1` for a v1 one; `--base-url`
fetches from somewhere other than the default) and commit
`src/gateway/data/models_dev_catalog.json`.

Rejecting a pending update means "not now": nothing remembers what was
rejected, so the next check re-offers the same update while upstream still
differs from what is active. One worker performs each scheduled update,
whichever claims the run first, so a deployment running several does not fetch,
accept or refresh once per worker.

Every accepted snapshot is recorded with who accepted it, `operator` or
`schedule`, and how many models it priced; `GET /api/v1/pricing/snapshots` lists
the history, which keeps the newest thirty. `GET /api/v1/pricing/drift` puts every stored deployment rate beside
the default it shadows, so a config-file price that has fallen behind the
provider's list is visible before it costs anyone. Both are operator reads, and
`pricing_refresh` can be changed at runtime through `PATCH /api/v1/settings`.

Each stored rate also carries its `unit` (one of the units under
[Each endpoint in its unit](#each-endpoint-in-its-unit)) and its `origin`:
`config` for a rate from this file, `api` for one set through the pricing API or
the dashboard, `seed` for one a catalog update copied from models.dev for a
model a hosted provider serves, and `migration` for one an upgrade wrote. It is
null on a rate stored before origins were recorded. A catalog update moves only
a `seed` rate.

A `seed` rate and a models.dev default are one headline price per model, so a
completion whose provider states what it charged is settled at that charge
instead, and its usage row and inline `pricing_source` read `reported`. Only
OpenRouter states one: its `usage.cost`, plus `cost_details.upstream_inference_cost`
on a key in its BYOK mode, where `cost` is OpenRouter's fee alone. Its price
depends on the upstream it routes a request to, which no single rate matches.
A rate set in this file, through the pricing API or on the dashboard still
prices the completion. The hold placed before the request is still estimated
at the rate. With `require_pricing` off, a model no rate prices is settled at
the stated charge too, rather than recorded with no cost.

### A public catalog

`public_catalog: true` serves `GET /api/v1/catalog/models` and the dashboard's
Models page to a visitor with no credential, so a deployment can show what it
serves before anyone signs up. A visitor sees the configured `providers:`
instances and any hosted models the deployment serves to every organization,
priced at the deployment's rates, and never a hosted provider restricted to one
organization, an organization's
override, key-scoped allow-list, or usage. A caller who sends a credential is
served as that caller, valid or not. The setting is off by default and can be
changed at runtime.

Anonymous reads are throttled per client address by
`public_catalog_rate_limit_per_minute`, sixty a minute by default and its own
budget: `rate_limit_rpm` keys on an authenticated user and covers no anonymous
path, and `dashboard_login_rate_limit_per_minute` is sized for password
attempts, not for browsing. Set it to `null` to remove the limit.

Two limits of that throttle are worth knowing before a catalog is put on the
open internet. Behind a reverse proxy every visitor shares the proxy's
address, and one scraper exhausts the budget for everyone, unless
`forwarded_allow_ips` trusts that proxy; see
[Behind a reverse proxy](deployment.md#behind-a-reverse-proxy). And the counter
is per worker, so a deployment running N workers serves up to N times the
configured number.

The hosted models a visitor sees are the deployment-wide roster (what
`ModelProviderPort.get_hosted_models` answers with no organization). No
organization's providers, overrides, or usage are public, whatever the flag is
set to.

### Cache and tiered pricing

Optional cache fields reprice cached input when a provider reports it:

- `cache_read_price_per_million`
- `cache_write_price_per_million`

A Claude model (one whose model name contains `claude`, on any provider) bills
a cache write made with a one-hour TTL at twice the input rate, which is
Anthropic's price for it, whenever its `cache_write_price_per_million` is set
above zero. Every other cache write bills at `cache_write_price_per_million`.

Use `pricing_tiers` for a rate that applies to an entire request after an input
token threshold. The OpenAPI pricing schemas and dashboard editor show the
accepted shape.

### Every price in its unit: `rates`

A price row also holds `rates`: every price of the model, as the models.dev
`cost` block, in the document's own shape. Each modality (`text`, `image`,
`audio`, `video`) has an `input` and an `output` price, each with its `unit`,
`cache` prices and `overrides`. `web_search` sits beside the modalities, and
so do `multipliers`. A price there is per its unit as published: one image,
one million tokens, one thousand searches.

Give `rates` instead of the per-million fields, in this file or on the
pricing API, and the per-million fields are read out of it: the text prices
first, else the one other price the block holds, scaled to per million. A
block that gives both is refused, and so is one with a key the shape does not
have.

```yaml
pricing:
  openai:gpt-image-1:
    rates:
      image:
        output: {price: 0.04, unit: image}
```

That row stores `unit: image` and `input_price_per_million: 40000`. Every row
reads back with both: one set through the per-million fields shows them in
`rates` too, in the slot its unit chooses. Each endpoint reads the slot it
bills under ([Each endpoint in its unit](#each-endpoint-in-its-unit)), so a
per-megapixel image price, a per-minute audio price and a per-second video
price are each read from their slot, and a `web_search` price prices the
searches a model's provider runs itself
([Searches a provider runs itself](tools.md#searches-a-provider-runs-itself)):

```yaml
pricing:
  gemini:veo-3.0-generate-001:
    rates:
      video:
        output: {price: 0.4, unit: second}
```

The `overrides` and `multipliers` a block holds apply to each request
([Overrides and multipliers](#overrides-and-multipliers)).

### Overrides and multipliers

A price in a `rates` block can change with the request. An override on an
`input`, an `output` or the `web_search` price replaces the prices it names
when its condition holds. A multiplier beside the modalities scales the prices
it names when its condition holds. Otari applies them in the order models.dev
states, to every modality and to `web_search` alike:

1. Each override whose condition the request meets merges over its price, in
   order, field by field: an override that names only a cache read price keeps
   the base cache write price, and a later override wins over an earlier one.
2. Then each multiplier whose condition the request meets scales the prices it
   names. A number `factor` scales every price of the block, the cache, the
   reasoning and the `web_search` prices included. An object scales only the
   prices on the paths it names: `{text: 2}` scales every text price, and
   `{text: {output: 2}}` only the text output price.

```yaml
pricing:
  openai:gpt-5:
    rates:
      text:
        input:
          price: 1.25
          overrides:
            - when: {context: {gt: 272000}}
              price: 2.5
        output: {price: 10}
      multipliers:
        - when: {service_tier: {equals: flex}}
          factor: 0.5
```

A condition holds when the request's value for each key it names meets that
key's operator: `equals`, `in`, `gt`, `gte`, `lt`, `lte` or `min_items`, and
`all`, `any` and `not` combine conditions. `equals` and `in` compare text
without regard to case. `gt` and the other comparisons order text as written,
which is how an `HH:MM` time or an ISO date compares. Otari answers these keys:

| Condition | Its value |
|---|---|
| `context` | The request's billable input in tokens: the estimate before the call, what the provider reports after it |
| `service_tier` | The tier the request asks for, before the call. After it, the tier the provider reports serving the call at; a response that names no tier settles at the price that names none |
| `resolution` | An image's `size`, or a video's `resolution`, else its `size` |
| `region` | The region a Messages request names in `inference_geo`, before the call. After it, the region the provider reports running the call in; Anthropic reports it, including for a request that names none and runs in the workspace's default. Unknown where the provider reports none, so a rule on it is skipped |
| `quality`, `style` | An image's `quality` and `style` |
| `aspect_ratio` | A video's `aspect_ratio` |
| `duration` | A video's length in seconds |
| `date`, `utc_days`, `utc_hours` | When the request started, in UTC: a video job's start, a batch's creation, an imported event's own time |
| `input_video` | Whether a video request names `reference_videos`. Unknown for an imported event or a video job started before Otari kept it, so a rule on it is skipped |
| `generate_audio` | A video request's `generate_audio`. Unknown when the request leaves it to the provider's default, so a rule on it is skipped |
| `websocket` | False: Otari sends nothing over a websocket. Unknown for an imported event, so a rule on it is skipped |

A rule whose condition names any other key, such as `mode` or `thinking`, is
skipped, as models.dev's schema says, and the base price applies. A key Otari
answers that the request gives no value for is a condition that does not hold,
so `{service_tier: {equals: priority}}` does not hold for a request that names
no tier, and `{not: {service_tier: {equals: priority}}}` does. A models.dev
override on context with `gt` starts above its bound. A tier given in
`pricing_tiers` is written as an override with `gte`, so it starts at its
`min_input_tokens`.

This file and the pricing API refuse a rule whose condition names a key or an
operator models.dev does not list, or gives an operator an operand it cannot
take, so a typo is never a rule that silently never matches. A block read back
from a row is shown as stored. Each charge line of a usage row names, under
`rules`, the condition of each rule that changed the price it bills at, in the
order they applied.

### Each endpoint in its unit

A rate's `unit` decides what it bills, and the slot of its `rates` block an
endpoint bills under decides which price it reads. A rate in `token` bills the
tokens the provider reports, a rate in `request` bills each request (or each
search unit, for rerank), and a rate in a measured unit bills the quantity the
endpoint measured in it. The per-million fields are always per million: USD
per million tokens, requests or images, so a $0.04 image is
`input_price_per_million: 40000`, which is what the dashboard stores when you
enter $40 per 1,000 images. A `rates` price is per its unit as published. A
rate never bills a count of another unit, so a per-request price is never read
as a price per image, and a token price is never read as a price per request.

| Endpoint | Slot | What it measures or reports | Rates that price it |
|---|---|---|---|
| Embeddings | none | Input tokens | `token`; `request` bills one request |
| Rerank | none | Search units, and tokens where the provider reports them | `request` per search unit, or `token` |
| Image generation and edits | `image` output | Images returned, their megapixels from the `size` asked for, and tokens for a token-billed model such as `gpt-image-1` | `image` per image, `megapixel` per megapixel, `request` per request, or `token` |
| Audio transcription | `audio` input | The length the provider reports, else the length of the uploaded file, and tokens for a token-billed model such as `gpt-4o-transcribe` | `second` per second, `minute` per minute, `request`, or `token` |
| Audio speech | `text` input | The characters of `input` | `character` per million characters, or `request` |
| Moderations | `text` input | One request | `request`; a rate priced at zero settles free |
| Videos | `video` output | The seconds the job reports, else the seconds asked for; a job neither gives a length is billed what the provider charged, else 8 seconds | `second` per second ([Videos](api-reference.md#videos)) |

A rate that cannot price what an endpoint reports leaves the request unpriced:

- A `token` rate on an endpoint that reports no tokens (a `dall-e-3` image, a
  moderation): the row keeps no cost and the needs-pricing view lists it. A
  rate priced at zero everywhere settles such a request free.
- A measured unit the request cannot measure, such as a per-megapixel rate on
  an image whose `size` is `auto` or unset.

With `require_pricing` on, speech, image generation, embeddings, rerank and
direct search refuse such a rate (an `image` rate on embeddings, a `token`
rate on speech) rather than serve unpriced. Audio transcription and
moderations are still served, with no cost on the row. A missing cost is
unknown, not free.

The unit names are singular, as models.dev writes them. Otari's earlier names,
`tokens`, `requests` and `images`, are still read, in this file and on the
pricing API, as `token`, `request` and `image`: a rate sent as
`unit: requests` is stored and served as `unit: request`. A later release will
refuse the old names; see [Upgrading](#upgrading).

The catalog prices each slot in its unit, so a catalog price reaches every
endpoint: `whisper-1` is priced per minute of audio and `tts-1` per character
of text, with no text token price at all, and a token-billed image or
transcription model is priced on the tokens it reports.

Image generation reserves the images asked for, at the `size` asked for,
before dispatch, and settles on the images returned. Speech reserves the
characters of its `input` and answers with its settled cost on response
headers, since its body is audio (see
[Request ID and inline cost](api-reference.md#request-id-and-inline-cost)).
Moderations reserve one request. Transcription holds
nothing before dispatch, so a priced transcription can exceed a budget by one
request's price.

A transcription model that reports tokens and no length, such as
`gpt-4o-mini-transcribe`, is billed at a `minute` or `second` rate for the
length Otari reads from the uploaded file. Otari reads the length of a WAV,
MP3, M4A or MP4, Ogg (Vorbis or Opus), FLAC, AIFF or WMA file by its content,
not its name. A file it cannot read, such as a WebM upload, gives no length,
so that request keeps no cost at a `minute` or `second` rate. A length the
provider reports always wins over the length read from the file.

A search tool declared in this file takes its rate as `price_per_call`, in USD,
on its own entry:

```yaml
search_tools:
  exa-search:
    provider: exa
    api_key: ${EXA_API_KEY}
    price_per_call: 0.005
```

It is stored as a price row of kind `search` keyed by the tool's name, apart
from every model price, so a model price is never read as the tool's.

- The same row can be set through `POST /api/v1/pricing` with
  `"kind": "search"` and the tool's name as `model_key`; that is how a tool
  stored from the dashboard is priced.
- A `price_per_call` in the file adds a new version at startup only when it
  differs from the rate in force, so an API-set rate for such a tool lasts
  until the next restart (as a `pricing:` entry without `effective_at` does).
- A tool stored from the dashboard under the same name and kind replaces this
  entry, and its `price_per_call` is not written while the stored tool exists;
  a rate an earlier start wrote stays in force until a newer one replaces it.
  A start whose stored tools fail to load writes no `price_per_call`.

How a search is charged:

- It reserves the rate against the caller's budget before it runs, and
  settles at the provider's reported charge if any, else at the rate.
- A search on a workspace's organization search key is priced under
  `otari:web_search` instead.
- A tool's own rate also prices the model's searches on it.
- With `require_pricing` on, a request that would run an unpriced tool is
  refused with a 402 naming the price to add (even if the provider reports its
  charge), unless the key is exempt from budgets. A zero rate is a valid
  price. With it off, such a search holds no dollars but still holds one
  request against a request-count budget.

A fetch instance's `price_per_call` is stored the same way, as kind `fetch`.
The built-in fetcher cannot be declared, so its rate is set through the pricing
API, kind `fetch` and `model_key: builtin_fetch`. Either alone prices the fetch
tool's calls on that instance; `otari:web_fetch` prices only a fetch on an
organization's fetch key.

A tool rate keyed as a model price, under `pricing:` or the pricing API as
`<provider>:<instance>` (`exa:exa-search`, `builtin:builtin_fetch`), is
deprecated: it still prices its tool, behind the tool's own row, until a later
release stops reading it; startup names every such row and how to replace it. A key under a
tool's provider that names no configured tool is a model's price like any
other, never a tool's.

Rerank accepts either unit. A rerank model priced `unit: token` is charged
on the input tokens the provider reports. Providers that bill reranking per
search unit (one query over up to 100 documents, a long document counting as
several) report that count and no tokens, so give such a model `unit: request`
and it is charged the reported units at `input_price_per_million / 1,000,000`
each, or one unit when the provider reports none. The budget reservation holds
the units the documents imply, from their count and length, and settles at the
reported count. The default-pricing dataset carries no rerank rates, so a rerank
model needs its own pricing row, for example `cohere:rerank-v3.5` at
`input_price_per_million: 2000` (USD 2 per 1,000 searches) with `unit: request`.

A model served over chat completions, the Responses API, or Messages can be
priced the same way, for an upstream that bills per call and reports little or
no token usage, such as an answer endpoint served through an OpenAI-compatible
provider instance named `exa`. Give its pricing a `price_per_request` in USD:

```yaml
pricing:
  exa:exa:
    price_per_request: 0.005
```

`price_per_request` is the readable form of `unit: request`: Otari stores it as
`unit: request` with `input_price_per_million` a million times the price
(`5000` here), and `GET /api/v1/pricing` reports both. `POST /api/v1/pricing`
takes it too. It replaces the token prices rather than adding to them, so it
is refused beside an `input_price_per_million` that differs, an
`output_price_per_million`, a cache price, `pricing_tiers` or another `unit`.
Writing the rate in full as `unit: request` still works.

Such a model is charged its price for each successful call, streamed or not:

- Token counts are recorded and count toward a token budget, but are not
  priced.
- A failed call (including a stream that errors or is abandoned) costs
  nothing.
- The budget reservation holds the flat amount, not a token estimate.
- A stream that ends without usage is charged the flat amount, so
  `stream_missing_usage_policy` does not apply.
- Default pricing never supplies a per-request rate, so the model needs its
  own pricing row.

## Search tools

`search_tools` names the deployment's search instances. Direct `POST /api/v1/search`
calls use them by name, or the default one (below) when they name none, and the
in-loop `otari_web_search` tool uses the default one. `fetch_tools` names fetch instances, for the `otari_web_fetch` tool
and for enriching search results. Both can also be stored at runtime through
`/api/v1/search-tools`, a fetch instance with `kind: fetch` and listed with
`?kind=fetch`, or from the dashboard's deployment Web search page, which also
sets each one's price and the defaults. The listing gives each instance's price
per call and the row it comes from (see
[Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool)). A stored
instance wins over a configuration file entry of the same name and kind.

```yaml
search_tools:
  local:
    provider: searxng
    api_base: "http://searxng:8080"
  exa:
    provider: exa
    api_key: "${EXA_API_KEY}"
    options: {type: auto}
    fetch_tool: exa-fetch
fetch_tools:
  exa-fetch:
    provider: exa
    api_key: "${EXA_API_KEY}"
    price_per_call: 0.001   # USD, when Exa reports no cost
web_search_default_tool: exa
web_fetch_default_tool: builtin_fetch
web_search_max_calls: 10
```

`GET /api/v1/search-tools/providers` publishes the supported search providers,
and with `?kind=fetch` the fetch providers: whether each requires an `api_key` or
`api_base`, and the native options an instance may set. Any signed-in user or
API key can read it. Only the deployment's operator also sees which of the
deployment's instances use each provider, and the `api_base` a `searxng`
instance inherits from `web_search_url`. Provider options and request filters
are covered in [Built-in tools](tools.md).

An instance takes its key only from its own `api_key`, such as
`"${EXA_API_KEY}"` above, as an LLM provider's `providers:` entry does. An entry
for a search provider that needs a key and carries none stops startup, where an
LLM provider's entry is ignored with a warning. A tool carrying an `api_key` must use an HTTPS
`api_base`; a keyless local SearXNG endpoint may use HTTP.

Rules for instances:

- `builtin_fetch`, the built-in fetcher, is a fetch instance that always exists and
  cannot be declared. A search instance's `fetch_tool` names the fetch instance that
  enriches its results; without one, `web_fetch_default_tool` does. Each page it
  reads is billed as a fetch on that instance, at the cost its provider reports, or
  else at the instance's price.
- A name contains no `/` or `:`, is not `builtin_fetch` or `none` in any case
  (`None` and `BUILTIN_FETCH` are reserved too), and is unique across both maps.
- `options` are checked against the provider's options. A search instance may also
  set `max_results` and `time_range`, and a fetch instance `max_chars`, which every
  provider of that tool takes. In the loop, a search instance's `max_results` can
  only lower the request's count, and on a direct call it is the count when the
  request names none. A fetch instance's `max_chars` cuts what each fetch tool call
  returns, and what an enrichment fetch asks for when it is below the 1,500
  characters a search result shows.
- A `search_tools` entry that breaks the name rules, or whose `options` carry a key
  the provider does not know or a value it refuses, still loads, with a warning
  that names the entry and the problem. Fix it: a later release refuses it, and
  once the instances serve requests, the options named are left out of every call.
  A `fetch_tools` entry that breaks the name rules, or reuses a configured search
  instance's name, stops startup; a stored one is left out, with an error.

Writes through `/api/v1/search-tools` are held to these rules in what they set, so
rotating the key of an instance stored before the rules still works. When the
in-loop default is the single search instance, with no default set and no legacy
settings, adding a second there first sets `web_search_default_tool` to the first,
so in-loop search does not turn off, and the response says so. That runtime value
wins over the configuration file until it is cleared.

`POST /api/v1/search-tools/test` runs one search or one fetch on an instance before
it is saved, and `POST /api/v1/search-tools/{name}/test` on one that exists. Each
says whether the call worked, the error's tag when it did not, and how many hits or
characters came back, never the content. `builtin_fetch` is tested as a fetch tool
call runs it: through the built-in fetcher, with no domain policy and no private
hosts.

The defaults and the call limit are runtime settings, also set from the
dashboard's deployment Web search page or `/api/v1/tool-settings`. Clearing a runtime value falls back to the configuration
file or environment, then to the built-in default.

- `web_search_default_tool`: the search instance the in-loop tool uses when no
  organization key applies, and the one an unnamed direct call uses. It must be one
  whose provider the in-loop tool can search with; `none` turns off the deployment's
  own in-loop search, while a workspace's organization search key still serves it.
  When unset, the in-loop tool uses the search backend the legacy settings
  describe (`web_search_provider` with its key, else `web_search_url`), else the
  only search instance, when there is exactly one and the in-loop tool can search
  with its provider. Several search instances with neither log a warning at
  startup.
- `web_fetch_default_tool`: the fetch instance for the fetch tool and for
  enrichment. Default `builtin_fetch`.
- `web_search_max_calls`: how many search and fetch calls one request may make
  together. Default 10. A call past it gets a tool error the model can recover
  from.

A default that names no instance, set in the file or earlier at runtime, is treated
as unset, with a warning at startup, or from the refresh that first finds it so.

### The legacy SearXNG settings

`web_search_url` and `web_search_engines` are deprecated, and a coming breaking
release removes them. `web_search_url` is the SearXNG instance the in-loop tool
searches when no search instance is its default and no `web_search_provider`
with its key is set, with the engines in `web_search_engines`, else
`duckduckgo`, `mojeek`, `qwant` and `wikipedia`. A `searxng` instance without an
`api_base` of its own inherits `web_search_url`, and one without an `engines`
option inherits `web_search_engines` when it is set, else searches its SearXNG
instance's own defaults.

That release converts a value stored from the dashboard when that is
unambiguous. A stored `web_search_url` becomes a stored instance named
`searxng`, with the stored engines, else the four above, and
`web_search_default_tool` names it when no default is in effect. A stored
instance that inherits a stored value gets its own copy, and a stored
`web_search_engines` that neither a stored URL nor a stored instance uses is
deleted. The release refuses to start:

- while a value is set in the configuration file or the environment, which otari
  cannot rewrite, even beside a stored one;
- while an instance inherits a value it cannot copy: a stored instance one from
  the configuration file or the environment, or an instance in the configuration
  file a stored one;
- when a stored `web_search_url` cannot be converted: another tool holds the
  name `searxng` (a fetch tool, a stored tool otari cannot read, or a search tool
  with another address or engines), or the new instance would move the unnamed
  calls of `POST /api/v1/search` off the instance that serves them.

Fix the stored values through the dashboard before upgrading: a refused start
may already have migrated the database.

Each start logs what that release does with the two settings in effect and with
the `searxng` instances that inherit them, including deleting stored engines that
neither a stored URL nor a stored instance uses. The warnings name settings and
instances, never a value.

Docker Compose sets `OTARI_WEB_SEARCH_URL` to the bundled SearXNG container
unless the variable is set, even to an empty value. To move off it, declare the
instance, with the engines `web_search_engines` names, else the four above:

```yaml
search_tools:
  local:
    provider: searxng
    api_base: "http://searxng:8080"
    options: {engines: "duckduckgo,mojeek,qwant,wikipedia"}
    price_per_call: 0   # USD; a SearXNG you run costs nothing
web_search_default_tool: local
```

The tool is priced on its own, not under `otari:web_search`: with
`require_pricing`, the default, a request that declares web search on a key that
counts toward budgets gets a 402 until the tool has a price, so set
`price_per_call`, or price the tool through `/api/v1/pricing` or the dashboard.
Each start writes `price_per_call` again, so keep the lasting rate there, or
leave it out to keep one set through the API (see
[Pricing a gateway-run tool](tools.md#pricing-a-gateway-run-tool)).

Leave `web_search_default_tool` out when the in-loop tool searches on
`web_search_provider`, so it keeps doing so. Then turn the compose default off in
`.env`, and remove `web_search_engines` wherever it is set:

```bash
OTARI_WEB_SEARCH_URL=
```

A variable exported in the shell that runs Docker Compose wins over `.env`, so
unset an exported `OTARI_WEB_SEARCH_URL` there too.

### The older in-loop backend

The older in-loop backend still serves some requests until it is removed: a
search on `web_search_provider`, or on a Tavily or Brave organization key, and a
request that declares only the fetch tool. `web_fetch_default_tool` governs every
fetch tool call, while `web_search_max_calls` and `fetch_tool` apply only where a
search instance serves the request: the older backend keeps its fixed limit of 10
calls, its 422 past it, and its own enrichment through the built-in fetcher.

## Decision providers

`decision_providers` configures the upstreams behind `POST /api/v1/decisions`. The
key is the prefix callers write in `model`, so the entry below serves
`typesafe:jev-latest`, `openrouter:typesafe/jev-1.13` and `local:openjev`:

```yaml
decision_providers:
  typesafe:
    api_key: ${TYPESAFE_API_KEY}
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
  local:
    provider: llamacpp
    api_base: "http://127.0.0.1:8080"
```

`provider` is one of `typesafe`, `openrouter` or `llamacpp`, and defaults to the
key. TypeSafe and OpenRouter need an `api_key`, and their default `api_base` can be
replaced with another https root. A `llamacpp` entry points at a `llama-server`
running a decision model and needs an `api_base`; it may use plain http only when
it has no `api_key`. `timeout` sets the seconds to wait for an answer (default 30).

These entries are separate from `providers` because none of these upstreams serves
chat: they never appear in `/api/v1/models` or provider health. Price a decision
model like any other, as `<provider>:<model>` in `pricing`.

### Agent model recommendations

For what `POST /api/v1/routing/recommend` does, see
[Agent model recommendations](routing.md#agent-model-recommendations). The
decision model it asks is set only on the
[Subagent router](routing.md#subagent-routers) page of the dashboard, never in
`config.yml`. One setting shapes the question:

```yaml
agent_recommender_candidates:
  haiku: Simple, well-specified, mechanical work. Little judgment needed.
  sonnet: Moderate reasoning across a few files, contained code changes.
  opus: Hard reasoning, ambiguous goals, large or security-sensitive changes.
```

`agent_recommender_candidates` maps each model name, as Claude Code spells it,
to what it is for; `null` is allowed when the name says enough. At least two
are required, and at most 255, the limit a choice question has. The defaults
are Claude Code's `haiku`, `sonnet` and `opus` aliases with descriptions along
the lines above.

The candidate list is Claude Code's alone: a pi request sends its own
`candidates` in the same shape, as Otari model ids (`<provider>:<model>`),
because [pi-otari](https://github.com/mozilla-ai/pi-otari) decides them on the
laptop.

## Mail

Mail is optional. An invitation that is not emailed returns its accept link, and an invitee new
to the deployment chooses a password on the page it opens, so members can join
a deployment with no transport configured. Without mail, signup, email verification, and password
reset are unavailable. A member who forgot their password gets back in through a password link an
operator creates from the dashboard's Accounts page and shares with them, or by signing in with a
provider where one is configured.

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
`POST /api/v1/settings/mail/test`.

### Invitations

An invitation link expires after `invitation_expiry_hours` (default `168`, seven
days). What an inviter and an invitee see is in
[Invitations](members.md#invitations).

Invitation email is limited over any 24 hours, because it carries text an
organization chose (its name, the inviter's name) from the deployment's own
sender, and on a deployment where anyone can create an organization that would
otherwise make it an open relay. Both limits are counted in `rate_limit_store`,
so with Redis they hold across replicas, and `null` turns one off:

| Setting | Limits | Default |
| --- | --- | --- |
| `invitation_mail_limit_per_organization` | Invitation emails one organization sends | 250 |
| `invitation_mail_limit_per_address` | Invitation emails one address receives, from every organization together | 5 |

An invitation over a limit is still created, and its link is returned to share
by hand, as when mail is off: the link costs the deployment nothing, and
refusing would stop a legitimate admin partway through a bulk invite.

A member added directly through `POST /api/v1/organizations/me/members` gets
no invitation link, so on a deployment with no mail and no provider sign-in a
new address waits for an operator's password link. Invite it instead, since the
invitation link can set its first password.

## Signup

`open_signup` (default `false`) decides what signup and a first Google or GitHub
sign-in do with an address nobody has added or invited. `true` gives each new
address an organization of its own. The signup form needs [mail](#mail)
configured either way, because it sends a verification link; a provider sign-in
does not. See [Signup](sign-in.md#signup).

## Built-in tools and guardrails variables

`GET /api/v1/tool-settings` shows effective sandbox, web-search, and guardrail
configuration. On the dashboard, the Tools pages show the sandbox and guardrail
settings, and the deployment's Web search page shows the web-search ones. Common startup settings are:

- `sandbox_provider` and `sandbox_url`
- `sandbox_container_idle_ttl_sec` and `sandbox_container_max_lifetime_sec`
- `code_execution_executor`
- `web_search_url` (deprecated, see [The legacy SearXNG settings](#the-legacy-searxng-settings))
- `web_search_provider` and `web_search_provider_api_key`
- `web_search_default_tool`, `web_fetch_default_tool` and `web_search_max_calls`
  (see [Search tools](#search-tools))
- `guardrails_url`
- `guardrail_thread_pool_size`
- `mcp_allow_loopback` and `mcp_allow_private_hosts`
- `web_search_allow_private_hosts`
- `provider_allow_private_hosts`
- `provider_endpoints_enabled` (see [Provider endpoints](provider-endpoints.md))
- `web_search_workspace_default` and `code_execution_workspace_default`
  (below)

See [Built-in tools](tools.md), [MCP](mcp.md), and
[Guardrails](guardrails-reference.md) for behavior and security boundaries.

### What a workspace with no tool policy gets

A deployment that serves several tenants from one search key or one sandbox can
make web access and code execution something each workspace turns on, rather than
something each workspace may turn off:

```yaml
web_search_workspace_default: false      # OTARI_WEB_SEARCH_WORKSPACE_DEFAULT
code_execution_workspace_default: false  # OTARI_CODE_EXECUTION_WORKSPACE_DEFAULT
```

Each decides only for a workspace with no stored policy: `false` refuses the
tool there as a stored disabled policy would, and a workspace owner or admin
turns it on from the Tools pages or the workspace policy API. A stored policy
always wins. Both default to `true`. See [Built-in tools](tools.md#per-workspace-web-access-policy) for what
each policy covers.

## Documentation links

By default the dashboard's Documentation link opens
[docs.otari.ai](https://docs.otari.ai/). Set `docs_url` or `OTARI_DOCS_URL` to
point it at another absolute HTTP or HTTPS URL.

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
Privacy row stays disabled. Wherever a page asks for the terms (signup, an
invitation, a first password, a provider sign-in), the box links the privacy
policy beside them when `privacy_url` is set too.
`GET /api/v1/bootstrap` publishes both addresses unauthenticated, so a URL
carrying a credential is refused at startup; the same check covers `docs_url`
and `site_url`.

## The public site

A deployment with a website of its own (a landing page beside the dashboard)
sets `site_url` or `OTARI_SITE_URL` to its absolute HTTP or HTTPS address. The
logo on the pages a visitor reaches without an account (the sign-in pages and
the public model catalog) then links there. Unset, it links to the catalog where
the deployment publishes one, and is not a link otherwise.

## The interface address

The gateway hands a browser absolute URLs in two places: the redirect that
finishes an OAuth sign-in, and the links in verification, password-reset and
invitation mail. Both point at `public_base_url`, which is right wherever this
process serves its own dashboard.

Where an edge serves the dashboard from another origin or path prefix, set
`ui_base_url` or `OTARI_UI_BASE_URL` to where a browser reaches it:

```yaml
public_base_url: "https://api.example.com"
ui_base_url: "https://app.example.com/dashboard"
```

Unset, `public_base_url` is used.

- Supply an absolute http(s) URL with no trailing slash.
- Credentials and fragments are refused, since the value goes into redirects
  and mail.
- A query string is kept and placed ahead of the hash route in every link
  (`https://app.example.com/dashboard/?edge=a#/verify-email?token=…`), for an
  edge that serves one interface for several deployments.

A dashboard served from a sibling host of this process may hold a session here
only when that host is listed in `cors_allow_origins`: the session cookie is
`SameSite=Strict`, and a same-site request from any origin not on that list is
refused.

Left unset on a split deployment, OAuth sign-in fails: the callback lands on an
origin with none of the sign-in state. Passkeys need their own settings there;
see [Set up passkeys](sign-in.md#set-up-passkeys).

## Extending Otari with a bootstrap module

`bootstrap` or `OTARI_BOOTSTRAP` names a trusted `module:callable` loaded
inside the gateway process. The callable can rebind extension ports and
contribute capability-gated routers. Most deployments should leave it unset.

This is executable code, not a feature flag. Install the module in the gateway
environment, pin it to a compatible Otari release, and authenticate every
contributed route. See [Architecture](../ARCHITECTURE.md) for the extension
boundary.

## Contacting the Otari team

When `feedback_enabled` is on, signed-in dashboard users can choose **Contact**,
beside Documentation in the top bar (in the account menu on a phone), to send a
message to the Otari team. It is off by default, so a self-hosted deployment
shows no Contact entry unless its operator turns it on. otari.ai turns it on.

What is sent:

- The message, and the signed-in account's own email if the team can reply to
  it (the form says so). It is never an address from the request, and is sent
  only if it matches the address the form showed; otherwise the page is asked
  to reload.
- No screenshot, page URL, deployment identifier or usage history.
- Nothing until the person sends: opening, typing and canceling make no
  outbound request.

On a self-hosted deployment, the gateway forwards the message and that address
to `https://api.otari.ai/api/v1/feedback/submissions`. It does not forward the
browser's cookies, authorization, referrer, or IP headers, or the account's
identifier. Network peers still see connection metadata, so this is private,
not anonymous. Keep request-body capture disabled for the feedback endpoint in
any additional logging or tracing you configure. On otari.ai, the message is
emailed to the support team with the account's email as the address to reply
to, and the team is notified in Slack with the message and the account's user
ID, never its email.

To turn Contact on, set `feedback_enabled: true` in YAML or
`OTARI_FEEDBACK_ENABLED=true`, then restart. Off, the endpoint is not mounted
and the Contact entry is hidden. This setting is visible in Settings but cannot
be changed there at runtime.

Each signed-in person (and the master key) can send five messages every ten
minutes; past that the gateway answers `429` with `Retry-After`.

Message text accepts up to 4,000 Unicode code points. The gateway waits up to
10 seconds for the receiver and does not retry automatically. An unconfirmed
submission stays in the form until the person retries or explicitly discards
it. Drafts are held only in memory and disappear on page reload. A manual retry
after an unconfirmed delivery can produce a duplicate.

## Upgrading

- **Price unit names.** A stored rate's `unit` is singular (`token`, `request`,
  `image`) since v0.38.0. Its migration renames every stored rate on the first
  start, and the old names are still read. Upgrade every instance from an
  earlier release together: an instance still running one reads a renamed rate
  as a unit it does not know, so it refuses a priced request on it under
  `require_pricing`.
- **Image rates.** An `image` rate is USD per million images. A rate written in
  `config.yml` as raw USD per image (for example `0.04`) bills a millionth of
  what it meant; multiply it by 1,000,000.
- **Declared budgets and keys.** The first start of a version that records a
  digest of each declared entry writes every entry once, as if each had changed.
- **`providers:` entries and the Deployment providers page.** See
  [Upgrading](hosted-providers.md#upgrading).
- **Contact is off by default.** A deployment that left `feedback_enabled`
  unset no longer shows the Feedback entry, now named Contact. Set
  `feedback_enabled: true` to keep offering it; see
  [Contacting the Otari team](#contacting-the-otari-team).
