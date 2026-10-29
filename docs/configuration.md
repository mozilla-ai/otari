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

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

String values support `${ENV_VAR}` interpolation. Keep credentials in the
environment or a secret store rather than committing them to YAML. A
reference to a variable that is not set fails the load. A variable that is
set but empty leaves a `providers` entry with no credential, and Otari ignores
such an entry with a warning that names the variable, so a credential stored
for the same provider through the dashboard serves its requests instead of
being shadowed by the empty one.

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
default_pricing: true
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
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
| `database_url` | SQLite or PostgreSQL connection. PostgreSQL is recommended for production. |
| `master_key` | Deployment-wide management credential. |
| `host`, `port` | Server bind address. |
| `auto_migrate` | Apply Alembic migrations at startup. |
| `require_pricing` | Reject unpriced, budgeted traffic. Defaults to `true`. |
| `default_pricing` | Use the models.dev price snapshot when no stored price exists. |
| `pricing_refresh` | Whether the catalog updates itself on a schedule, and whether its price changes wait for review: `manual`, `review`, or `auto`. |
| `feedback_enabled` | Allow deliberate feedback submissions to the Otari team. Defaults to `true`; startup setting. See [Product feedback](#product-feedback). |
| `health_checks_enabled` | Offer the Health page and its scheduled checks of the hosted providers' models. Defaults to `true`; startup setting. See [Health checks](health-checks.md). |
| `public_catalog` | Serve the model catalog to visitors without a session. Defaults to `false`. |
| `public_catalog_rate_limit_per_minute` | Anonymous catalog reads per client address per minute. Defaults to 60. |
| `rate_limit_rpm` | Per-user request limit. Unset disables it. |
| `rate_limit_store` | Where `rate_limit_rpm` and the `rate_limits` rules are counted: `memory` (the default) or `redis`. See [Rate limits across replicas](#rate-limits-across-replicas). |
| `rate_limit_redis_url` | The Redis that the `redis` store counts in. |
| `rate_limits` | Requests per minute, tokens per minute and requests in flight, per deployment, API key, user or model. Also managed from the dashboard. See [Rate limit rules](#rate-limit-rules). |
| `budgets` | Budgets declared by id and written to the database at every start. See [Budgets and API keys in config](#budgets-and-api-keys-in-config). |
| `api_keys` | API keys, service keys among them, declared with a secret you choose and written to the database at every start. See [Budgets and API keys in config](#budgets-and-api-keys-in-config). |
| `federation_issuers` | OpenID Connect issuers whose identity tokens may authenticate as an API key in place of its secret. Empty, the default, turns it off. Setting it requires `OTARI_SECRET_KEY`, or the gateway refuses to start. See [Workload identity](workload-identity.md). |
| `federation_member_rules` | Let an organization's owners and admins give their own API keys federation rules naming any public https issuer, matched only in the token exchange that names them. Defaults to `false`. Setting it requires `OTARI_SECRET_KEY`. See [Rules on a member's own key](workload-identity.md#rules-on-a-members-own-key). |
| `federation_allow_private_hosts` | Let a federation issuer's discovery document and signing keys be fetched from a private address. Defaults to `false`. |
| `federation_access_token_ttl_sec` | How long an access token from the workload identity token exchange lasts, in seconds. Defaults to 600; between 60 and 3600. See [Claude Code and the Anthropic SDKs](workload-identity.md#claude-code-and-the-anthropic-sdks). |
| `provider_max_retries` | How many times a provider SDK retries a failed call before Otari sees the failure. Unset keeps each SDK's default. See [Provider retries](#provider-retries). |
| `provider_supported_countries` | The countries each provider type serves, such as `openai: [US, DE]`. A request from another country skips that provider. Empty, the default, turns it off. See [Supported countries](#supported-countries). |
| `client_country_header` | A header a trusted proxy sets to the client's country code, such as `CF-IPCountry`. See [Supported countries](#supported-countries). |
| `client_country_database` | Path to a MaxMind DB country file the client address is looked up in. When set, a country file stored from the Settings page is not read. Needs the `geoip` extra. See [Supported countries](#supported-countries). |
| `streaming_keepalive_interval_ms` | Idle interval after which a streaming response sends a keepalive, so an intermediary with a read timeout does not cut a slow stream. Defaults to 15000; `0` disables keepalives. |
| `streaming_first_chunk_hold_ms` | How long the gateway holds a streaming response's status line for the provider's first chunk, so a provider failure before any output gets its own HTTP status, and falls over in a routing policy, rather than ending a `200` in an error event. A slower provider still streams. Defaults to 15000; `0` commits the response as soon as the stream opens. See [Error codes](api-reference.md#error-codes). |
| `idempotency_retention_sec` | How long a completion sent with an `Idempotency-Key` is kept for a retry to replay. Defaults to a day; `0` ignores the header. Needs `OTARI_SECRET_KEY`, which encrypts the stored response. See [Retrying safely](api-reference.md#retrying-safely). |
| `provider_request_timeout_seconds` | How long a non-streaming provider call may take, in seconds. Defaults to 600. Connecting stays capped at 5 seconds for providers whose SDK separates the two (Anthropic, OpenAI and OpenAI-compatible, Groq, Cerebras, Together); Bedrock, Gemini, Vertex AI, Mistral and Cohere apply the value as a whole. Without it, Anthropic's SDK refuses a non-streaming request with a large `max_tokens`. Providers with no per-request timeout, and a provider whose `client_args` sets its own `timeout`, keep their client's timeout. |
| `enable_metrics` | Serve Prometheus metrics at `/metrics`. Needs the `metrics` extra (`pip install gateway[metrics]`), which the Docker image installs; setting this without it refuses to start. |
| `trace_content_enabled` | Keep what an agent session's spans were about: prompts, commands, file paths. Defaults to `false`. See [Keeping what a span was about](traces.md#keeping-what-a-span-was-about). |
| `trace_platform_break_glass_enabled` | Let the deployment's operator open any organization's session content, recorded with a reason and a reference. Defaults to `false`. See [Who reads content](traces.md#who-reads-content). |
| `trace_content_key_backend` | What seals that content: `secret_key` (the default, `OTARI_SECRET_KEY`), `aws_kms`, or `aws_kms_only`, which opens nothing `OTARI_SECRET_KEY` sealed. See [Sealing content with AWS KMS](traces.md#sealing-content-with-aws-kms). |
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

The pool sizes bound concurrent *database* work, not concurrent provider calls:
a request hands its connection back before the upstream call. Raise them for a
deployment whose dashboard and management traffic are heavy, not because
inference is.

Recycling and the two statement timeouts matter most behind a managed database
or a NAT, which drop idle connections without closing them. The pool pings a
connection that has sat idle for more than five seconds before handing it out,
which would catch a closed one, but the ping is itself a statement and blocks
on a socket that went away silently, so leaving these unset turns a dropped
connection into a request that hangs for minutes.

`db_command_timeout` is enforced client-side and `db_statement_timeout_ms`
server-side, so the second one still ends a statement when the client is the
stuck half. Configure the server-side value above the client-side one, which
the defaults do and startup validation requires: set equal, whichever fires
first is a race.

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
without the extra, because quietly counting per process would multiply the
limit again.

Each request is checked and counted in one step, against Redis's own clock, so
replicas never both take the last slot and their clock skew does not matter.
If Redis cannot be reached, each replica counts on its own instead of refusing
traffic, and tries Redis again a few seconds later; the gateway log says when
that starts and stops.

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
  used once it completes, as LiteLLM does, so the last request admitted in a
  minute can take the count past the limit. A request that fails is charged the
  tokens its provider reported before failing, usually none, and a request
  refused after admission (by its budget, say) is charged nothing.
  `tpm_admission: estimate` counts the way providers count their own quotas
  instead: a request holds an estimate (its prompt plus `max_tokens`, or
  `budget_estimate_default_output_tokens` when it sets none) while it runs, and
  is refused when that does not fit. Use it for a limit meant to stay under a
  provider's quota, such as a `per: model` rule that should move traffic to the
  next model before the provider refuses it. Under it, a request whose estimate
  alone exceeds the limit is always refused, with no `Retry-After`, since
  waiting would not let it in.
- `max_concurrent`: requests in flight at once. A slot is given back when the
  response ends, streamed or not. `lease_sec` (15 minutes by default) bounds how
  long a slot outlives a process that dies holding it.

`per: key` does not limit a request made without an API key (the master key or
a dashboard session), and `per: user` does not limit one billed to no user. A request has to fit every rule that applies to it; one
that does not is refused with a 429 naming the rule and the limit it hit, counted by none of them,
and holds no budget. Rules count in `rate_limit_store`, so with Redis they hold
across replicas. They apply to chat completions, messages, responses and search, after
`rate_limit_rpm`.

`per: model` limits each model in `models`, written as `instance:model` (the
provider instance it is called through, then the model). One count is kept per
model and shared by every policy, alias and direct call that reaches it, so it
can stand for a provider's quota. It is checked when an attempt is about to call
the model, after any policy has picked its candidates, rather than at admission:
a [routing policy](routing.md#spill-over-when-a-model-is-full-priority-routing)
skips a full model and tries its next candidate, and a request is refused with a
429 only when no candidate has room. A direct call to a full model is refused.
A candidate that fails before responding gives back its tokens and its slot but
keeps its request counted, since the provider was sent it.
A search tool is limited the same way, named as `provider:tool` (`exa:exa-search`).

`per: budget` keeps one count for every user on the same budget, such as all of
a service key's end users on one feature's budget, so it limits a feature as a
whole where `per: user` limits each user in it. `budgets` narrows it to the
budgets listed by `budget_id` (as `GET /api/v1/budgets` returns it), each still
counted on its own; unset, it counts every budget. A user on no budget is not
counted by it, and neither is a request from a key with `exclude_from_budget`
or any request while `budget_strategy` is `disabled`. Like `per: deployment`, it is counted
after the rules that count the caller alone, so a request over its own limit
takes no place in a count other users share.

`mode: report` counts a rule the same way and refuses nothing: a request over it
is served, routing still picks the same candidate, and the response carries
`Otari-Rate-Limit-Exceeded` naming every report rule it was over (comma-separated).
The gateway also logs it and counts it in `gateway_rate_limit_reported_total`
by rule. A request something else refuses is not reported. A reported request is not counted in the window it was over, as a
refused one is not, so the count is the one the rule would keep if it were
enforced. Use it to watch a limit before enforcing it, or to see when traffic
borrows beyond a share without refusing anyone; `mode: enforce` is the default.

Rules can also be added, changed and removed from the dashboard (Settings, Rate
limit rules) or through `/api/v1/rate-limits`. A change applies at once on the
replica that served it and on every other replica within 30 seconds; a changed
rule keeps the requests it already counted. Choose "Each model" to add a
`per: model` rule and pick its models. `per: budget`, `budgets` and `mode` are
set through the API. The rules in config.yml are listed there read-only, and a stored rule cannot take the name of one. A stored rule
whose name config.yml later declares is skipped with a warning at startup and
left out of the list; `DELETE /api/v1/rate-limits/{name}` still removes it.

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
  end-user-budget-ai:
    max_budget: 0.1
    reset_alignment: calendar_day
    rpm_limit: 40
    tpm_limit: 2000
  mlpa-global:
    max_budget: 100
    reset_alignment: calendar_day

api_keys:
  mlpa:
    secret: ${MLPA_SERVICE_KEY}
    user_id: mlpa
    is_service_key: true
    end_user_budget_ids: [end-user-budget-ai]
    end_user_budget_id: end-user-budget-ai
    ceiling: mlpa-global
```

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
| `is_service_key` | Lets requests name end users. See [Service keys and end users](access-control.md#service-keys-and-end-users). |
| `end_user_budget_ids` | Budgets a request may start a new end user on with `Otari-End-User-Budget`. |
| `max_end_users` | Most live end users the key may have on the budgets it assigns, or on any budget when it assigns none. See [Capping a key's end users](access-control.md#capping-a-keys-end-users). |
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
  when its entry in config.yml differs from the one a start last wrote over it,
  so a change made through the API or the dashboard, such as lowering a budget
  or deactivating a key in an emergency, lasts across restarts and new replicas
  until config.yml changes that entry. Changing any field of the entry writes
  the whole entry back: a key that was deactivated is active again. Fields a
  key does not declare (expiry, allowed models, metadata) are left as they are.
- Changing a key's secret in config, including through the environment
  variable it reads, changes its secret at the next start, and the old one
  stops working.
- `end_user_budget_ids`, `end_user_budget_id` and `ceiling` are part of the
  key's entry and are reconciled with it. End users already created keep the budget they started on, as they
  do when the key is changed through the API. A ceiling moved to another budget
  restarts its period and keeps the spend it has recorded; a ceiling already on
  the declared budget keeps its period, so a restart does not reset it. A key
  with no `ceiling` keeps any ceiling it has, and an unchanged entry leaves a
  ceiling changed or removed through the API as it is.
- Otari keeps a digest of each entry it wrote beside the budget or key, never
  the secret itself (the secret enters as its hash), to tell whether the entry
  changed. The first start of a version that keeps it writes every entry once,
  as every start used to.
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
deployment that declares a key does not mint and print one.

## Provider configuration

> **Deprecated.** The `providers:` block is being retired in favor of hosted
> providers, which the Hosted providers page and `/api/v1/hosted-providers`
> manage. At startup, each entry a hosted provider can hold is copied onto a
> hosted provider of the same name, once: its `${VAR}` values are read at that
> start, and the hosted provider serves the name from then on, with a serving
> switch, a test, its models and removal like any other. The start asks the
> provider for its models and switches on every one it lists, priced or not,
> and a later refresh keeps them on; one that lists none is not copied, and keeps serving from `config.yml`. The start then logs
> that the entry is no longer read, so it can be taken out of `config.yml`;
> rotate its key on the Hosted providers page rather than in the environment.
> Removing the copied provider on the page sticks: the entry is not copied
> again. An entry a hosted provider cannot hold keeps serving from `config.yml`,
> and the start logs why: one that sets `models`, `max_retries`,
> `messages_api` or Vertex AI's `credentials`, `project` or `location`, a
> keyless Bedrock or SageMaker entry served on the server's own cloud role, a
> name that names another provider or that a hosted provider already holds, or
> a key with no `OTARI_SECRET_KEY` to encrypt it. Once copied, an organization's
> own key for the provider is asked first, where the entry was asked before it.
> Set `import_configured_providers: false` (`OTARI_IMPORT_CONFIGURED_PROVIDERS`)
> to keep serving every entry from `config.yml` for now.

The `providers` map is keyed by provider instance. Only a provider it declares
(or a hosted provider, or an organization's key) serves requests. A standard
provider needs only its credential, which the entry names:

```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

A keyless local backend or a provider authenticating from ambient cloud
credentials is declared with a bare entry:

```yaml
providers:
  ollama:
  bedrock:
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
variants), and it keeps a reasoning model's reasoning across turns: each
reasoning item comes back as a `thinking` block whose signature carries the
item's encrypted content, and when a client such as Claude Code sends that block
back on the next turn, Otari hands the reasoning back to the provider. The
reasoning is encrypted for the provider account that produced it, so the
signature also names the instance that served it: only that instance gets the
reasoning back, and any model it serves can use it. A thinking block signed this
way is dropped before a call to any other instance or provider, and another
provider's thinking blocks are dropped before a Responses call. When the block it
drops opened an assistant turn whose tool calls the request answers, `thinking`
is also left off that call, because Anthropic refuses a tool-use turn continued
under thinking without one; a route that falls back from such an instance to an
Anthropic model mid-loop therefore answers that one turn without extended
thinking.

Reasoning fields are sent only to a model whose name looks like a reasoning
model (`o1`, `o3`, `o4-mini` and the rest of the `o` series, `gpt-5*` except its
`-chat` variants, and anything with `codex` in it), whether or not the client
asked for thinking, because any other model fails the call over them. An Azure
OpenAI deployment is called by its deployment name, so name a reasoning
deployment after its model (`gpt-5-mini`, not `prod-reasoner`) or its reasoning
is neither requested nor carried across turns.

Every call is sent with `store: false`, so nothing is kept upstream. A Messages
`thinking` budget becomes a Responses `reasoning.effort`: up to 2,048 tokens is
`low`, up to 16,384 is `medium`, and more is `high`. An `output_config.effort`
takes precedence (`max` becomes `high`). Reasoning summaries are requested unless
thinking is disabled. `stop_sequences` is refused with a 400, because the
Responses API has no equivalent; `top_k`, `cache_control` and `metadata` are
dropped (OpenAI caches prompts on its own). When the caller sends no
`prompt_cache_key`, the request's session keys the provider's prompt cache, so the
turns of one conversation land on the same cache: `Otari-Conversation-Id`, or the
session id Claude Code sends on every request. The key is scoped to the caller
like any prompt cache key, so the session id itself never reaches the provider.
Anthropic's server tools are not
forwarded: Otari's own web search, web fetch and code execution serve them, as
for every non-Anthropic model.

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

Operators add provider credentials at runtime as hosted providers, on the
Hosted providers page or through `/api/v1/hosted-providers` (see
[Hosted providers in config.yml](#hosted-providers-in-configyml) for the
fields). Config-file entries remain read-only in the dashboard.

Stored credentials require `OTARI_SECRET_KEY`, a Fernet key generated with
`otari gen-secret-key`. To rotate it, configure the new key before the old key,
run the hosted provider and search-tool re-encryption endpoints
(`POST /api/v1/hosted-providers/reencrypt` and
`POST /api/v1/search-tools/reencrypt`), then remove the old key. A subagent
router's key has no re-encryption endpoint yet: re-enter it on the Subagent
router page after the rotation. Losing every configured encryption key makes
stored credentials unrecoverable.

#### Upgrading from the Deployment providers page

Earlier releases stored runtime providers on a Deployment providers page
(`/api/v1/provider-credentials`), which is gone. At startup, each provider it
stored becomes a hosted provider of the same name, served by the
implementation its `provider_type` names, so an `instance:model` selector,
routing policy or alias naming it keeps working. The start asks each provider
for its models, so the first start after the upgrade can take a little longer,
and offers every model it lists switched on, priced or not, as the stored
provider served it, and a later refresh keeps them on, on every path: completions, routing-policy fallbacks, the
pass-through endpoints and batches. A model it does not list is refused until
it is added on the Hosted providers page, or until "Serve models not on its
list" is turned on there.

A provider the move cannot take is logged at startup and listed on the Hosted
providers page with the reason (`GET /api/v1/hosted-providers/migration`), and
serves nothing until it is added again: a provider that lists no models, a
name a hosted provider already holds, a key no configured `OTARI_SECRET_KEY`
reads, an instance named after another provider (`groq` declared as
`openai`), or a keyless Bedrock or SageMaker provider served on the server's
own cloud role, which belongs under `providers:` in `config.yml` instead.

Two things behave differently once moved. A hosted provider is asked after an
organization's own key, where a stored provider was asked before it, so an
organization holding its own key is served on that key. And a `config.yml`
provider of the same name now serves ahead of it, where a stored provider
used to override the configured one.

### Hosted providers in config.yml

A hosted provider is a credential the deployment serves requests on when
nothing else credentials them (see [Hosted providers](hosted-providers.md#hosted-providers-and-byo-credentials)).
Besides the Hosted providers page, a deployment can declare
them in a `hosted_providers:` block, keyed by name, with the fields the
`/api/v1/hosted-providers` body takes:

```yaml
hosted_providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  openai-eu:
    provider: openai
    api_key: ${OPENAI_EU_API_KEY}
    api_base: https://eu.api.openai.com/v1
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    restricted_to_organization_id: 4f1c2a7e-0b9d-4c55-9a0e-2d6f8b3c1e90
  vllm:
    api_base: http://vllm.internal:8000/v1
    models: [qwen3-32b]
```

The name is what a request addresses (`openai-eu:gpt-4o`), and what the
provider's models, prices and usage are keyed on. It defaults to the
implementation; a name of its own lets two endpoints of one implementation be
served side by side. A name may not contain `:` or `/`, may not be `otari` or
`hosted`, and may not be another implementation's name.

| Field | Meaning |
|---|---|
| `provider` | The any-llm implementation serving it. Defaults to the name. |
| `api_key` | The key the provider is served on. Required unless the provider needs none (listed below) or has an `api_base` of its own. |
| `api_base` | An endpoint other than the provider's default. |
| `client_args` | What the provider's SDK client needs beyond the key, such as Bedrock's region. |
| `enabled` | `false` keeps the entry without serving it. Defaults to `true`. |
| `restricted_to_organization_id` | Serve one organization only, by its id. Checked at boot rather than at load, because the organizations are in the database: an id that names none is logged, and the provider then serves no organization. |
| `models` | Model ids a refresh offers when the provider cannot list its own. |
| `supported_countries` | The two-letter country codes whose requests it serves. Unset serves every country. See [Supported countries](#supported-countries). |

A key may be left out for the self-hosted backends `vllm`, `ollama`,
`llamacpp`, `llamafile`, `lmstudio`, `cascadia` and `otari`, for `vertexai` and
`vertexaianthropic`, which authenticate through the Google Cloud credentials in
the gateway's own environment, and for any provider given an `api_base`, which
is called with a placeholder key, as a self-hosted OpenAI-compatible server is.
Every other provider, Bedrock included, needs `api_key`. The Hosted providers
page and API follow the same rule. As for any hosted provider, a request is served on one
only when no organization key or `providers:` instance answers for it first.

Reference the key as `${VAR}` rather than writing it into the file: it is
resolved from the environment as every `${VAR}` in the file is, so the key stays
in your secret store. The whole block can instead be set as JSON through
`OTARI_HOSTED_PROVIDERS`, where `${VAR}` resolves the same way:

```bash
OTARI_HOSTED_PROVIDERS='{"openai": {"api_key": "${OPENAI_API_KEY}"}}'
```

When it is set, it replaces the `hosted_providers:` block of the file (and of
`OTARI_CONFIG_YAML`) whole: a provider only the file names is not served, and
the two are never merged per provider. An empty value leaves the file's block
in place.

The block is validated at load: an unknown implementation, a `provider_type`
key (the implementation goes in `provider`), a name a hosted provider cannot
take, an unknown field, or a missing `api_key` for a provider that needs one
fails the start. With
`OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false`, an `api_base` that resolves to a
private address fails it too, as it is refused for a provider stored through the
page; that check runs at boot, because it resolves the host. The block is never
shown by `/api/v1/settings`.

A configured entry is never written to the database or encrypted. The Hosted
providers page lists it with a `config.yml` badge and the API marks it
`source: "config"` with no `id`; changing or deleting it through the API answers
`409`, so change it in the configuration and restart. Its models, their serving
switches and their prices are managed on the page as any hosted provider's are.
An entry starts with no models and serves only the ones switched on under it,
so refresh its models or add them before it serves a request.
A provider stored through the page or the API under the same name wins over the
entry, and the start logs each entry it shadows; deleting the stored provider
serves the entry again on the same models. Taking an entry out of the
configuration leaves its models in the database, serving nothing and offered
in no catalog; they come back if the entry does, and the next catalog update run
with its cleanup on the Hosted providers page (`POST
/api/v1/hosted-providers/catalog-update` with `cleanup`) deletes them and their
prices, naming the provider in `orphaned_rosters`. Where no hosted provider
offers a model at all, the cleanup deletes those models and leaves every price
in place.

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

Every request a deployment sends to a provider reaches it under one account. A
provider that enforces its usage policy per end user needs an identifier for
each one, or a block meant for one person lands on the account, and every user
and workspace on the deployment loses that provider.

So Otari sends each provider call a digest of the workspace and the resolved
user, and of the user tag the caller sent Otari when there is one. On chat
completions and Responses the tag is `safety_identifier`, or `user` when the
request has none; on Messages it is `metadata.user_id`. A tag equal to the
resolved user counts as none. The table below names the field each provider
receives it in. The provider never sees the tag, which may be personal data, or
an Otari id. The same tag from the same user in the same workspace always gets
the same identifier, which is what a provider scopes enforcement by. Because the
resolved user is always in the digest, a tag only tells apart the end users
behind one user: tagging a request with another user's id never gives it that
user's identifier. The dashboard's Playground ignores a `user` or
`safety_identifier` sent in its request body, so the identifier its calls carry
is always the signed-in member's.

`user` has a second job: on a service key it names the billed end user, and on
any other key it must match the key's user unless `reject_user_mismatch` is
off. `safety_identifier` names nobody billed and nothing checks it, so one
request can bill one user and tag another. Send it when one key serves several
end users that `user` does not tell apart: a request with neither tag gets the
key holder's identifier, so a block meant for one lands on all of them. The
caller chooses the tag, so a caller that sends a new one on each request gets a
new identifier each time and no block lands on anyone. Otari cannot detect
that; the answer is to block that customer.

| Provider | Field |
| --- | --- |
| OpenAI, Azure OpenAI | `safety_identifier` |
| Cerebras, DeepSeek, Fireworks, Groq, OpenRouter, xAI | `user` on chat completions (DeepSeek's `user_id`) |
| Anthropic, including Azure and Vertex AI | `metadata.user_id` |
| Bedrock | `requestMetadata.otari_user`, except `metadata.user_id` on Messages to a Claude model that sets no `output_format` |
| Gemini API | the `safety_identifier` label, cut to the 63 characters a label value allows, on chat completions and Messages |
| Meta | `safety_identifier` on chat completions |
| Any other provider with a native Messages API, on Messages | `metadata.user_id` |

Bedrock's abuse detection reads no end-user identifier, but `requestMetadata`,
like the request body that carries `metadata.user_id`, lands in the account's
invocation logs, which is where the prompt ids in an AWS abuse report are traced
back to a user. Every other provider gets nothing: some have no such field
(Vertex AI, whose labels are for billing only, and Mistral), some run your own
models, and for the rest an unknown field can fail the call. A Responses or
Messages request served as a chat completion gets that provider's chat field. The
call that describes an image for a text-only model (`vision_describe_model`)
carries the request's identifier too, in the field the vision model's provider
reads.

Nothing turns the identifier off. OpenAI's partner program for gateways requires
one on every request, and a provider that cannot take one already gets nothing.

Set `OTARI_PROVIDER_USER_IDENTIFIER_KEY` to a random value of at least 32
characters (`openssl rand -base64 32`) to key the digest. Without it the digest
is unkeyed, so the same workspace id and user give the same identifier on any
deployment, and Otari logs a warning at startup. Otari refuses to start when the
key equals the master key or an `OTARI_SECRET_KEY` key. Rotating it gives every
user a new identifier: a block a provider placed on an old identifier stops
landing, and the provider's history for that user starts again. Rotate it only
when the key has leaked.

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

The rows hold the user behind the key, not the tag the caller sent. When the
identifier came from a tag, the customer that owns the workspace maps it to their
own end user. A row has no identifier where its request had none: usage imported
from outside, the pass-through endpoints (embeddings, images, audio,
moderations, batches), a request refused before Otari resolved its user, and
every row written before Otari stored it.

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
- A policy whose `knn` router embeds the prompt with a
  `router_embedding_model` on such a provider does not embed it: the router
  declines, and the policy serves its default target.
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

A hosted provider can also have a list of its own: its `supported_countries`,
set in the dashboard's Add and Edit dialogs, through `/api/v1/hosted-providers`,
or in its `hosted_providers:` entry. It is checked when a request would be
served on that provider's credential, so it never blocks an organization's own
key for the same provider. Both lists apply: with `openai: [US, DE]` above and
`supported_countries: [DE]` on a hosted `openai`, only Germany is served on that
hosted provider. A routing policy or a catalog id skips a hosted candidate its
list leaves out, and a plain model, a pass-through call or a batch call is
refused with a `403` and the `country_not_supported` code. Through the API,
null serves every country and an empty list none; in the dashboard a blank
field serves every country. A hosted provider's list needs one of the sources below too: without
one, every request served on it is from an unknown country and is refused.

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
  - Upload it on the dashboard's Settings page, in the Country file card, or
    with `PUT /api/v1/settings/country-database`. Otari stores it in its
    database, so every replica reads the same file and nothing is mounted.
    The replica that took the upload reads the file at once, and every other
    replica within 30 seconds. Only a country database is accepted, up to
    32 MiB. A rejected file leaves the stored one in place, and a stored file
    that a replica cannot open leaves that replica on the file it had. The card
    shows the stored file's type and build date, so a stale file is plain to
    see: DB-IP publishes a new one each month, and MaxMind twice a week.
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

Config-file prices seed the database. Prices stored through `/api/v1/pricing` take
precedence. Rates and settled costs use decimal arithmetic and costs are rounded
once to a micro-dollar. Use PostgreSQL for durable accounting.

### Default pricing

`default_pricing: true` prices from a models.dev snapshot when no stored price
exists. Explicit database or config pricing always wins. A snapshot ships inside
the package, so defaults work offline and a fresh deployment needs no network;
once an operator accepts a newer one (below), that one serves instead. The
`models_dev_metadata` setting only controls the descriptions and capabilities
shown beside a model: prices and context windows come from the snapshot whether
or not it is on.

A model is matched on its provider and id, including a pinned HuggingFace
backend and Bedrock geo-prefixed ids. A bare id that providers list at different
prices stays unpriced rather than guessing, so set a price for it or name the
provider.

Default pricing is off because provider catalogs and reseller rates change.
With `require_pricing: true`, a budgeted request with no effective price is
rejected instead of bypassing the budget. The 402 says what to do: with
`default_pricing` on, a model the bundled list does not price is most often a
misspelled name, so it points at `GET /api/v1/models` and at adding a price;
with it off, it offers both adding a price and turning the fallback on. A model
whose provider has no credential at all is refused as `provider_not_configured`
instead, since pricing it would not make it servable, and a selector nothing
resolves (no configured provider, alias, policy or catalog id) as `invalid_model`,
since there is no model to price.

With `require_pricing: false`, such a request is served, its model tokens carry
no cost, and its response carries no inline `cost_usd`. The usage row records no
cost unless the request also ran priced gateway tools, whose charges are still
recorded. The same setting governs an organization's own provider keys: a model
a key offers that nothing prices is switched on and served at no cost, where
`require_pricing: true` keeps it switched off until it has a rate (see
[Who is shown which models](models.md#who-is-shown-which-models)). The gateway logs a warning for each unpriced model at most once an
hour per process, and the dashboard shows operators a banner naming the models
that served unpriced traffic in the selected workspace in the last 24 hours
and still have no stored price, linked to those requests on the Overview, where
each can be priced.

### Keeping the defaults current

A catalog update does two things in order: it fetches the latest models.dev
prices and applies them, then it asks every enabled hosted provider what it
lists, offers what is new, and prices each model at the prices now in force. A
model that was waiting for a price is served once it has one. Running the two in
that order is what lets a model released after the bundled snapshot arrive
priced and serving in one update.

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
- `auto` runs it and applies changed prices at once, unless they look wrong: an
  update that would lose or remove more than a tenth of the priced models is
  held for review as under `review`.

`POST /api/v1/pricing/refresh/reject` discards held prices.

Each accepted snapshot is kept, and a request or report priced for a past
instant (`as_of`, usage backfills) uses the snapshot in force then, or the oldest
one for a time before the first. History therefore starts at the first accepted
snapshot; models.dev publishes no dated price changes of its own.

To refresh the bundled snapshot, as a maintainer does at release, run
`python scripts/update_models_dev_snapshot.py` (or pass `--input FILE` with a
saved `api.json`) and commit `src/gateway/data/models_dev_pricing.json`.

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

Each stored rate also carries its `unit` (`tokens`, `requests`, or `images`)
and its `origin` (`config` or `api`), so a rate the config file re-seeds on
every restart is distinguishable from one set in the dashboard.

### A public catalog

`public_catalog: true` serves `GET /api/v1/catalog/models` and the dashboard's
Models page to a visitor with no credential, so a deployment can show what it
serves before anyone signs up. A visitor sees the configured `providers:`
instances and any hosted models the deployment serves to every organization,
priced at the deployment's rates, and never a hosted provider restricted to one
organization, an organization's
override, key-scoped allow-list, or usage. A caller who sends a credential is
served as that caller, valid or not. The setting is off by default and can be changed at runtime.

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

The instance names `otari` and `hosted` are reserved for a managed platform's
own offerings and are refused in `providers:`.

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

### Per-request pricing (audio, moderations, rerank, and completion models)

A rate's `unit` decides what it bills. A rate in `tokens` bills the tokens the
provider reports, a rate in `images` bills each image returned, and a rate in
`requests` bills each request (or each search unit, for rerank). The rate reuses
`input_price_per_million`, always per million: USD per million tokens, requests
or images. A $0.04 image is `input_price_per_million: 40000`. A rate never bills
a count of another unit, so a per-request price is never read as a price per
image.

| Endpoint | What it reports | Rates that price it |
|---|---|---|
| Embeddings | Input tokens | `tokens`; `requests` bills one request |
| Rerank | Search units, and tokens where the provider reports them | `requests` per search unit, or `tokens` |
| Image generation | Images returned, and tokens for a token-billed model such as `gpt-image-1` | `images` per image, `requests` per request, or `tokens` |
| Audio transcription | One request, and tokens for a token-billed model such as `gpt-4o-transcribe` | `requests`, or `tokens` |
| Audio speech, moderations | One request | `requests` |

A rate in `tokens` on image generation, audio or moderations is still read in
that endpoint's old unit (raw USD per image, or USD per million requests), as
this guide used to document, and logs a warning once per model. A later release reads such a rate
strictly, as tokens, so set `unit: images` or `unit: requests` on those rates
now. A token-billed model (`gpt-image-1`, `gpt-4o-transcribe`) can then be priced
in `tokens`, because its tokens are recorded on the usage row.

A flat per-image or per-request rate cannot follow what the caller chooses: an
image's size and quality, or an audio file's length. Set it at the dearest
option you allow.

An unpriced request on audio or moderations is served at zero cost. Image
generation, embeddings, rerank and direct search are subject to
`require_pricing`, which also refuses a rate that cannot price the endpoint,
such as an `images` rate on embeddings, rather than serving it unpriced.

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
from every model price, so no provider entry under `providers:` is needed and a
model price can never be read as the tool's. The same row can be set through
`POST /api/v1/pricing` with `"kind": "search"` and the tool's name as
`model_key`, which is how a tool stored from the dashboard, with no entry here,
is priced; a `price_per_call` in the file adds a new version at startup only
when it differs from the rate in force, so a rate set through the API for a tool
that has a `price_per_call` lasts until the next restart, as a `pricing:` entry
without `effective_at` does for a model. A tool stored from the dashboard under
the same name and kind replaces this entry, so its `price_per_call` is not
written while the stored tool exists; a rate an earlier start wrote stays in
force until a newer one, such as one set from the dashboard, replaces it. A
start whose stored tools fail to load writes no `price_per_call` at all, and the
next start that loads them does. That rate is what a search reserves
against the caller's budget before it runs. A successful search settles at the
provider's reported charge when it reports one, and at the rate otherwise. A
search a workspace's organization search key served is priced under
`otari:web_search` instead, since the deployment cannot price the
organization's own provider account. With `require_pricing` on, a direct search
with no rate is refused with a 402 naming the price to add, even on a provider
that reports its charge, unless the key is exempt from budgets; a zero rate is
a valid price. With it off, such a search holds no dollars, though it still
holds one request against a request-count budget.

A fetch instance's `price_per_call` is stored the same way, as kind `fetch`.
The built-in fetcher cannot be declared, so its rate is set through the pricing
API, kind `fetch` and `model_key: builtin_fetch`. Either prices the fetch tool's
calls on that instance, ahead of `otari:web_fetch`.

A rate written before tools had a kind of their own, under `pricing:` or the
pricing API as `<provider>:<instance>` (`exa:exa-search`, `builtin:builtin_fetch`),
still prices its tool, behind the tool's own row, until a later release stops
reading it; startup names every such row and how to replace it. A key under a
tool's provider that names no configured tool is a model's price like any
other, never a tool's.

Rerank accepts either unit. A rerank model priced `unit: tokens` is charged
on the input tokens the provider reports. Providers that bill reranking per
search unit (one query over up to 100 documents, a long document counting as
several) report that count and no tokens, so give such a model `unit: requests`
and it is charged the reported units at `input_price_per_million / 1,000,000`
each, or one unit when the provider reports none. The budget reservation holds
the units the documents imply, from their count and length, and settles at the
reported count. The default-pricing dataset carries no rerank rates, so a rerank
model needs its own pricing row, for example `cohere:rerank-v3.5` at
`input_price_per_million: 2000` (USD 2 per 1,000 searches) with `unit: requests`.

A model served over chat completions, the Responses API, or Messages can be
priced the same way, for an upstream that bills per call and reports little or
no token usage, such as an answer endpoint served through an OpenAI-compatible
provider instance named `exa`. Give its pricing a `price_per_request` in USD:

```yaml
pricing:
  exa:exa:
    price_per_request: 0.005
```

`price_per_request` is the readable form of `unit: requests`: Otari stores it as
`unit: requests` with `input_price_per_million` a million times the price
(`5000` here), and `GET /api/v1/pricing` reports both. `POST /api/v1/pricing`
takes it too. It replaces the token prices rather than adding to them, so it
is refused beside an `input_price_per_million` that differs, an
`output_price_per_million`, a cache price, `pricing_tiers` or another `unit`.
Writing the rate in full as `unit: requests` still works.

Such a model is charged its price per request for each
successful call, streamed or not, whatever tokens it reports. The token counts
are still recorded on the usage row, and still count toward a token budget, but
they are not priced. A failed call, including a stream that errors or that the
client abandons, costs nothing. The budget reservation holds that flat amount
rather than a token estimate. A stream that ends without usage data is charged
the flat amount too, so `stream_missing_usage_policy` does not apply to it.
`require_pricing` treats the model like any other completion model, and the
default-pricing fallback never supplies a per-request rate, so a per-request
model needs its own pricing row.

### Per-image pricing (image generation)

An `images` rate is USD per million images, like a request rate: $0.04 an image
is `input_price_per_million: 40000`, which is what the dashboard stores when you
enter $40 per 1,000 images. Before this release an `images` rate was read as
raw USD per image, so a rate written that way in `config.yml` (for example
`0.04`) now bills a millionth of what it meant; multiply it by 1,000,000.

Image generation reserves the requested image count before dispatch at an
`images` rate, then settles on the images the provider returned. Audio holds
nothing before dispatch, so a priced audio request can exceed a budget by one
request's price.

## Search tools

`search_tools` names the deployment's search instances. Direct `POST /api/v1/search`
calls use them by name, or the default one (below) when they name none, and the
in-loop `otari_web_search` tool uses the default one. `fetch_tools` names fetch instances, for the `otari_web_fetch` tool
and for enriching search results. Both can also be stored at runtime through
`/api/v1/search-tools`, a fetch instance with `kind: fetch` and listed with
`?kind=fetch`, or from the dashboard's deployment Web search page, which also
sets each one's price and the defaults. A stored instance wins over
a configuration file entry of the same name and kind.

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
  enriches its results; without one, `web_fetch_default_tool` does.
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

The older in-loop backend still serves some requests until it is removed: a
search on the legacy settings, or on a Tavily or Brave organization key, and a
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

`POST /api/v1/routing/recommend` asks a decision model which of a few
candidate models a coding agent's subagent should run on (see
[Use with Claude Code](use-with-claude-code.md#let-otari-choose-a-subagents-model)).
Two settings shape the question:

```yaml
agent_recommender_model: typesafe:jev-latest
agent_recommender_candidates:
  haiku: Simple, well-specified, mechanical work. Little judgment needed.
  sonnet: Moderate reasoning across a few files, contained code changes.
  opus: Hard reasoning, ambiguous goals, large or security-sensitive changes.
```

`agent_recommender_model` is the fallback: a router stored on the
[Subagent router](routing.md#subagent-routers) page answers in its place, at
the price stored with it, and with routers stored and all of them switched off
nothing answers at all. It is a `decision_providers` selector, resolved per
request, so a deployment that never asks need not configure the provider.
`agent_recommender_candidates` maps each model name, as Claude Code spells it,
to what it is for; `null` is allowed when the name says enough. At least two
are required, and at most 255, the limit a choice question has. The defaults
are Claude Code's `haiku`, `sonnet` and `opus` aliases with descriptions along
the lines above. The list is Claude Code's alone: a pi request sends its own
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

Invitation email is limited per organization and per recipient address over 24
hours, so an organization anyone can create cannot use the deployment's sender to
mail strangers at will. An invitation over a limit is still created and its link
returned to share by hand. Both limits are counted in `rate_limit_store`, and
`null` turns one off:

```yaml
invitation_mail_limit_per_organization: 250   # default
invitation_mail_limit_per_address: 5          # default, from every organization together
```

See [Invitations](access-control.md#invitations).

`mail_transport: console` writes complete messages to logs for local testing.
Those messages can contain invitation or password-reset tokens, so never use it
where logs are shared. Test delivery from Settings or
`POST /api/v1/settings/mail/test`.

## Signup

Signup is closed by default: only an address an owner or admin already added or invited can get an account, by setting a password or by signing in with Google or GitHub. Open it where the deployment serves many tenants and each new address should arrive with an organization of its own:

```yaml
open_signup: true
```

The signup form needs mail configured in either case, because it sends a verification link. A Google or GitHub sign-in needs no mail, because the provider has already verified the address. See [Access control](access-control.md) for what each value does with an address nobody has added.

## Built-in tools and guardrails variables

`GET /api/v1/tool-settings` shows effective sandbox, web-search, and guardrail
configuration. On the dashboard, the Tools pages show the sandbox and guardrail
settings, and the deployment's Web search page shows the web-search ones. Common startup settings are:

- `sandbox_provider` and `sandbox_url`
- `sandbox_container_idle_ttl_sec` and `sandbox_container_max_lifetime_sec`
- `code_execution_executor`
- `web_search_url`
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
[Guardrails](guardrails.md) for behavior and security boundaries.

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
Privacy row stays disabled. A deployment whose dashboard sits beside a site that
owns the documents points at that site. `GET /api/v1/bootstrap` publishes both
addresses unauthenticated, so a credential in either is refused at startup. The
same check covers `docs_url` and
`site_url`.

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

Unset, `public_base_url` answers for it. Supply an absolute http(s) URL with no
trailing slash; a relative one would survive the redirect and mean nothing in an
inbox. Credentials and fragments are refused: this value travels in a redirect
and into outgoing mail. A query string is kept and placed ahead of the hash
route in every link (`https://app.example.com/dashboard/?edge=a#/verify-email?token=…`),
for an edge that serves one interface for several deployments and needs each
link to say which one built it.

A dashboard served from a sibling host of this process may hold a session here
only when that host is listed in `cors_allow_origins`: the session cookie is
`SameSite=Strict`, and a same-site request from any origin not on that list is
refused.

Left unset on a split deployment, an OAuth callback lands the browser on an
origin holding none of the sign-in state it started with, and the sign-in fails
with the authorization code unspent. Passkeys need their own settings there; see
[Access control](access-control.md#passkeys).

## Extending Otari with a bootstrap module

`bootstrap` or `OTARI_BOOTSTRAP` names a trusted `module:callable` loaded
inside the gateway process. The callable can rebind extension ports and
contribute capability-gated routers. Most deployments should leave it unset.

This is executable code, not a feature flag. Install the module in the gateway
environment, pin it to a compatible Otari release, and authenticate every
contributed route. See [Architecture](../ARCHITECTURE.md) for the extension
boundary.

## Product feedback

Signed-in dashboard users can choose **Feedback**, beside Documentation in the
top bar (in the account menu on a phone), to send a message to the Otari team.
The team receives it privately in Slack. Only the message is sent: no email,
screenshot, page URL, account identifier, deployment identifier, or usage
history is attached. Opening the form, typing, and canceling make no outbound
request.

The gateway forwards the message to
`https://api.otari.ai/api/v1/feedback/submissions`. It does not forward the
browser's cookies, authorization, referrer, or IP headers. Network peers still
see connection metadata, so this is private feedback, not anonymous feedback.
Keep request-body capture disabled for the feedback endpoint in any additional
logging or tracing you configure.

Feedback is on by default. To turn it off, set `feedback_enabled: false` in
YAML or `OTARI_FEEDBACK_ENABLED=false`, then restart. Off, the endpoint is not
mounted and the Feedback entry is hidden. This setting is visible in Settings but
cannot be changed there at runtime.

Each signed-in person (and the master key) can send five messages every ten
minutes; past that the gateway answers `429` with `Retry-After`.

Feedback text accepts up to 4,000 Unicode code points. The gateway waits up to
10 seconds for the receiver and does not retry automatically. An unconfirmed
submission stays in the form until the person retries or explicitly discards
it. Drafts are held only in memory and disappear on page reload. A manual retry
after an unconfirmed delivery can produce a duplicate.
