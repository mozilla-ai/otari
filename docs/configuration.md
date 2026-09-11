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
or a NAT, which drop idle connections without closing them. The pool's pre-ping
would catch a closed connection, but the ping is itself a statement and blocks
on a socket that went away silently, so leaving these unset turns a dropped
connection into a request that hangs for minutes.

`db_command_timeout` is enforced client-side and `db_statement_timeout_ms`
server-side, so the second one still ends a statement when the client is the
stuck half. Configure the server-side value above the client-side one, which
the defaults do and startup validation requires: set equal, whichever fires
first is a race.

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
or `/api/v1/provider-credentials`. Stored entries override config-file entries with
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

Config-file prices seed the database. Prices stored through `/api/v1/pricing` take
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

`search_tools` configures direct `POST /api/v1/search` calls. The same entries can
be managed at runtime from Tools or `/api/v1/search-tools`.

```yaml
search_tools:
  local:
    provider: searxng
    api_base: "http://searxng:8080"
```

`GET /api/v1/search-tools/providers` publishes the supported providers and whether
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
`POST /api/v1/settings/mail/test`.

## Built-in tools and guardrails variables

The Tools pages and `GET /api/v1/tool-settings` show effective sandbox, web-search,
and guardrail configuration. Common startup settings are:

- `sandbox_url`
- `web_search_url`
- `web_search_provider` and `web_search_provider_api_key`
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
owns the documents points at that site. `GET /api/v1/bootstrap` publishes both
addresses unauthenticated, so a credential in either is refused at startup, the
way `data_plane_url` refuses one. The same check covers `docs_url`.

## The data-plane address

A hosted control plane does not serve inference. Set `data_plane_url` or
`OTARI_DATA_PLANE_URL` to the gateway origin used by client snippets:

```yaml
data_plane_url: "https://gateway.example.com"
```

Supply the origin or path prefix without a trailing slash or `/api/v1`. Credentials,
query strings, and fragments are refused because `GET /api/v1/bootstrap` publishes
this value without authentication.

## otari.ai variables

Hybrid mode requires `OTARI_AI_TOKEN`. Optional platform settings control the
platform API URL, management URL, health-probe path, resolution timeout,
usage-report timeout and retries, and first-chunk fallback timeout. See
[Modes](modes.md) and the normative
[hybrid-mode protocol](hybrid-mode-protocol.md).

## Extending Otari with a bootstrap module

`bootstrap` or `OTARI_BOOTSTRAP` names a trusted `module:callable` loaded
inside the gateway process. The callable can rebind extension ports,
contribute routers, contribute background tasks, and contribute an Alembic
migration chain for tables of its own. Most deployments should leave it unset.

This is executable code, not a feature flag. Install the module in the gateway
environment, pin it to a compatible Otari release, and authenticate every
contributed route. See [Architecture](../ARCHITECTURE.md) for the extension
boundary.

### Budget alerts, the plugin this repository ships

`otari_alerts` is a worked example of all three contributions below, and a
feature in its own right: it tells an organization that one of its budget
ceilings is running out, before the ceiling starts refusing requests. A rule is
an [Apprise](https://github.com/caronc/apprise) URL (so Slack, Discord,
PagerDuty, mail and a plain webhook are one field) plus the thresholds that
reach it.

```bash
pip install 'gateway[alerts]'
export OTARI_BOOTSTRAP=otari_alerts:register
```

That is the whole of the setup. The rules are managed at
`/api/v1/organizations/me/alert-rules`, by an organization's owners and admins.
Two environment variables of its own:

| Variable | Default | What it does |
| --- | --- | --- |
| `OTARI_ALERT_EVALUATION_INTERVAL_SEC` | `60` | How often the ceilings are checked. `0` keeps the rules and sends nothing. |
| `OTARI_ALERT_ALLOW_PRIVATE_HOSTS` | `false` | Allow a destination that resolves to a private, loopback or reserved address. Turn it on to alert an internal chat server or webhook receiver. |

Apprise is an extra rather than a core dependency, so a gateway that does not
run the plugin does not carry it. There is no dashboard page yet: the surface
is the API.

### Contributing a router

A router contribution carries a `capability` naming the licensing axis its
surface sits on. Given a name, Otari mounts the router behind that gate and a
deployment not entitled to the capability gets the same 404 a path nothing
serves gets. Use it for a surface an overlay licenses per deployment.

Leaving `capability` as `None` mounts the router with no entitlement
dependency. That is the right answer for a contribution that is simply present
once the module is installed, which is what a plugin is: there is no licensing
decision to make, and inventing a capability name only to satisfy the gate
would invent one. Entitlement is not authentication either way, so each
contributed route still declares the credential it needs, the way Otari's own
routes do.

### Contributing a background task

A background task is a coroutine function that receives the gateway config.
Otari starts it beside its own periodic refreshers, in every mode, and cancels
it at shutdown under the same bounded wait, so a task that never yields cannot
hold the process open:

```python
import asyncio

from gateway.container import BackgroundTaskContribution, Container


async def run_alert_evaluator(config) -> None:
    while True:
        ...
        await asyncio.sleep(60)


def register(container: Container) -> None:
    container.contribute_background_task(BackgroundTaskContribution(name="budget alerts", start=run_alert_evaluator))
```

Names are unique per container; a second task registered under a taken name
fails startup.

### Contributing a migration chain

A module that owns tables records its own Alembic script directory and its own
version table on the container:

```python
from gateway.container import Container, MigrationContribution


def register(container: Container) -> None:
    container.contribute_migrations(
        MigrationContribution(
            name="alerts",
            script_location="/opt/alerts/alembic",
            version_table="alerts_alembic_version",
        )
    )
```

When `auto_migrate` is on (the default), startup upgrades Otari's own chain to
`head` first and then each contributed chain, on the same database URL. Each
chain stamps only the version table it named, so the histories never share a
row. `alembic_version` is Otari's and is refused, as is a version table or a
name another contribution already claimed; the refusal happens while the
container is built, before anything touches the database.

Otari hands the contributed `env.py` two things on the Alembic config. The
database URL travels on two channels, `sqlalchemy.url` and
`config.attributes["database_url"]`, and a contributed chain should prefer the
attribute: `sqlalchemy.url` is read back through configparser, whose
interpolation treats a percent sign as a token, so a password containing one
breaks it. The declared version table travels as
`config.attributes["version_table"]`.

```python
from alembic import context

config = context.config
database_url = config.attributes["database_url"]
version_table = config.attributes.get("version_table", "alerts_alembic_version")

# ... build the engine from database_url ...
context.configure(connection=connection, target_metadata=metadata, version_table=version_table)
```

Reading `version_table` from the attributes is offered, not required: a chain
may equally hardcode a constant of its own. What Otari requires is that the
`version_table` declared on the `MigrationContribution` is the table the chain
actually stamps. Otari uses the declared value for one thing only, refusing a
collision with core's `alembic_version` and with another contribution, so a
chain that declares one table and stamps another defeats that check.

Two cautions. A contributed chain must not reference a core table by foreign
key in a way that would block a core migration: the core chain runs first and
knows nothing about contributed tables, so a core revision that drops or
rebuilds a table the contribution points at fails on a constraint the core
chain did not create. Prefer plain indexed id columns over enforced foreign
keys into core tables. And `otari migrate` runs the core chain only; a deployment that
migrates with the CLI instead of on startup has to run each contributed chain
itself for now. Hybrid mode skips database initialization entirely, contributed
chains included, since it has no local database.

Contributed chains run under Otari's existing `auto_migrate` gate and get no
knob of their own. Setting `auto_migrate` to true is already a deployment's
explicit acceptance of boot-time DDL, and registering a bootstrap that
contributes a chain is a second explicit choice, so a third switch would only
add a way for a deployment to be half configured. A deployment that does not
want DDL at boot turns `auto_migrate` off and migrates out of band, which holds
for contributed chains exactly as it does for Otari's own.
