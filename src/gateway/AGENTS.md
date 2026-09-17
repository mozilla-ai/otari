# Gateway internals (`src/gateway/`)

`src/gateway/CLAUDE.md` is a one-line `@AGENTS.md` import. Edit this file and
never replace or remove the import.

Before changing the backend, read
[backend-standards](../../.github/skills/backend-standards/SKILL.md). The root
[AGENTS.md](../../AGENTS.md) owns runtime modes, validation, and generated
artifacts. [ARCHITECTURE.md](../../ARCHITECTURE.md) owns the extension boundary.
[The modular monolith](../../ARCHITECTURE.md#the-modular-monolith) names the
target shape and its import rules,
[Layering](../../.github/skills/backend-standards/SKILL.md#layering) gives the
rules for each layer, and [docs/domains.md](../../docs/domains.md) maps every
module to its domain.

## Ports and composition

Domain protocols live in `ports/`, core implementations in `adapters/`, and
bindings in `container.py`. `OTARI_BOOTSTRAP=module:callable` may rebind a port
or contribute a capability-gated router after core bindings are installed.
Which mechanism new code uses is in
[Where new code goes](../../ARCHITECTURE.md#where-new-code-goes), and the steps
for an optional feature are in
[How to add a core feature](../../ARCHITECTURE.md#how-to-add-a-core-feature).

A route module that backs a dashboard page declares `SURFACE` beside its router
and adds it to `_DECLARED_SURFACES` in `api/routes/bootstrap.py`. A core feature
sets `surface` on its registry entry instead.

Add a port only when a real second implementation exists. Core never imports an
overlay. Dependencies request protocols from the container and never name an
adapter.

A port that writes within a request shares that request's `AsyncSession` and
does not commit. A hybrid-capable port may receive no session; a control-plane-only
port should use the ordinary database dependency instead.

## Request lifecycle

Completion requests follow this order:

1. Extract and verify the API key or master key in `api/deps.py`.
2. Resolve the billed user, workspace, and organization.
3. Compile local routing or resolve a hybrid attempt plan.
4. Resolve pricing and reserve applicable budgets.
5. Apply organization, policy, and request guardrails.
6. Resolve MCP and gateway-run tool policy.
7. Dispatch through any-llm.
8. Record usage and reconcile or refund reservations.

Chat, Messages, and Responses share the pipeline in
`api/routes/_pipeline.py`; hybrid attempt reporting lives in `_platform.py`,
and streaming settlement in `streaming.py`. Pass-through endpoints use
`run_passthrough`. Direct search has its own dispatch scaffold.

A new provider-calling scaffold must register with `track_request` so
`/api/v1/usage/in-flight` sees it. Removal belongs to `InFlightMiddleware`,
which wraps the complete ASGI response and therefore outlives a streaming route
handler.

The API is mounted once, under `API_ROOT`, in `api/main.py`. A route module
declares its resource and nothing above it, `APIRouter(prefix="/keys")`, and
never writes `/api/v1` or `/v1` in a prefix or a decorator. Code that must
build a path takes `API_ROOT`, `API_VERSION`, or `OTLP_ROOT` from
`core/config.py`; the root is never spelled. OTLP is a sibling namespace at
`OTLP_ROOT`, and the `/v1/{traces,logs,metrics}` tail under it belongs to
OTel, not to us.

Some strings look like paths and are not: the usage-log and in-flight labels
(`USAGE_ENDPOINT*` and the other `*_ENDPOINT` constants in `api/routes/`), the
metric `endpoint` label, and operation ids. They are identifiers, written to
rows or into generated clients, and keep their values when a route moves. A
docstring in `services/`, `models/`, `ports/`, or `repositories/` never names
the route that calls it. After adding or moving a route, run
`tests/integration/test_api_prefix_contract.py`; it also fails on a spelled
root anywhere under `src/gateway`.

## Authentication and authority

`verify_master_key` authenticates either a header master key or an active
dashboard session. It does not prove that the session may act deployment-wide.

- Deployment-wide management routers declare
  `require_deployment_operator` on the router, not per route, so a route added
  to one inherits it. Where such a router holds a route with a different rule
  (a catalog read, external usage ingestion), that route goes on a router of its
  own rather than overriding the gate in place.
- Tenant routes authenticate, resolve `CurrentIdentity`, and authorize the
  organization or workspace in `services/tenancy/authorization.py`.
- Data-plane routes use `verify_api_key_or_master_key`, which never accepts a
  dashboard cookie. The Playground's own completion endpoint
  (`routes/playground.py`) is the one surface that runs a completion from a
  session, and it is a separate route rather than a relaxation of that one: it
  resolves the caller's attribution user and proves their workspace membership
  itself, then hands the pipeline a `SessionPrincipal`
  (`types/session_principal.py`). Adding a second such route means doing all
  three, not reusing the type.
- Non-billable catalog reads use `verify_catalog_reader`.

Tenant lookups include the tenant predicate and return 404 for a foreign ID.
Client filters may narrow the server-derived scope and never widen it. A tenant
that needs a deployment-wide route gets a separately scoped endpoint, as
organization usage does for reads and `/api/v1/organizations/me/keys`
(`organization_keys.py`) does for writes; do not loosen the original route. A
member's key surface derives its owner as well as its scope, so it takes no
`user_id` and mints nothing budget-exempt.

The API key determines the billed workspace. Client `user`, workspace, or
organization fields cannot move billing or credential resolution.

## Pricing and money

`core/metered_pricing.py` is the only cost calculation. Settlement, estimates,
repricing, and imported usage all use it. Cache-token convention is explicit,
and a settled total is rounded once, half-up, to the micro-dollar.

Money columns use the decimal types in `models/money.py`. Widen incoming
floats with `to_usd` at service boundaries; never cast a decimal to float
inside database arithmetic. Convert to float only for wire responses and
metrics.

Price lookup order belongs to `services/pricing_service.py`. Do not duplicate
it in a route or tool backend.

## Budget enforcement

Per-user budgets and scoped budgets are both enforced. A shared `Budget` row
gives each attached user the full limit; it is not a pooled account.

A budget caps dollars, tokens and requests independently, and both mechanisms
hold and settle all three: a limit added to `budgets` needs a counter and a hold
on `users` and on `scoped_budgets`, or it binds through one reachability and is
silently ignored through the other.

`budget_service.reserve_budget` places the estimate before dispatch and the
shared settlement helpers reconcile or refund it. Scoped reservations use
conditional updates in one total order and compensate earlier holds when a
later ceiling refuses.

`budget_reservation_ledger.py` gives each request's holds one identity.
Settlement claims that row before changing counters, making duplicate release a
no-op. Write the ledger after the holds it records; a top-up grows the existing
row rather than creating a second independently expiring hold.

Every exit after reservation must settle or refund, including cancellation,
client disconnect, tool failure, and provider error. Preserve the sweeper and
opportunistic reclaim paths for abandoned holds.

## Routing

`services/routing/compiler.py` is pure and synchronous. It turns a policy and
request facts into a `CompiledPlan`. Keep database reads, embeddings, and
router backends outside it so CLI and API explain can compile without I/O.

Asynchronous router decisions live under `services/routing/` and pass a
`RouterOrdering` into the compiler. A declined decision uses the policy
default. The API attempt walker executes the compiled order and owns fallback
settlement.

## Tools, MCP, and guardrails

Only `otari_*` tool types run in the gateway; other declarations pass through
to the provider. The tool loop is in `services/mcp_loop.py`, sandbox and search
backends under `services/`, and outbound URL checks in
`services/url_safety.py`.

Deployment settings establish available backends. Workspace code-execution and
web-search policy can disable or narrow those settings but cannot widen them.
MCP servers are workspace resources rather than a refinement of a deployment
server list.

Organization guardrails add restrictions and have no workspace veto.
`prepare_gateway_tools` resolves workspace policy from the API key's context,
never a request header. Hybrid mode gets workspace policy from the control
plane.

A new path into an existing capability must honor the same veto. Direct search,
for example, enforces the workspace search disable switch.

## Provider and search credential stores

`provider_store_service.py` and `search_tool_store_service.py` overlay
encrypted stored rows on a preserved config baseline and refresh them across
workers. Stored values win over config values of the same name; config entries
remain read-only through the management API.

`secret_box.py` owns encryption through `OTARI_SECRET_KEY`. Public responses
return metadata such as `last4`, never plaintext credentials.

Provider resolution asks `ModelProviderPort` for a deployment-owned managed
credential only after local and tenant BYO sources fail. Managed credentials
must never move ahead of BYO or leave their trusted gateway.

## Mail

Features import `Mailer`, ask `can_send_links` before offering a mail-only
flow, and inspect the `MailDelivery` returned by `send`. They do not select
transports or catch transport exceptions.

Mail is optional. A flow with a manual fallback, such as invitations, continues
and returns the link. A flow with no fallback is hidden or refuses before doing
work.

The console transport deliberately logs token-bearing message bodies for local
testing. Do not widen this exception. Templates live under `templates/email/`
and use the shared renderer and header sanitization.

## Activation guide

`workspace_activation_service.py` derives activation from the first successful
usage row served by this deployment. Imported and absorbed rows do not count,
and neither does a Playground row: the guide marks somebody integrating Otari
from their own code, so it filters on the endpoint label as well as the source
(`core/usage_source.integration_traffic`).
The activation-state table stores only dismissal and setup-key state.

Key issuance requires workspace management authority and rotates the existing
setup key. `activation_guide` disables eligibility without unmounting the
endpoints.

## Data and migrations

Put a table in its domain's model module (`models/budgets.py`,
`models/tenancy.py`, and so on). `models/base.py` holds `Base` and the shared
column types and mixins. Tables use the declarative `Base`, except those whose
`Public` schemas are endpoint contracts, which use SQLModel (`models/tenancy.py`,
`models/provider_keys.py`, `models/playground.py`). A new table module must join
the import list in `models/__init__.py`, or Alembic proposes dropping its tables.

Two classes are named `User`: `models/users.py` is the billing identity that
keys, budgets, and usage attach to; `models/tenancy.py` is the dashboard sign-in
identity.

Request code gets a session through `get_db`; non-request code uses
`create_session()`; the usage-log writer uses `create_log_session()`, which
draws from a pool of its own so metering is not starved by request traffic.
Services own commits and rollbacks. The one exception is
`release_session(db)`, which the request path calls before dispatching upstream
so a pooled connection is not held across the provider call. Both describe the
code as it is today;
[Who commits](../../.github/skills/backend-standards/SKILL.md#who-commits)
gives the target shape, where a Unit of Work block commits and no route does.
Migrations live under `alembic/versions/`.

Once a client-side `db_command_timeout` is configured, a database call can
raise a bare `TimeoutError` as well as a `SQLAlchemyError`. Statement timeouts
are translated back to `SQLAlchemyError` per engine, but a *connect* timeout is
not, so handlers on the request path catch `DATABASE_ERRORS` from
`core/database.py` rather than `SQLAlchemyError` alone.

## Configuration

`GatewayConfig` loads the YAML file, structured environment config, scalar
`OTARI_<FIELD>` overrides, then database-backed runtime overrides. YAML
supports `${VAR}` interpolation. Service-level environment reads go through
`otari_env()`.

Validate a new security or routing setting at config load. Annotate every new
field with its settings view (`core/settings_view.py`): shown in a group,
omitted, or secret. The settings endpoint derives its view from that, and a
field without one fails at import.

A domain's settings live in `core/settings/<domain>.py`. `GatewayConfig`
inherits them rather than nesting them, because a nested model does not read
a flat `OTARI_<FIELD>` variable. A new setting goes in its domain's module
where one exists.

## Usage filters

Usage list, count, series, and bulk mutation must share filter semantics through
`core/sql.py` and the usage services. Every visible filter belongs on
`UsageSelection` with the same scalar-or-list shape and
`MAX_FILTER_VALUES` bound.

Bulk mutation re-derives rows from the submitted filters. It never trusts a
count calculated earlier by the dashboard. Imported-only guards do not replace
tenant or filter predicates.

One exception to the shared semantics: `GET /api/v1/usage/count` narrows
`counts_toward_budget=false` to imported rows, because it sizes a mutation
rather than a page. A count that sizes a mutation applies the mutation's fixed
scope and not only its filter set. Nothing else narrows it, and
`UsageEntry.bulk_editable` is how a client learns which rows that scope admits.

## Agent Gates

`agent_runtime/` evaluates a caller-submitted `.otari-gates.yml` policy
against caller-submitted evidence, served by the Hook Server
(`POST /api/v1/hooks/check`, `routes/hooks.py`). Everything under it is pure:
no filesystem, network, subprocess, or clock access. Otari never reads a
caller's repository itself. See [docs/agent-gates.md](../../docs/agent-gates.md).

## Logging

Use `gateway.log_config` with lazy `%s` formatting. Log opaque IDs, model,
provider, status, and counts when needed. Never log credentials or user content,
apart from the explicit local console-mail transport.
