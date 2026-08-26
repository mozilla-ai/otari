# Gateway internals (`src/gateway/`)

`src/gateway/CLAUDE.md` is a one-line `@AGENTS.md` import of this file. Always edit `AGENTS.md` directly; never modify `CLAUDE.md`, and do not delete it: Claude Code discovers `CLAUDE.md`, not `AGENTS.md`, so the import is what loads this file.

Loaded when working under `src/gateway/`. For house style (async SQLAlchemy 2.0,
layering, the budget/reservation lifecycle, migrations, config/logging), read
[../../.github/skills/backend-standards/SKILL.md](../../.github/skills/backend-standards/SKILL.md).

The runtime modes and the OSS/enterprise seam are in the root `AGENTS.md`;
read that first if a change touches mode selection.

## Ports, container, bootstrap
`ports/` holds the domain-named `Protocol` interfaces the core depends on, `adapters/` holds Otari's own implementation of each, and `container.py` is the composition root that binds them, built once per app in `create_app` and read through the port dependencies in `api/deps.py`. `OTARI_BOOTSTRAP=module:callable` points at an overlay's register function, imported after the defaults are bound; unset, nothing is imported. Why the seam is shaped this way, and the rules for keeping it, are in [../../ARCHITECTURE.md](../../ARCHITECTURE.md); `scripts/check_architecture.py` enforces them.

Two local consequences worth knowing before you use it. A port factory receives `AsyncSession | None`, because hybrid mode runs with no local database and an adapter resolved on a hybrid request has to say what it does without one. A port whose every surface is standalone-only does not need that shape and should not take it: `get_telemetry_storage_port` and `get_identity_provider_port` both name `get_db` rather than `PortSessionDep`, which hands the adapter the caller's session instead of opening a second one against the same database for the same request. And a capability earns its port when a second implementation is real, not before: of the six that exist, three have core callers (`TelemetryStoragePort`, `ModelProviderPort` at the last rung of the credential ladder below, and `IdentityProviderPort`), and the other three are still mechanism alone.

`IdentityProviderPort` is the worked example of what a port is *for*. `POST /v1/auth/oauth/{provider}/callback` (`api/routes/auth_oauth.py`) proves the person holds a Google or GitHub account, and then asks the port who that makes them *here*. The two halves are split on purpose: the proving is protocol work that does not vary by edition, so it stays a plain service (`services/oauth_service.py`, on apron-auth), and only the policy is inverted. `RosterIdentityProviderAdapter` is that policy for the base build: sign in as an account an operator already added, refuse one nobody did, and never provision. An overlay that provisions on first sight binds its own adapter and edits nothing here.

Its adapter writes and does not commit, which is the wiring detail to copy rather than rediscover: the route and the port have to share one session, or the adapter's writes land in a transaction nobody commits. `auth_oauth.py` and `get_identity_provider_port` both name `get_db`, per the standalone-only rule above, which is what makes the provider link, the verification stamp and the session row one transaction.

## Request lifecycle (chat completions)
Read these together before changing request behavior, the flow spans several files.

1. App + middleware: `src/gateway/main.py` builds the FastAPI app, adds CORS + a security-headers middleware, and enforces auth on every path except `_PUBLIC_PREFIXES` (`/health`).
2. Auth: `src/gateway/api/deps.py` extracts the key from `Otari-Key` (canonical `API_KEY_HEADER` in `core/config.py`) or `Authorization: Bearer`; validates the SHA-256 hash against the `api_keys` table, or matches the master key.
3. Route handler: `src/gateway/api/routes/chat.py` resolves the billed user, runs budget checks (standalone) or resolves platform credentials, applies input guardrails, and extracts gateway-managed tools.
4. Dispatch: the provider/model is split with `AnyLLM.split_model_provider(...)` and the call is made via `acompletion(...)` from `any_llm`. Hybrid mode walks multiple resolved attempts with fallback (`src/gateway/api/routes/_platform.py`, streaming in `src/gateway/streaming.py`).
5. Usage + budget reconciliation: standalone writes a `UsageLog` row via the log writer and reconciles spend; platform reports usage upstream.

A usage row therefore only exists once a request has settled. What is *currently*
running lives in `src/gateway/inflight.py`: an in-memory, per-process registry,
populated once the budget, access and model-resolution gates have passed and the
provider is about to be called (so a request refused by one of those never
appears; a later guardrail or tool-declaration refusal does appear while that
check runs) and emptied by `InFlightMiddleware`, read by
`GET /v1/usage/in-flight` for the dashboard's Activity page. There are three
registration points, one per dispatch scaffold: `resolve_request_context`
(chat/messages/responses), `run_passthrough` (embeddings, images, audio, rerank,
moderations), and `_dispatch_search`. A new provider-calling scaffold needs its
own `track_request` call or it is invisible to the panel. Removal belongs to the
middleware and not to any
settlement path: a streaming response outlives its route handler, and the
`finally` that wraps the whole ASGI call is the only place that runs exactly once
per request (the same reason `gateway_active_requests` is instrumented there).

## Cost math
`src/gateway/core/metered_pricing.py` is the only place a cost is derived from a rate: settlement, the reserve-time estimate, repricing, and imported usage all go through it. It is `Decimal` throughout, takes no database, and reads a pricing object structurally, so a stored `ModelPricing`, an organization override, and a genai-prices default are all priced by one implementation. Two rules it enforces rather than assumes: **which cached-token convention a caller speaks is an argument with no default** (`cache_tokens_included`; `GatewayUsage.cache_tokens_in_prompt` is where the request path gets it), and **rounding happens once**, half-up, to the micro-dollar, at the point an amount becomes a settled total. Settlement and the external-usage ingest also record which convention a row's counts arrived under, in `usage_logs.cache_tokens_in_prompt`; it is nullable and nothing backfills it, so a row written before the column, or by a path that does not record one, reads NULL and repricing recovers the convention from `billing_meters` instead (`usage_admin_service._row_cache_tokens_included`). The pricing *lookup* chain, which is a different concern, stays in `services/pricing_service.py`.

Money columns are the exact types in `models/money.py`, not floats, and that now covers the whole path: the `model_pricing` and `organization_model_pricing` rates, `usage_logs.cost`, and (since mozilla-ai/otari#691) every budget counter, `users.spend` / `reserved`, `budgets.max_budget`, `scoped_budgets.max_budget` / `current_spend` / `reserved_spend`, `workspace_budget_defaults.max_budget`, and `budget_reset_logs.previous_spend`. So a settled cost reaches the counter a 403 is decided against unchanged, and a user's spend is the sum of the rows that produced it rather than an approximation of it.

**Nothing narrows on the way in, and the widening is spelled once per entry point.** `budget_service.reserve_budget` and `reconcile_reservation` take `Decimal | float` and widen through `models/money.to_usd`, because a caller may still hold a float (an imported amount, a flat dollar rate a route wrote as a literal). Do not push that conversion outward and do not add a `float()` inward: in PostgreSQL a float added to a `NUMERIC` column resolves the whole expression as double precision, which puts the drift back. The same rule is why `ZERO` (a module-level `Decimal`) is what the CASE clamps and the `.values()` zeroes are written with, in both `budget_service` and `scoped_budget_service`.

**The reserve-time estimate is `Decimal` too, and that was a decision rather than a consequence.** It is an upper bound reconciled against an exact settlement, so it *could* have stayed float. It did not, because on the stream-without-usage path the estimate is what settles the row (`cost_override=reservation.estimate` in `_pipeline.py`), which makes it accounting after all, and because a hold released at the amount it was taken needs both sides written the same way.

Amounts narrow in exactly one direction: **on the way out**, for a response body or a metric, through `models/money.as_float` or a bare `float()`. The wire contract and the generated dashboard client are float and stay that way, so `budgets.py`, `scoped_budgets.py`, `users.py` and `workspace_budget_default_service.py` each convert at the response boundary, and `get_budget_state` converts for the routing compiler, which is float throughout. Those are display conversions; none of them feeds arithmetic that settles anything.

## Budget enforcement
Two mechanisms, both enforced, neither replacing the other.

`src/gateway/services/budget_service.py` reserves an estimated cost before the call and reconciles/refunds after. Strategy is selectable (`for_update` row-lock, `cas` compare-and-swap, or `disabled`) via `OTARI_BUDGET_STRATEGY`. Per-period resets are driven by `next_budget_reset_at` on the user. `budgets` is many-to-one from `users` and the cap is checked against `users.spend + users.reserved`, so N users sharing one budget each get the full limit; that is per-user, not pooled, and folding counters onto the `budgets` row would silently change it.

`src/gateway/services/scoped_budget_service.py` enforces `scoped_budgets`, a ceiling per `(scope_type, scope_id)` with an optional provider narrowing, resolved from the workspace the request bills to and the tenancy identity behind the key. Every applicable ceiling must admit the estimate, and there is deliberately no sum-under-parent rule. It is opted into per call site by passing `scope=BudgetScopeRequest(...)` to `reserve_budget`; the handle then carries the rows so reconcile and refund unwind all of them. Reservation stays lock-free: one conditional UPDATE per ceiling, each committed on its own, taken in one total order (most specific first, provider-narrowed before aggregate, id as tiebreak) and compensated when a later one refuses. Its management surface is `routes/scoped_budgets.py` (`/v1/scoped-budgets`, master-key, standalone only).

`src/gateway/services/budget_reservation_ledger.py` is the row behind both. The two counters above stay the fast path the gate reads; the ledger gives each hold an identity, which is what the counters could not (mozilla-ai/otari#742). Two things follow from it. Reconcile and refund claim the row before touching a counter, so the second settlement site to fire for one request is a no-op rather than a second refund; that matters because `release_reservation` is reachable from roughly seven sites in `_pipeline.py` and only a `raise` after each has kept two of them from firing, and a double release passes silently as an under-count of live holds because the release expression clamps at zero. And a hold the request never gets back to is reclaimable on its own, by the opportunistic per-user reclaim on the reserve path and by the lifespan sweeper (`budget_reservation_sweep_interval_sec`), rather than waiting for a budget reset that never gave it back anyway: both reset paths zero *spend* and leave the hold in place, so before the ledger a leak shrank the headroom permanently.

Two invariants to keep when editing it. The row is written **after** the holds it records, never before: a hold with no row is the leak the sweep bounds, while a row with no hold would have the sweep hand back an amount nobody is holding. And a top-up grows the row the request already has (`ledger.grow`) instead of opening a second one, because two rows for one request would each carry their own TTL and could be reclaimed independently, releasing part of a live hold.

## Routing policies and router backends
`services/routing/` is the decision half of routing; the API layer's attempt walker executes the plan. `compiler.py` is pure and synchronous: it turns a `PolicySpec` plus request facts into an ordered `CompiledPlan`. A policy whose `select` names a `router` gets its ordering from a backend in `backends.py` (`knn` lives in `knn.py`, the weighted load balancer in `weighted.py`), which is asynchronous (an embedding call, a scan of stored examples), so it runs in `_pipeline._compile_request_plan` via `decide.py` and the result is passed into the compiler as a `RouterOrdering` value. Keep it that way: the compiler must stay callable from `explain` and the CLI with no DB and no I/O. A backend that declines returns an empty ordering, which compiles to the policy's default target; that is the safe path every uncertain case takes.

## Built-in tools vs pass-through
Only `otari_*` tool types are run by the gateway; every other tool type is forwarded to the provider untouched (`src/gateway/api/routes/_tools.py`). `otari_code_execution` → `SandboxBackend` (`services/sandbox_backend.py`), `otari_web_search` → `WebSearchBackend` (`services/web_search_backend.py`). The agentic tool/MCP loop lives in `services/mcp_loop.py`. Request-level guardrails (`services/guardrails.py`) are a caller-opted, input-side check run before the provider; SSRF checks for outbound URLs live in `services/url_safety.py`.

## Outgoing mail
`services/mail/` is the whole of it, and `Mailer` is the only thing a feature should import. A caller asks two questions and never a third: *can this deployment send a message that links back to itself* (`can_send_links`) and *did this one go out* (`send`/`send_template`). It never learns the transport, never wraps a send in a `try` (the mailer turns a delivery failure into a `MailDelivery`, so a mail failure cannot fail the request that triggered it), and never off-loads its own blocking I/O (`asyncio.to_thread` lives in `Mailer.send`, because smtplib is synchronous socket I/O with a timeout per step).

**Mail is optional, and the no-transport case is answered before a send, not by one.** `select_transport` returns `None` for a deployment that configured none, which is what makes "is mail available" a question a surface can ask while deciding whether to offer an affordance. Two shapes follow, and which one a feature takes is a design decision, not a detail:

- **A surface with a non-mail fallback degrades**: an invitation is created either way, and the accept link is returned for an operator to share (`organization_service.invite_active_organization_member_for_user` gates on `can_send_links`). Never refuse there; refusing takes away a flow that works fine without mail.
- **A surface with no fallback is absent or refuses up front**: `Mailer.require_ready()` raises `MailNotConfiguredError`, which the API layer renders as a 503 naming the missing settings (`api/routes/mail.py`). This is what the password-reset and verification flows (#650) gate on, and the dashboard hides such a surface off `mail_ready` from `/v1/bootstrap` so the refusal is a race, not the normal path.

Transport selection is `config.mail_transport` (`auto` derives SMTP from `smtp_host` + `mail_from_email`, `console` logs instead of delivering, `none` is off) and an explicit `smtp` missing either setting is refused at load by `validate_mail_transport`, not at send time. `config.mail_ready` is `mail_enabled` plus `public_base_url`, one flag rather than one per feature because every message this control plane sends carries a link into this deployment.

`console` is the one sanctioned exception to the never-log-a-token rule, and it is deliberately narrow: it writes the rendered message body to the log, which for a control-plane message means a bearer credential in a link (an invitation's accept token, a reset token next). Redacting it would leave the transport unable to do the only thing it is for, so instead it is opt-in per deployment, never reachable from the `auto` default, and announced with a startup warning. Do not widen it, and do not add a second logging transport that carries the same content without the same treatment.

A new message is a body template pair under `templates/email/` plus a typed render function (`services/tenancy/invitation_email.py` is the pattern); the shared layout, escaping, and header sanitization belong to `mail.templates` and should not be reimplemented. Placeholders are `{{LIKE_THIS}}` and are filled in one pass, so a value is never rescanned and a placeholder with no value raises rather than reaching an inbox.

## Runtime credential stores
Two tables let an operator configure at runtime what previously needed `config.yml`, and both
follow one shape: a service overlays stored rows onto the config object, keeping a per-config
baseline so the overlay is recomputed rather than compounded, refreshed by a TTL refresher
wired into the lifespan, standalone-only.

- `services/provider_store_service.py` overlays `provider_credentials` onto `config.providers`
  (baseline on `config._provider_baseline`). `services/secret_box.py` encrypts the keys with
  `OTARI_SECRET_KEY` (Fernet), and `services/master_key_service.py` generates and prints a
  master key on first run when none is set (hash in `runtime_settings`). Keys are write-only
  over the API: a response carries `last4` and nothing more.
- `services/search_tool_store_service.py` overlays `search_tool_credentials` onto
  `config.search_tools` (baseline on `config._search_tool_baseline`), which is what lets
  `POST /v1/search` work with no config file. Tools defined in the config file stay read-only.

Their pages are in [../../web/AGENTS.md](../../web/AGENTS.md).

Below both stores is one more rung, deliberately last: when a candidate needed a credential
and none of them had one, `resolve_dispatch_provider` asks `ModelProviderPort` whether this
build serves it from a deployment-owned fleet. `_serve_from_hosted_credential` in
`api/routes/_pipeline.py` is the whole of it and says why on each branch. Nothing here may
move above BYO.

Neither store becomes workspace-keyed. Per-tenant tool configuration (web search, code
execution, MCP servers, guardrails) is resolved at admission in `prepare_gateway_tools`
off `RequestContext.workspace_id` and `organization_id`, never from a header, and lands on
`ToolContext` for the tool loop to read (guardrails, which never enter that loop, go
straight into the pre-provider check instead). A lower layer may only narrow what the
layer above it permits, so a workspace row is a veto and a refinement and never a grant,
and no row means no narrowing (MCP, which has no deployment-level server list to narrow,
is the stated exception). The
question is [#655](https://github.com/mozilla-ai/otari/issues/655) and the reasoning is in
the PR that answered it,
[#678](https://github.com/mozilla-ai/otari/pull/678). All four surfaces the decision
covers have landed now, and each is described below as a worked example of it; read both
before adding a fifth.

MCP's own surface is the one that has landed (#658). `workspace_mcp_servers`
(`models/entities.py`) holds a workspace's configured servers, with the bearer token
encrypted by the same `secret_box` the two stores above use; it is managed over
`/v1/workspaces/{workspace_id}/mcp-servers` through
`services/tenancy/workspace_mcp_server_service.py`, and read on the request path by
`_pipeline._resolve_mcp_server_ids`, the standalone half of an `mcp_server_ids` field the
platform answers in hybrid mode. There is no overlay and no cache: it resolves per
request against `ctx.db`, which is what the decision above says the seam should do.

Code execution is the second to land (#657), and the first that has a
deployment-wide setting to compose over, so it is the worked example of the
narrowing rule rather than of the exception. `workspace_code_execution_policies`
(`models/entities.py`) holds one row per workspace or none, managed over
`/v1/workspaces/{workspace_id}/code-execution-policy` through
`services/tenancy/workspace_code_execution_policy_service.py`, whose two halves are
the role-gated CRUD and the identity-free `resolve_workspace_code_execution_policy`
that `prepare_gateway_tools` calls. The row carries no URL and no credential: the
sandbox stays deployment-scoped on `/v1/tool-settings`, and the row only refuses it,
lowers its two ceilings, supplies a hint the request omitted, removes tool kinds
from what the backend serves, or pins an image from a list the operator curated
(`sandbox_allowed_session_images`, deliberately config-only rather than dashboard-editable,
because it is a supply-chain gate and not a tool setting). No row means no
narrowing. The image guard is enforced twice on purpose, at the write and again at
admission: an operator can shrink the curated list after a workspace pinned from it,
and a stale pin refuses the request rather than quietly falling back.

Web search is the third (#656), and the one whose narrowing is not just a number.
`workspace_web_search_configs` (`models/entities.py`) holds one row per workspace
or none, managed over `/v1/workspaces/{workspace_id}/web-search` through
`services/tenancy/workspace_web_search_service.py`, whose halves are the
role-gated CRUD, the identity-free `resolve_workspace_web_search_config`, and the
pure `narrow_web_search_tool_entry` that composes a row onto a request's tool
entry. It carries no URL and no credential: the backend stays deployment-scoped
on `/v1/tool-settings`, and the `/v1/search` tools stay on `/v1/search-tools`.

Two things about it are worth knowing before editing either side.

- **It narrows where the hybrid path defaults.** `prepare_gateway_tools`'s hybrid
  arm applies the policy it resolves from otari.ai as a set of defaults a request
  overrides, which is the platform's own contract and is left alone. The
  standalone arm floors `max_results`, unions `blocked_domains` and intersects
  `allowed_domains` instead, because under default-only precedence a request
  sheds a workspace's block-list simply by sending one of its own. An empty
  intersection is refused (`WorkspaceWebSearchDomainsExcludedError`) rather than
  stored as `[]`, which `_build_web_search_backend` would read as *no* allow-list.
- **`POST /v1/search` honors the veto too**, in `routes/search.py`, and only the
  veto. It is a second door into the same capability, and leaving it open would
  make the switch bypassable by any key in the workspace; the row's other fields
  shape the in-loop backend's own request and have no counterpart in that
  endpoint's provider adapters.

Guardrails are the fourth and last (#654), and the one exception to the direction
of the other three: the plane sits *above* the deployment rather than below it, because a
guardrail is a restriction a tenant accepts and not a capability it acquires, so
adding one can only make fewer requests succeed. `organization_guardrails` plus
`organization_guardrail_workspaces` (`models/entities.py`) hold what an
organization mandates and which of its workspaces run it, managed over
`/v1/organizations/me/guardrails` through
`services/tenancy/organization_guardrail_service.py`. Three consequences follow
from the direction, and none of them generalize to the other three:

- **An entry may carry its own `url` and credential**, where a workspace code-execution
  policy may not. A caller can already point a request-body guardrail at an endpoint of
  their own (`models/guardrails.GuardrailConfig.url`, SSRF-checked on the request path),
  so storing one reaches nothing new. The credential is encrypted with the same
  `secret_box` and sent as `Authorization: Bearer` by `run_input_guardrails`, keyed by
  profile rather than carried on `GuardrailConfig`, which is a request-body model.
- **The scope is org-controlled and a workspace has no veto**, since a veto is the one
  thing that would widen what succeeds.
- **The resolve is unconditional**, unlike the MCP and code-execution ones, which run
  only when a request opts into the feature: a mandate that ran only when the caller
  asked for it would not be a mandate. `_pipeline._resolve_organization_guardrails` is
  one indexed read per completion request, folded in by `merge_guardrail_layers`
  alongside a routing policy's mandate, with the operator's layer outermost. No entry in
  scope means nothing runs, and `guardrails_url` stays a `runtime_settings` concern.

## The first-request setup guide
`services/tenancy/workspace_activation_service.py` is the state behind the dashboard's setup guide (`routes/workspace_activation.py`, `/v1/workspaces/{id}/activation`): whether to offer it, the API key it hands out, what the workspace's traffic says about the attempt, and the dismissal that retires it. Ported from the platform's `WorkspaceActivationService`, and the one departure to know is that **activation is derived, not recorded**: the first successful row in `usage_logs` this deployment served itself (`core/usage_source.served_here`, which covers a row migrated from hosted history as well as one recorded live) *is* the evidence, read through `ix_usage_logs_workspace_source_status_timestamp`. The platform stores that telemetry in columns because its usage pipeline is asynchronous and crosses services; here the row is written by this process into this database, so a second copy could only drift. `workspace_activation_state` therefore holds one row per workspace carrying what cannot be observed elsewhere (the dismissal, when a key was last issued, and which key it was).

Two consequences worth keeping. Imported usage (`POST /v1/usage/external-events`) is excluded, so a workspace whose only rows came from an import has still never called this gateway; and `absorbed` rows are excluded from the latest-attempt read, because a failed attempt a routing policy recovered from is not the request's outcome. Issuing the key is a workspace management action (`authorization.require_workspace_management_access`) and rotates one `api_keys` row in place rather than minting a second, while `GET` is readable by any member and answers `experience_eligible: false` for one who may not act. `config.activation_guide` turns the whole flow off without unmounting the endpoints.

## Data, sessions, migrations
ORM entities are in `src/gateway/models/entities.py` (User, APIKey, Budget, ScopedBudget, UsageLog, ModelPricing, BudgetResetLog). The async engine/session factory and `init_db` live in `src/gateway/core/database.py`; routes get a session via the `get_db` dependency, non-request code uses `create_session()`. Alembic migrations are in `alembic/versions/` and run on startup when `auto_migrate` is set.

**Two model styles, one schema.** The reconciled control plane's tables arrive from the platform as SQLModel classes (`models/tenancy.py`: organizations, workspaces, identities, memberships) and keep that form, because the `Create`/`Update`/`Public` schemas on them *are* the endpoint contracts the generated dashboard client is built from. `entities.py`'s `Base` shares `SQLModel.metadata`, so both styles land in one collection and one Alembic chain; `models/__init__.py` imports every model module so anything touching that metadata sees the whole schema. New *gateway* tables still go in `entities.py`. How to write against the SQLModel half (the async `BaseRepository`, the `col()` rule, the flush-not-commit contract) is in [the backend skill](../../.github/skills/backend-standards/SKILL.md).

## Config layering
`GatewayConfig` (`src/gateway/core/config.py`) loads `config.yml` (with `${VAR}` env interpolation) and layers env vars on top. The user-facing prefix is `OTARI_`, applied both as init overrides for every scalar field via `_apply_otari_env_overrides` (so `OTARI_` wins over YAML) and as the native pydantic `env_prefix` (which covers complex fields). Service-level env vars (e.g. web search, guardrails) read through `otari_env()` in `core/env.py`, which reads the `OTARI_` prefix.

## Database patterns
- Use Alembic migrations; do not manually mutate live schemas.
- For write operations:
  - `db.commit()` in `try`,
  - `db.rollback()` on `SQLAlchemyError`,
  - re-raise mapped API/domain errors.
- Reuse repository helpers (e.g., `get_active_user`) for shared query logic.
- Conditions shared between a route's read filters and a service's write selection live in `core/sql.py` (e.g. `match_any`, `MAX_FILTER_VALUES`), not duplicated on each side: a service may not import the API layer, so `core` is the only place both can reach.

## Usage filters: read and mutate must agree
A bulk usage mutation (`DELETE /v1/usage`, `POST /v1/usage/set-price`) can target rows two ways: an explicit `ids` list, or `by_filter` plus filter fields. The dashboard sizes its "select all N matching" affordance from `GET /v1/usage/count` and then sends the filter fields, so the server re-derives the target set from the body rather than from the count. Three consequences, each of which has been violated at least once:

- **Every filter that scopes the table must exist on `UsageSelection`.** One left out widens the delete past the rows the operator was shown (a session drill-down would wipe every other session).
- **A filter's accepted value space must match on both sides.** `model` / `user_id` / `api_key_id` are repeatable, so the selection body takes lists too; a body that could only express one value would target every value of that dimension.
- **The bounds must match as well.** The read endpoints cap a repeatable filter at `MAX_FILTER_VALUES`, so the selection body carries the same ceiling. Without it a value set `/count` refuses (422) was still deletable, on an unbounded `IN` list. Annotate the bound on the list arm (`str | Annotated[list[str], Field(max_length=...)] | None`); on the union it would also cap a single value's character length and reject a long `provider:model` name.

The imported-only guards in `_selection_conditions` (`core/usage_source.not_served_here` plus `counts_toward_budget = False`) bound the damage but do not substitute for any of the above: they keep a mutation off gateway rows, not off the wrong imported rows.

## Logging
- Use module logger from `gateway.log_config`.
- Prefer structured/contextual log messages with `%s` formatting placeholders.
- Never log secrets, tokens, or raw API keys (see the root `AGENTS.md`; the bootstrap master-key print is an intentional one-time exception).
