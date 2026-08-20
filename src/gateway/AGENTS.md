# Gateway internals (`src/gateway/`)

`src/gateway/CLAUDE.md` is a one-line `@AGENTS.md` import of this file. Always edit `AGENTS.md` directly; never modify `CLAUDE.md`, and do not delete it: Claude Code discovers `CLAUDE.md`, not `AGENTS.md`, so the import is what loads this file.

Loaded when working under `src/gateway/`. For house style (async SQLAlchemy 2.0,
layering, the budget/reservation lifecycle, migrations, config/logging), read
[../../.github/skills/backend-standards/SKILL.md](../../.github/skills/backend-standards/SKILL.md).

The two runtime modes and the OSS/enterprise seam are in the root `AGENTS.md`;
read that first if a change touches mode selection.

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

## Budget enforcement
Two mechanisms, both enforced, neither replacing the other.

`src/gateway/services/budget_service.py` reserves an estimated cost before the call and reconciles/refunds after. Strategy is selectable (`for_update` row-lock, `cas` compare-and-swap, or `disabled`) via `OTARI_BUDGET_STRATEGY`. Per-period resets are driven by `next_budget_reset_at` on the user. `budgets` is many-to-one from `users` and the cap is checked against `users.spend + users.reserved`, so N users sharing one budget each get the full limit; that is per-user, not pooled, and folding counters onto the `budgets` row would silently change it.

`src/gateway/services/scoped_budget_service.py` enforces `scoped_budgets`, a ceiling per `(scope_type, scope_id)` with an optional provider narrowing, resolved from the workspace the request bills to and the tenancy identity behind the key. Every applicable ceiling must admit the estimate, and there is deliberately no sum-under-parent rule. It is opted into per call site by passing `scope=BudgetScopeRequest(...)` to `reserve_budget`; the handle then carries the rows so reconcile and refund unwind all of them. Reservation stays lock-free: one conditional UPDATE per ceiling, each committed on its own, taken in one total order (most specific first, provider-narrowed before aggregate, id as tiebreak) and compensated when a later one refuses. Its management surface is `routes/scoped_budgets.py` (`/v1/scoped-budgets`, master-key, standalone only).

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

Neither store becomes workspace-keyed. Per-workspace tool configuration (guardrails, web
search, code execution, MCP servers) is resolved at admission in `prepare_gateway_tools`
off `RequestContext.workspace_id`, never from a header, and lands on `ToolContext` for the
tool loop to read. A lower layer may only narrow what the layer above it permits, so a
workspace row is a veto and a refinement and never a grant, and no row means no narrowing
(MCP, which has no deployment-level server list to narrow, is the stated exception). The
question is [#655](https://github.com/mozilla-ai/otari/issues/655) and the reasoning is in
the PR that answered it,
[#678](https://github.com/mozilla-ai/otari/pull/678). Read both before building any of
#654, #656, #657 or #658, each of which settles its own surface under the rule above.

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

The imported-only guards in `_selection_conditions` (`source != "gateway"` plus `counts_toward_budget = False`) bound the damage but do not substitute for any of the above: they keep a mutation off gateway rows, not off the wrong imported rows.

## Logging
- Use module logger from `gateway.log_config`.
- Prefer structured/contextual log messages with `%s` formatting placeholders.
- Never log secrets, tokens, or raw API keys (see the root `AGENTS.md`; the bootstrap master-key print is an intentional one-time exception).
