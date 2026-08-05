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

## Budget enforcement
`src/gateway/services/budget_service.py` reserves an estimated cost before the call and reconciles/refunds after. Strategy is selectable (`for_update` row-lock, `cas` compare-and-swap, or `disabled`) via `OTARI_BUDGET_STRATEGY`. Per-period resets are driven by `next_budget_reset_at` on the user.

## Routing policies and router backends
`services/routing/` is the decision half of routing; the API layer's attempt walker executes the plan. `compiler.py` is pure and synchronous: it turns a `PolicySpec` plus request facts into an ordered `CompiledPlan`. A policy whose `select` names a `router` gets its ordering from a backend in `backends.py` (`knn` lives in `knn.py`), which is asynchronous (an embedding call, a scan of stored examples), so it runs in `_pipeline._compile_request_plan` via `decide.py` and the result is passed into the compiler as a `RouterOrdering` value. Keep it that way: the compiler must stay callable from `explain` and the CLI with no DB and no I/O. A backend that declines returns an empty ordering, which compiles to the policy's default target; that is the safe path every uncertain case takes.

## Built-in tools vs pass-through
Only `otari_*` tool types are run by the gateway; every other tool type is forwarded to the provider untouched (`src/gateway/api/routes/_tools.py`). `otari_code_execution` → `SandboxBackend` (`services/sandbox_backend.py`), `otari_web_search` → `WebSearchBackend` (`services/web_search_backend.py`). The agentic tool/MCP loop lives in `services/mcp_loop.py`. Request-level guardrails (`services/guardrails.py`) are a caller-opted, input-side check run before the provider; SSRF checks for outbound URLs live in `services/url_safety.py`.

## Data, sessions, migrations
ORM entities are in `src/gateway/models/entities.py` (User, APIKey, Budget, UsageLog, ModelPricing, BudgetResetLog). The async engine/session factory and `init_db` live in `src/gateway/core/database.py`; routes get a session via the `get_db` dependency, non-request code uses `create_session()`. Alembic migrations are in `alembic/versions/` and run on startup when `auto_migrate` is set.

## Config layering
`GatewayConfig` (`src/gateway/core/config.py`) loads `config.yml` (with `${VAR}` env interpolation) and layers env vars on top. The user-facing prefix is `OTARI_`, applied both as init overrides for every scalar field via `_apply_otari_env_overrides` (so `OTARI_` wins over YAML) and as the native pydantic `env_prefix` (which covers complex fields). Service-level env vars (e.g. web search, guardrails) read through `otari_env()` in `core/env.py`, which reads the `OTARI_` prefix.

## Database patterns
- Use Alembic migrations; do not manually mutate live schemas.
- For write operations:
  - `db.commit()` in `try`,
  - `db.rollback()` on `SQLAlchemyError`,
  - re-raise mapped API/domain errors.
- Reuse repository helpers (e.g., `get_active_user`) for shared query logic.

## Logging
- Use module logger from `gateway.log_config`.
- Prefer structured/contextual log messages with `%s` formatting placeholders.
- Never log secrets, tokens, or raw API keys (see the root `AGENTS.md`; the bootstrap master-key print is an intentional one-time exception).
