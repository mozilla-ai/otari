# AGENTS.md

Guidance for agentic coding tools working in this repository.
Scope: entire repo.

`CLAUDE.md` is a symlink to this file. Always edit `AGENTS.md` directly; never modify `CLAUDE.md`.

## Project Snapshot
- Project: `otari`, an OpenAI-compatible LLM gateway (API key management, budget enforcement, usage tracking). The Python package is named `gateway` (not `otari`): the `otari` distribution name on PyPI belongs to the Otari client SDK, which `any-llm-sdk` depends on, so a top-level `otari` import package here would collide with it. User-facing names (CLI, env vars, docs, OpenAPI title) are `otari`; only the internal import path stays `gateway`.
- Language/runtime: Python 3.13+.
- Package manager + task runner: `uv`.
- App type: FastAPI gateway service with async SQLAlchemy + Alembic.
- Source root: `src/gateway` (the package is imported as `gateway.*`; uvicorn runs with `--app-dir src`).
- Tests: `tests/unit` and `tests/integration`.
- Database: SQLite by default (async via `aiosqlite`), PostgreSQL in integration tests (async via `asyncpg`).
- Provider calls go through the `any-llm` SDK (`any_llm`), not hand-rolled HTTP clients.

## Architecture (Big Picture)
Read these together before changing request behavior, the flow spans several files.

### Two runtime modes
- Mode is derived, not configured directly: `GatewayConfig.is_hybrid_mode` / `effective_mode` (`src/gateway/core/config.py`) return `hybrid` when a platform token (`OTARI_AI_TOKEN`, plus legacy aliases) is set, else `standalone`. Setting `OTARI_MODE=hybrid` (legacy value `platform`; legacy alias `GATEWAY_MODE`) without a token fails at startup.
- **Standalone**: provider credentials come from the `providers:` block in `config.yml`; users/keys/budgets/usage live in the local DB. All routers are registered.
- **Hybrid**: per-request provider credentials are resolved from the platform service (otari.ai); local DB/user/budget management is skipped and usage is reported upstream. `register_routers()` (`src/gateway/api/main.py`) only mounts `chat`, `messages`, `responses`, and `health`; management routers (keys/users/budgets/pricing/usage/etc.) are standalone-only.
- Hybrid mode spans two trust contexts that this codebase treats identically: a gateway someone self-hosts against otari.ai using a workspace's own (BYO) provider keys, and the gateway mozilla.ai operates as part of otari.ai, which additionally serves mozilla.ai-managed models. The managed-vs-BYO boundary (platform-owned upstream credentials are returned only to mozilla.ai's gateway, never to a self-hosted one) is enforced on the platform side (otari-ai), not here. User-facing explanation lives in `docs/modes.md`; the wire contract in `docs/hybrid-mode-protocol.md`.

### Request lifecycle (chat completions)
1. App + middleware: `src/gateway/main.py` builds the FastAPI app, adds CORS + a security-headers middleware, and enforces auth on every path except `_PUBLIC_PREFIXES` (`/health`).
2. Auth: `src/gateway/api/deps.py` extracts the key from `Otari-Key` (canonical `API_KEY_HEADER` in `core/config.py`), the legacy `AnyLLM-Key`/`X-AnyLLM-Key` aliases, or `Authorization: Bearer`; validates the SHA-256 hash against the `api_keys` table, or matches the master key.
3. Route handler: `src/gateway/api/routes/chat.py` resolves the billed user, runs budget checks (standalone) or resolves platform credentials, applies input guardrails, and extracts gateway-managed tools.
4. Dispatch: the provider/model is split with `AnyLLM.split_model_provider(...)` and the call is made via `acompletion(...)` from `any_llm`. Hybrid mode walks multiple resolved attempts with fallback (`src/gateway/api/routes/_platform.py`, streaming in `src/gateway/streaming.py`).
5. Usage + budget reconciliation: standalone writes a `UsageLog` row via the log writer and reconciles spend; platform reports usage upstream.

### Budget enforcement
`src/gateway/services/budget_service.py` reserves an estimated cost before the call and reconciles/refunds after. Strategy is selectable (`for_update` row-lock, `cas` compare-and-swap, or `disabled`) via `OTARI_BUDGET_STRATEGY` (legacy `GATEWAY_BUDGET_STRATEGY`). Per-period resets are driven by `next_budget_reset_at` on the user.

### Built-in tools vs pass-through
Only `otari_*` tool types are run by the gateway; every other tool type is forwarded to the provider untouched (`src/gateway/api/routes/_tools.py`). `otari_code_execution` → `SandboxBackend` (`services/sandbox_backend.py`), `otari_web_search` → `WebSearchBackend` (`services/web_search_backend.py`). The agentic tool/MCP loop lives in `services/mcp_loop.py`. Request-level guardrails (`services/guardrails.py`) are a caller-opted, input-side check run before the provider; SSRF checks for outbound URLs live in `services/url_safety.py`.

### Data, sessions, migrations
ORM entities are in `src/gateway/models/entities.py` (User, APIKey, Budget, UsageLog, ModelPricing, BudgetResetLog). The async engine/session factory and `init_db` live in `src/gateway/core/database.py`; routes get a session via the `get_db` dependency, non-request code uses `create_session()`. Alembic migrations are in `alembic/versions/` and run on startup when `auto_migrate` is set.

### Config layering
`GatewayConfig` (`src/gateway/core/config.py`) loads `config.yml` (with `${VAR}` env interpolation) and layers env vars on top. The user-facing prefix is `OTARI_` (applied as init overrides for every scalar field via `_apply_otari_env_overrides`); the legacy `GATEWAY_` prefix is still honored as the native pydantic prefix and as a fallback (`OTARI_` wins when both are set). Service-level env vars (e.g. web search, guardrails) read through `otari_env()` in `core/env.py`, which applies the same `OTARI_`-then-`GATEWAY_` precedence.
## Setup Commands
- Create venv: `uv venv`
- Activate venv: `source .venv/bin/activate`
- Install deps (dev): `uv sync --dev`
- Install deps exactly as lockfile (CI-style): `uv sync --dev --frozen`
## Run Commands
- Run Otari from config: `uv run otari serve --config config.yml` (the `gateway` command remains as a legacy alias)
- Run dev server (reload + `.env`): `make dev`
- Initialize DB schema: `uv run otari init-db --config config.yml`
- Run migrations to head: `uv run otari migrate --config config.yml`
- Run migrations to specific revision: `uv run otari migrate --revision <rev>`
## Build / Packaging Commands
- Python package build backend is configured via `setuptools` in `pyproject.toml`.
- If you need a local package build artifact, use: `uv build`
- Docker local build/run: `docker compose up --build`
- CI Docker smoke check is implemented in `scripts/docker_liveness_check.sh`.
## Lint / Typecheck Commands
- Ruff is configured for linting (rules: `E`, `F`, `I`; line length: 120) in `pyproject.toml`.
- Run lint checks with `make lint` (or `uv run ruff check src tests scripts`).
- Ruff is also enforced in CI via `.github/workflows/otari-lint.yml`.
- Primary static checks present in dev dependencies: `ruff`, `mypy`.
- mypy is configured `strict` over `src`, `tests`, and `scripts` (`pyproject.toml`).
- Run type checks with `make typecheck` (or `uv run mypy`).
- mypy is also enforced in CI via `.github/workflows/otari-typecheck.yml`.
- If introducing a formatter/linter, keep changes in a separate PR unless requested.
## Test Commands
- Main local suite (matches Makefile):
  - `make test`
  - expands to `uv run pytest -v tests/unit tests/integration`
  - unit only: `make test-unit`; integration only: `make test-integration`
- CI-style tests with coverage + parallel (unit and integration run separately):
  - `uv run pytest tests/unit -v --cov --cov-report=xml -n auto`
  - `uv run pytest tests/integration -v --cov --cov-report=xml --cov-append -n auto`
### Running a Single Test (important)
- Single test file:
  - `uv run pytest tests/unit/test_gateway_cli.py -v`
- Single test function via node id:
  - `uv run pytest tests/unit/test_gateway_cli.py::test_gateway_config_defaults_to_sqlite -v`
- Single integration test function:
  - `uv run pytest tests/integration/test_health.py::test_health_check -v`
- Pattern-filtered run:
  - `uv run pytest tests/integration -k "budget and reset" -v`
### Test Environment Notes
- `pytest.ini` sets:
  - `timeout = 120`
  - `reruns = 5`
  - `reruns_delay = 10`
- Integration tests can use:
  - `TEST_DATABASE_URL` (if provided), or
  - Testcontainers PostgreSQL (`postgres:17`) if not provided.
- Running integration tests without Docker may fail when Testcontainers is required.
## OpenAPI Spec Commands
- Generate OpenAPI JSON:
  - `uv run python scripts/generate_openapi.py`
- Check OpenAPI is up to date (CI behavior):
  - `make openapi-check` (or `uv run python scripts/generate_openapi.py --check`)
- Default output path:
  - `docs/public/openapi.json`
## Repository Conventions
### Imports
- Use grouped imports in this order:
  1) stdlib
  2) third-party
  3) local (`gateway.*`)
- Separate groups with one blank line.
- Prefer explicit imports over wildcard imports.
- Use `TYPE_CHECKING` for type-only imports when helpful (`routes/_helpers.py`).
### Formatting
- Follow existing style consistent with Black-like formatting:
  - trailing commas in multiline literals/calls,
  - one statement per line,
  - clear vertical spacing between top-level defs/classes.
- Keep docstrings concise and meaningful for public functions/classes.
- Avoid adding comments unless logic is non-obvious.
### Typing
- Add type hints to new/changed functions (project is strongly typed in practice).
- Prefer modern syntax:
  - `str | None` over `Optional[str]`
  - built-in generics like `list[str]`, `dict[str, Any]`
- Annotate SQLAlchemy fields with `Mapped[...]` in ORM models.
- For FastAPI dependencies, prefer `Annotated[..., Depends(...)]`.
### Naming
- `snake_case`: functions, variables, module names.
- `PascalCase`: classes, Pydantic models, SQLAlchemy entities.
- Constants: `UPPER_SNAKE_CASE` (`API_KEY_HEADER`, `_PUBLIC_PREFIXES`).
- Test files: `test_*.py`; test functions: `test_*`.
### FastAPI / API Patterns
- Define request/response schemas with Pydantic models near route handlers.
- Keep route prefixes/tags explicit on each router.
- Return typed response models instead of raw dicts when practical.
- Use HTTP status constants from `fastapi.status`.
### Database Patterns
- Use Alembic migrations; do not manually mutate live schemas.
- For write operations:
  - `db.commit()` in `try`,
  - `db.rollback()` on `SQLAlchemyError`,
  - re-raise mapped API/domain errors.
- Reuse repository helpers (e.g., `get_active_user`) for shared query logic.
### Error Handling
- Raise explicit `HTTPException` in API layer with clear `detail` messages.
- Preserve security posture: do not leak internals in public error responses.
- Service-specific exceptions live alongside their service modules in `src/gateway/services/` (e.g. `UnsafeURLError`, `GuardrailsNotReachableError`).
- Prefer specific exceptions (`ValueError`, `SQLAlchemyError`) over broad `except Exception`.
### Logging
- Use module logger from `gateway.log_config`.
- Prefer structured/contextual log messages with `%s` formatting placeholders.
- Do not log secrets, tokens, or raw API keys (bootstrap exception is intentional one-time behavior).
## CI Rules to Mirror Locally
- Python version for CI: 3.14 (`.github/workflows/otari-tests.yml`), matching the Docker image; the package still supports 3.13+ (`requires-python = ">=3.13"`).
- Install deps with frozen lockfile in CI.
- Tests run with coverage and xdist in CI.
- OpenAPI spec freshness is enforced in CI (`--check`).
- `CHANGELOG.md` and the GitHub Release body are generated from Conventional
  Commits by git-cliff (`cliff.toml`) at release time, not per-PR. Because PRs are
  squash-merged, the PR title is what git-cliff parses; `otari-pr-title.yml`
  enforces a conventional title. Visibility rules live in `RELEASE.md`
  ("Changelog visibility"). Do not hand-edit `CHANGELOG.md`; the release
  workflows regenerate it.
## Practical Agent Workflow
- Before coding: read nearby module + related tests.
- After coding: run the smallest relevant pytest node id first.
- Then run broader impacted tests.
- If API contract changed: regenerate or check OpenAPI spec.
- Keep diffs focused; avoid opportunistic refactors unless requested.

## Key Paths
- App entry and wiring: `src/gateway/main.py`
- CLI entrypoint: `src/gateway/cli.py`
- Config + env resolution: `src/gateway/core/config.py`
- DB init/session helpers: `src/gateway/core/database.py`
- API routers: `src/gateway/api/routes/`
- ORM models: `src/gateway/models/entities.py`
- Alembic migrations: `alembic/versions/`
- OpenAPI generator: `scripts/generate_openapi.py`

## Change Validation Checklist
- If you touched API routes or schemas, run relevant integration tests first.
- If you touched DB models/repositories, run related integration tests and migration paths.
- If you touched config loading, run config/env tests in `tests/integration`.
- If you touched CLI behavior, run `tests/unit/test_gateway_cli.py`.
- If you touched auth headers or key handling, run key-management and auth-related tests.
- If OpenAPI-affecting code changed, run `make openapi-check` (or `uv run python scripts/generate_openapi.py --check`).

## Writing style

- Avoid em dashes and double hyphens (`--`) used as separators in prose
  (README, docs, doc comments, commit messages, PR descriptions). Use commas,
  semicolons, colons, parentheses, or periods, or rephrase. This does not apply
  to code (for example CLI flags like `--all`) or en-dash numeric ranges like `3–4`.

## Notes for Agents
- Prefer minimal, targeted edits over broad refactors.
- Maintain import order and existing typing style in touched files.
- Preserve security-relevant behavior (header parsing, auth checks, error detail boundaries).
- Keep test additions close to changed behavior (unit for pure logic, integration for route/database behavior).
