# AGENTS.md

Guidance for agentic coding tools working in this repository.
Scope: entire repo.

`CLAUDE.md` is a one-line `@AGENTS.md` import of this file, not a symlink, so it survives Windows clones (Git for Windows checks symlinks out as plain text files by default). Always edit `AGENTS.md` directly; never modify `CLAUDE.md`. The same pairing is used in `web/` and `src/gateway/`.

## Project Snapshot
- Project: `otari`, an OpenAI-compatible LLM gateway (API key management, budget enforcement, usage tracking). The Python package is named `gateway` (not `otari`): the `otari` distribution name on PyPI belongs to the Otari client SDK, which `any-llm-sdk` depends on, so a top-level `otari` import package here would collide with it. User-facing names (CLI, env vars, docs, OpenAPI title) are `otari`; only the internal import path stays `gateway`.
- Provider calls go through the `any-llm` SDK (`any_llm`), not hand-rolled HTTP clients.

## Skills & Scoped Instructions
Detailed, task-scoped guidance lives outside this file so it loads only when relevant
(progressive disclosure). Read the applicable one before editing:

- **Backend (`src/gateway/`)** → [src/gateway/AGENTS.md](src/gateway/AGENTS.md) (request lifecycle, budget enforcement, built-in tools, data/sessions, config layering, DB + logging patterns) and [.github/skills/backend-standards/SKILL.md](.github/skills/backend-standards/SKILL.md): async SQLAlchemy house style, layering, the budget/reservation lifecycle, migrations, config/logging conventions.
- **Dashboard (`web/`)** → [web/AGENTS.md](web/AGENTS.md) (auth/session model, runtime provider management, build + bundled guide, PWA, serving) and [.github/skills/frontend-standards/SKILL.md](.github/skills/frontend-standards/SKILL.md): HeroUI v3, the `--otari-*` design tokens, TanStack Query patterns, and Vitest testing for the admin dashboard.
- **Reviewing a change** → the path-scoped files in [.github/instructions/](.github/instructions/) auto-apply during Copilot reviews (they carry `applyTo` globs): [security-review](.github/instructions/security-review.instructions.md) (budget/tenant isolation, auth, SSRF, prompt injection) and [performance-review](.github/instructions/performance-review.instructions.md) (N+1, indexes, pagination limits, transaction atomicity) for `src/gateway/`, and [frontend-standards](.github/instructions/frontend-standards.instructions.md) (HeroUI v3, design tokens, TanStack Query) for `web/`.
- **Adding a table or a cross-repo surface** → [reconciliation-ledger](.github/instructions/reconciliation-ledger.instructions.md): append an entry to the M4 ledger ([otari-ai#1587](https://github.com/mozilla-ai/otari-ai/issues/1587)) when a PR adds or changes a persistent table, a control-plane contract the platform consumes or serves, a capability the platform also has, or a mode-specific surface. One ledger covers both repositories.

The `.claude/skills` directory symlinks to `.github/skills`, so the same skills are available to Claude and to GitHub Copilot from one source.

## Architecture (Big Picture)
For the open-core OSS/enterprise seam (ports, adapters, the capability lines, and the rules for keeping the boundary), see [ARCHITECTURE.md](ARCHITECTURE.md). It is a north-star document describing the intended architecture, so ground current-state work in `src/gateway/`.

### Two runtime modes
- Mode is derived when `OTARI_MODE` is unset, and honored when set: `GatewayConfig.is_hybrid_mode` / `effective_mode` (`src/gateway/core/config.py`) return `hybrid` when the config field `mode` is `hybrid` (legacy `platform`) or, when `mode` is unset, when the platform token (`OTARI_AI_TOKEN`) is set; otherwise `standalone`. Startup validation (`validate_mode_selection`) rejects the two conflicting combinations: `OTARI_MODE=hybrid` (legacy value `platform`) without a token, and `OTARI_MODE=standalone` with a token set (the token would otherwise silently select hybrid). The token is resolved once at config-load time (cached on the config), not re-read from `os.getenv` on every access.
- **Standalone**: provider credentials come from the `providers:` block in `config.yml`; users/keys/budgets/usage live in the local DB. All routers are registered.
- **Hybrid**: per-request provider credentials are resolved from the platform service (otari.ai); local DB/user/budget management is skipped and usage is reported upstream. `register_routers()` (`src/gateway/api/main.py`) only mounts `chat`, `messages`, `responses`, and `health`; management routers (keys/users/budgets/pricing/usage/etc.) are standalone-only.
- Hybrid mode spans two trust contexts that this codebase treats identically: a gateway someone self-hosts against otari.ai using a workspace's own (BYO) provider keys, and the gateway mozilla.ai operates as part of otari.ai, which additionally serves mozilla.ai-managed models. The managed-vs-BYO boundary (platform-owned upstream credentials are returned only to mozilla.ai's gateway, never to a self-hosted one) is enforced on the platform side (otari-ai), not here. User-facing explanation lives in `docs/modes.md`; the wire contract in `docs/hybrid-mode-protocol.md`.

The per-request flow (auth → budget → dispatch → reconciliation) spans several files and is documented in [src/gateway/AGENTS.md](src/gateway/AGENTS.md). Read it before changing request behavior.

## Lint / Typecheck
- Run lint checks with `make lint`; it runs the architecture check and then Ruff. **Ruff alone is not equivalent.**
- The architecture check (`scripts/check_architecture.py`, also `make check-architecture`) enforces the `src/gateway/` layer rules: services must not import the API layer, repositories must not import services or the API layer, API routes must not import `sqlalchemy.orm`, and repository modules end in `_repository.py`.
- If introducing a formatter/linter, keep changes in a separate PR unless requested.

## Test Notes
- There is no global rerun policy in `pytest.ini`. A blanket `reruns` would let a test that
  passes one time in several count as green, hiding flakiness and ordering
  bugs suite-wide. Mark a genuinely flaky test with
  `@pytest.mark.flaky(reruns=...)` (from `pytest-rerunfailures`) and say why,
  rather than reintroducing a global retry.
- Integration tests need PostgreSQL: `TEST_DATABASE_URL` if set, otherwise a Testcontainers `postgres:17`, so without Docker the suite cannot start. SQLite is not a fallback even though `_to_async_url` accepts one: the fixtures tear down with `DROP TABLE ... CASCADE`, which SQLite rejects, so every test errors in teardown. With no Docker, point `TEST_DATABASE_URL` at any reachable PostgreSQL instead.
- Two tests assert the provider-error sanitization by making a real outbound call (`test_error_detail_leakage.py::test_provider_error_does_not_leak_details`, `test_streaming_error_event.py::test_streaming_creation_error_returns_http_error`). With no network egress the upstream fails differently and both report a status mismatch, so treat them as environment noise rather than a regression, and confirm a change against the rest of the suite.

## Generated Artifacts
- The Postman collection is generated **from** `docs/public/openapi.json`, so it goes stale
  whenever the spec does, including for a change that only edits a route's
  docstring (descriptions are carried into the collection). Regenerate both and commit
  both: `uv run python scripts/generate_openapi.py`, then `make postman`. Note there is
  no `make openapi` target; `make openapi-check` only validates. Verify with
  `make openapi-check` and `make postman-check`. The same `openapi-spec` CI job runs both
  checks, so missing this fails CI even when `openapi-check` passes.
- `docs/public/code-execution-openapi.yaml` is the exception in that directory: it is
  **hand-maintained**, not generated. It specifies a backend Otari calls, so no app here
  serves those paths and `generate_openapi.py` neither reads nor writes it. Edit it
  together with `docs/code-execution-protocol.md`, which is normative for the semantics a
  schema cannot carry; `tests/unit/test_code_execution_contract.py` fails when the two
  disagree, and `scripts/check_code_execution_conformance.py` checks a live backend
  against it.
- `CHANGELOG.md` and the GitHub Release body are generated from Conventional
  Commits by git-cliff (`cliff.toml`) at release time, not per-PR. Because PRs are
  squash-merged, the PR title is what git-cliff parses; `otari-pr-title.yml`
  enforces a conventional title. Visibility rules live in `RELEASE.md`
  ("Changelog visibility"). Do not hand-edit `CHANGELOG.md`; the release
  workflows regenerate it.
- The dashboard's route tree (`web/src/routeTree.gen.ts`) **is** committed, unlike the
  bundle below: `npm run typecheck` runs `tsc` alone and never invokes Vite, so without
  it in the tree a type-check of a fresh clone fails on a missing module. Any
  `npm test` or `npm run build` regenerates it from `web/src/routes/`, so adding a
  route and not committing the regenerated file shows up as a dirty working tree.
- The dashboard bundle (`src/gateway/static/dashboard/`) is **not** committed; it is
  gitignored and built on demand (`make dashboard`), and the Docker image builds it in
  its own Node stage. Vite content-hashes every asset filename, so committing it made
  any two branches touching `web/src` conflict by construction. Nothing to rebuild or
  commit after a `web/src` or `docs/dashboard.md` change; run `make dashboard` only when
  you need to *see* the dashboard locally or to build a wheel that ships it. See
  [web/AGENTS.md](web/AGENTS.md).

## Repository Conventions
- Use `TYPE_CHECKING` for type-only imports when helpful (`routes/_helpers.py`).
- Avoid adding comments unless logic is non-obvious; keep docstrings concise and meaningful for public functions/classes.
- Preserve security posture: do not leak internals in public error responses.
- Service-specific exceptions live alongside their service modules in `src/gateway/services/` (e.g. `UnsafeURLError`, `GuardrailsNotReachableError`).
- Do not log secrets, tokens, or raw API keys (bootstrap exception is intentional one-time behavior).
- CI runs Python 3.14 (`.github/workflows/otari-tests.yml`), matching the Docker image; the package still supports 3.13+ (`requires-python = ">=3.13"`).

## Change Validation Checklist
- If you touched API routes or schemas, run relevant integration tests first.
- If you touched DB models/repositories, run related integration tests and migration paths.
- If you touched config loading, run config/env tests in `tests/integration`.
- If you touched CLI behavior, run `tests/unit/test_gateway_cli.py`.
- If you touched auth headers or key handling, run key-management and auth-related tests.
- If OpenAPI-affecting code changed, regenerate and commit **both** generated
  artifacts, then verify with `make openapi-check` **and** `make postman-check`.
  A route docstring counts as OpenAPI-affecting: its text lands in the spec and
  in the Postman collection.

## Writing style

- Avoid em dashes and double hyphens (`--`) used as separators in prose
  (README, docs, doc comments, commit messages, PR descriptions). Use commas,
  semicolons, colons, parentheses, or periods, or rephrase. This does not apply
  to code (for example CLI flags like `--all`) or en-dash numeric ranges like `3–4`.
- Spell prose in **US English** (`behavior`, `recognize`, `serialize`, `catalog`,
  `labeled`, `license`, `color`). This covers docs, READMEs, comments,
  docstrings, commit messages, and user-visible UI copy, which is what the
  dashboard already uses (`Default pricing catalog`) and what otari.ai's own nav
  says (`Organization`).
- Three things keep whatever spelling they already have, because they are not
  ours to respell: an identifier or attribute borrowed from an external API
  (`aria-labelledby`, `asyncio.CancelledError`, GitHub's `cancelled` job
  conclusion), a value that travels on the wire or into a database, and a
  third-party product's own name. `cancelled` is therefore left alone repo-wide:
  it names an asyncio method and a CI conclusion far more often than it appears
  as prose, and splitting the spelling by context would read as a typo either way.

## Notes for Agents
- Prefer minimal, targeted edits over broad refactors.
- Maintain import order and existing typing style in touched files.
- Preserve security-relevant behavior (header parsing, auth checks, error detail boundaries).
- Keep test additions close to changed behavior (unit for pure logic, integration for route/database behavior).
