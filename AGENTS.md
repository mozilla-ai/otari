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
- **Dashboard (`web/`)** → [web/AGENTS.md](web/AGENTS.md) (auth/session model, runtime provider management, build + bundled guide, PWA, serving) and [.github/skills/frontend-standards/SKILL.md](.github/skills/frontend-standards/SKILL.md): HeroUI v3, the semantic design tokens rehomed from `otari-ai/frontend`, TanStack Query patterns, component architecture, responsiveness, layout stability, performance under the React Compiler, and the three test suites. Its topic guides load one at a time; the SKILL indexes them.
- **Reviewing a PR or a diff** → [.github/skills/review/SKILL.md](.github/skills/review/SKILL.md): the procedure (which scoped guidance to load for which paths), the repo-specific gates that have broken a PR here before, and how findings are expressed.
- **Taking an issue to a ready PR** → [.github/skills/pr-cycle/SKILL.md](.github/skills/pr-cycle/SKILL.md): the implement, self-review, open, request-review, fix loop, including which checks and generated artifacts a PR owes, how to read back the inline comments of the two bots that review here (Copilot and CodeRabbit), and what the repo's squash-merge and its `protect-main` ruleset mean for merging.
- **Reviewing a change** → the path-scoped files in [.github/instructions/](.github/instructions/) auto-apply during Copilot reviews (they carry `applyTo` globs): [security-review](.github/instructions/security-review.instructions.md) (budget/tenant isolation, auth, SSRF, prompt injection) and [performance-review](.github/instructions/performance-review.instructions.md) (N+1, indexes, pagination limits, transaction atomicity) for `src/gateway/`, and [frontend-standards](.github/instructions/frontend-standards.instructions.md) (HeroUI v3, design tokens, TanStack Query, mobile, layering, tests) for `web/`.

The `.claude/skills` directory symlinks to `.github/skills`, so the same skills are available to Claude and to GitHub Copilot from one source.

**Adding guidance: pick the narrowest layer that covers it, and link rather than restate.**
This file is loaded every session, so it carries only what applies repo-wide plus the pointers
above. A scoped `AGENTS.md` (`web/`, `src/gateway/`) describes the structure of its directory,
loaded when you work there. A skill carries the house style for writing code inside that
structure, loaded on demand. `.github/instructions/` is the one place restatement is expected,
because it loads for a different reader (Copilot review) that never sees the rest. Everywhere
else, a fact told in two layers is a fact that will go stale in one of them, and the stale copy
is the one someone believes.

## Architecture (Big Picture)
For the open-core OSS/enterprise seam (ports, adapters, the capability lines, and the rules for keeping the boundary), see [ARCHITECTURE.md](ARCHITECTURE.md). It is a north-star document describing the intended architecture, so ground current-state work in `src/gateway/`.

### Runtime modes
- Mode is derived when `OTARI_MODE` is unset, and honored when set: `GatewayConfig.is_hybrid_mode` / `effective_mode` (`src/gateway/core/config.py`) return `hybrid` when the config field `mode` is `hybrid` (legacy `platform`) or, when `mode` is unset, when the platform token (`OTARI_AI_TOKEN`) is set; otherwise `standalone`. Startup validation (`validate_mode_selection`) rejects the conflicting combinations: `OTARI_MODE=hybrid` (legacy value `platform`) without a token, and `OTARI_MODE=standalone` or `OTARI_MODE=hosted` with a token set (the token would otherwise silently select hybrid). The token is resolved once at config-load time (cached on the config), not re-read from `os.getenv` on every access.
- **Standalone**: provider credentials come from the `providers:` block in `config.yml`; users/keys/budgets/usage live in the local DB. All routers are registered.
- **Hosted** (`OTARI_MODE=hosted`) is standalone's multi-tenant variant, not a third runtime: same database, same management API, same sign-in, and `is_hybrid_mode` is false. Two things differ. What `GET /v1/bootstrap` publishes, `deployment_type: "hosted"` and `HOSTED_SURFACES` rather than `STANDALONE_SURFACES`, which drops the process-global `providers` page (`provider_credentials` is keyed on instance name alone, so one row serves every tenant) and adds the organization-scoped `organization_providers` one. Hiding a page is not a guard over the table; #818 tracks that. And the data plane: `_register_core_routers` mounts the inference routers only when `not is_hosted_mode`, and `api/routes/hosted_mode.py` answers those prefixes with a 404 naming the reason (and the `data_plane_url` to use, where one is set), because inference belongs on a hybrid gateway whose usage report is what debits the wallet (#822). `/v1/models` stays mounted, since discovery is not dispatch.
- **Hybrid**: per-request provider credentials are resolved from the platform service (otari.ai); local DB/user/budget management is skipped and usage is reported upstream. `register_routers()` (`src/gateway/api/main.py`) only mounts `chat`, `messages`, `responses`, `health`, and `bootstrap`; management routers (keys/users/budgets/pricing/usage/etc.) are standalone-only. `bootstrap` (`GET /v1/bootstrap`) is mounted in both modes and is unauthenticated, because it is how a browser learns which mode it reached; see [web/AGENTS.md](web/AGENTS.md).
- Hybrid mode spans two trust contexts that this codebase treats identically: a gateway someone self-hosts against otari.ai using a workspace's own (BYO) provider keys, and the gateway mozilla.ai operates as part of otari.ai, which additionally serves mozilla.ai-managed models. The managed-vs-BYO boundary (platform-owned upstream credentials are returned only to mozilla.ai's gateway, never to a self-hosted one) is enforced on the platform side (otari-ai), not here. User-facing explanation lives in `docs/modes.md`; the wire contract in `docs/hybrid-mode-protocol.md`.

The per-request flow (auth → budget → dispatch → reconciliation) spans several files and is documented in [src/gateway/AGENTS.md](src/gateway/AGENTS.md). Read it before changing request behavior.

## Lint / Typecheck
- Run lint checks with `make lint`; it runs the architecture check and then Ruff. **Ruff alone is not equivalent.**
- The architecture check (`scripts/check_architecture.py`, also `make check-architecture`) enforces the `src/gateway/` layer rules: services must not import the API layer, repositories must not import services or the API layer, API routes must not import `sqlalchemy.orm`, and repository modules end in `_repository.py`.
- **`make lint` does not touch the dashboard.** `pnpm --dir web run lint` is its counterpart (Biome: formatting, recommended rules, and the `web/src/` layer boundaries), run separately in CI. See [web/AGENTS.md](web/AGENTS.md) for what those boundaries are and why the config mirrors `otari-ai/frontend`.
- If introducing a formatter/linter, keep changes in a separate PR unless requested.

## Test Notes
- There is no global rerun policy in `pytest.ini`. A blanket `reruns` would let a test that
  passes one time in several count as green, hiding flakiness and ordering
  bugs suite-wide. Mark a genuinely flaky test with
  `@pytest.mark.flaky(reruns=...)` (from `pytest-rerunfailures`) and say why,
  rather than reintroducing a global retry.
- Integration tests need PostgreSQL: `TEST_DATABASE_URL` if set, otherwise a Testcontainers `postgres:17`, so without Docker the suite cannot start. Whichever it is, it is a *server* URL: each xdist worker creates a database of its own on it (`postgres` becomes `postgres_gw0`, and so on) and drops it at the end of the session, so the credentials it points at need `CREATE DATABASE` and a `postgres` database to connect through. Two suites must not share one server URL: they pick the same worker database names and drop each other's database mid-run. SQLite is not a fallback even though `_to_async_url` accepts one: none of that is available there. With no Docker, point `TEST_DATABASE_URL` at any reachable PostgreSQL instead.
- The schema is built once per worker, not once per test. `tests/integration/conftest.py` runs the migration chain on first use and then returns each test a clean database by truncating it and restoring the migration-seeded rows (`clean_database`, autouse). A fixture that needs a client on a config of its own gets it from `build_test_client`, which is why no test module drops tables any more: dropping them would take the schema out from under every later test on that worker.
- The OSS-edition smoke gate (`scripts/oss_edition_smoke.py`, run by
  `otari-oss-edition.yml` on any PR touching the app, the migrations, or dependency
  resolution) boots the packaged CLI as a subprocess with no overlay
  bootstrap and no platform token, then walks health, key creation, a stored BYO
  provider credential, a fallback-routed completion against a mock provider, and
  the usage row. Run it locally with
  `uv run --frozen --no-dev python scripts/oss_edition_smoke.py`; it defaults to a
  throwaway SQLite file, so it needs no Docker, and `--database-url` points it at
  PostgreSQL as CI does. Keep it standard-library only and keep it running under
  `--no-dev`: that is what makes it able to catch a dev-only or enterprise-only
  import that reached an OSS code path, and a single third-party import in it
  (httpx, pyyaml) gives that up.
- Two tests assert the provider-error sanitization by making a real outbound call (`test_error_detail_leakage.py::test_provider_error_does_not_leak_details`, `test_streaming_error_event.py::test_streaming_creation_error_returns_http_error`). With no network egress the upstream fails differently and both report a status mismatch, so treat them as environment noise rather than a regression, and confirm a change against the rest of the suite.
- `tests/integration/test_mcp_dependency_ceiling.py::test_mcp_constraint_resolves_to_an_importable_version` also needs network egress, to install `mcp` fresh from PyPI into a throwaway venv. Unlike the two above, a missing egress here does not look like a status mismatch: it fails hard after burning both `@pytest.mark.flaky` reruns. It also skips outright (not fails) when `uv` is not on `PATH`.

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
- The dashboard has three more, and [web/AGENTS.md](web/AGENTS.md) owns them: its API client
  (`web/src/client/schema.ts`) and route tree (`web/src/routeTree.gen.ts`) are generated **and
  committed**, each with a CI drift check, while the bundle (`src/gateway/static/dashboard/`)
  is generated and **not** committed. A change under `web/src` therefore sometimes leaves a
  file to commit and never leaves a bundle to commit. Screenshot baselines are a fourth
  artifact that is deliberately neither: the suite runs on demand and its PNGs are gitignored
  while the dashboard is mid-migration.

## Repository Conventions
- Prefer minimal, targeted edits over broad refactors, and match the import order and typing style of the file you are in (`TYPE_CHECKING` for type-only imports where it helps, as in `routes/_helpers.py`).
- Add a comment only where the logic is not obvious; keep docstrings concise and meaningful on public functions and classes. Do not restate the code, narrate the change, or record what the code used to do: the commit message is where that belongs. Leave the comments around a change shorter than you found them: prune narration, repeated rationale, and implementation history as you touch them.
- Preserve security-relevant behavior: header parsing, auth checks, and the error-detail boundary. Do not leak internals in public error responses, and never log secrets, tokens, or raw API keys (the one-time bootstrap key print is the deliberate exception).
- Keep test additions next to the behavior they cover: unit for pure logic, integration for route or database behavior.
- CI runs Python 3.14 (`.github/workflows/otari-tests.yml`), matching the Docker image; the package still supports 3.13+ (`requires-python = ">=3.13"`).

## Change Validation Checklist
- If you touched API routes or schemas, run relevant integration tests first.
- If you touched DB models/repositories, run related integration tests and migration paths.
- If you touched config loading, run config/env tests in `tests/integration`.
- If you touched CLI behavior, run `tests/unit/test_gateway_cli.py`.
- If you touched auth headers or key handling, run key-management and auth-related tests.
- If OpenAPI-affecting code changed, including a route docstring, regenerate and commit **both**
  generated artifacts (see Generated Artifacts above).

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
