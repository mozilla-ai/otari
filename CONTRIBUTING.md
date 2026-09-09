# Contributing to Otari

## Before you start

- Search [existing issues](https://github.com/mozilla-ai/otari/issues) and [open PRs](https://github.com/mozilla-ai/otari/pulls) to avoid duplicates.
- For significant changes (new endpoints, auth changes, breaking config changes, new dependencies), open an issue first to align on approach.
- All contributors must follow Mozilla's [Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/).

## Dev setup

**Prerequisites:** Python 3.13+, `uv`, Docker (for integration tests).

```bash
git clone https://github.com/mozilla-ai/otari
cd otari
uv venv && source .venv/bin/activate
uv sync --dev
cp config.example.yml config.yml
# Set master_key and at least one provider. If you don't have a local Postgres,
# change database_url to: sqlite+aiosqlite:///./otari.db
uv run otari serve --config config.yml
```

For hot reload: `make dev`.

## Making changes

Branch naming: `feature/`, `fix/`, `docs/`, `refactor/`.

```bash
git checkout -b fix/your-description
```

After making changes:

```bash
make lint        # architecture check + ruff
make typecheck   # mypy --strict
make test        # unit + integration
```

Run a single test: `uv run pytest tests/unit/test_gateway_cli.py -v`

If you changed any API routes or schemas, regenerate both generated artifacts and commit them:

```bash
uv run python scripts/generate_openapi.py
uv run python scripts/generate_postman.py
make openapi-check
make postman-check
```

## Dependencies

Most dependencies are floored (`>=`) rather than pinned, and `uv.lock` is committed
because CI and the Docker image both install from it with `--frozen`. Two mechanisms
keep it current, and they cover different things:

- **Dependabot** (`.github/dependabot.yml`) opens weekly version-update PRs for the
  `uv` and `github-actions` ecosystems. Security updates are separate, need no
  config, and already cover every supported ecosystem here. The dashboard is not
  covered by either right now; see #1001.
- **`.github/workflows/otari-lock-refresh.yml`** re-resolves `uv.lock` weekly against
  the newest versions the existing constraints already allow, which is the case
  Dependabot does not open PRs for. A floored dependency can otherwise stay at
  whatever version was current the day it was first locked.

To do either by hand:

```bash
uv lock --upgrade --dry-run                    # what would move, and to where
uv lock --upgrade-package genai-prices         # refresh one package
uv lock --upgrade                              # refresh everything
```

A change to dependency resolution also owes the OSS-edition smoke gate, which boots
the packaged CLI with no dev dependencies:

```bash
uv run --frozen --no-dev python scripts/oss_edition_smoke.py
```

## Tests

- New features need tests covering the happy path and error cases.
- Unit tests for pure logic (`tests/unit/`), integration tests for route or database behavior (`tests/integration/`).
- Integration tests require PostgreSQL. They start Testcontainers by default;
  without Docker, set `TEST_DATABASE_URL` to a test server where workers may
  create and drop databases. Never use production, and do not run concurrent
  suites against the same server URL because their worker database names collide.

## Pull requests

- PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/); CI enforces this.
- PRs are squash-merged, so the title is what ends up in the changelog.
- Keep diffs focused; avoid unrelated refactors in the same PR.
- Do not hand-edit `CHANGELOG.md`; it is regenerated from commit history at release time.
- The PR description must keep the **PR Type**, **Checklist**, and **AI Usage** sections from the [PR template](https://github.com/mozilla-ai/otari/blob/main/.github/pull_request_template.md). CI checks for these sections and will auto-close PRs that are missing them after 24 hours.

## Questions?

- [GitHub Discussions](https://github.com/mozilla-ai/otari/discussions) for design questions.
- [Discord](https://discord.gg/ZfZPfTdtSe) for quick questions.
- Tag `@maintainers` in an issue if you need guidance.

**License:** By contributing, you agree your contributions will be licensed under Apache 2.0 (see [LICENSE](LICENSE)).
