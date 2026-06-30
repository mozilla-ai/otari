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
make lint        # ruff
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

## Tests

- New features need tests covering the happy path and error cases.
- Unit tests for pure logic (`tests/unit/`), integration tests for route or database behavior (`tests/integration/`).
- Integration tests require Docker (Testcontainers spins up Postgres automatically).

## Pull requests

- PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/) — the CI enforces this.
- PRs are squash-merged, so the title is what ends up in the changelog.
- Keep diffs focused; avoid unrelated refactors in the same PR.
- Do not hand-edit `CHANGELOG.md`; it is regenerated from commit history at release time.
- The PR description must keep the **PR Type**, **Checklist**, and **AI Usage** sections from the [PR template](https://github.com/mozilla-ai/otari/blob/main/.github/pull_request_template.md). CI checks for these sections and will auto-close PRs that are missing them after 24 hours.

## Questions?

- [GitHub Discussions](https://github.com/mozilla-ai/otari/discussions) for design questions.
- [Discord](https://discord.gg/ZfZPfTdtSe) for quick questions.
- Tag `@maintainers` in an issue if you need guidance.

**License:** By contributing, you agree your contributions will be licensed under Apache 2.0 (see [LICENSE](LICENSE)).
