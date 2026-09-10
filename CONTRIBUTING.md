# Contributing to Otari

## Before you start

- Search [existing issues](https://github.com/mozilla-ai/otari/issues) and [open PRs](https://github.com/mozilla-ai/otari/pulls) to avoid duplicates.
- For significant changes (new endpoints, auth changes, breaking config changes, new dependencies), open an issue first to align on approach.
- All contributors must follow Mozilla's [Community Participation Guidelines](https://www.mozilla.org/about/governance/policies/participation/).

## Is this an Otari change or an any-llm change?

Otari dispatches every provider call through
[any-llm](https://github.com/mozilla-ai/any-llm). Much of what looks like an
Otari bug is a provider integration bug, and fixing it here means patching
around the SDK instead of fixing it for every any-llm user. Work out which repo
owns the change before writing code.

**The seam:** any-llm owns talking to a provider. Otari owns deciding whether a
call happens, with which credentials, and what it cost.

**The test:** reproduce it without Otari. If a direct any-llm call shows the same
behavior, it belongs upstream.

```python
# pip install "any-llm-sdk[all]"
import asyncio
from any_llm import acompletion

print(asyncio.run(acompletion(
    model="openai:gpt-4o-mini",   # your provider:model
    messages=[{"role": "user", "content": "hi"}],
    # plus the param or option you think Otari is mishandling
)))
```

**File it in [any-llm](https://github.com/mozilla-ai/any-llm/issues) if:**

- a provider is unsupported, or needs a new implementation
- a request param is dropped or mistranslated on the way to a provider SDK
- a provider's response or stream chunks are not normalized to the OpenAI shape
- `list_models` behavior, a provider capability flag, a credential env var name, or a default base URL is wrong
- a provider SDK upgrade breaks the call itself

**File it here if it touches:**

- keys, users, orgs, workspaces, budgets, usage records, or pricing
- route schemas, the OpenAPI spec, or the Anthropic and Responses envelopes
- routing, fallback across attempts, or routing memory
- `config.yml` layering, the `providers:` block, or provider instances and aliases
- the dashboard, built-in tools, the MCP loop, guardrails, or hybrid mode

Two things that look upstream but are ours: the per-provider setup guides in
`docs/providers/`, and how a provider error becomes a status code and a
sanitized message.

### When the fix is upstream but Otari cannot wait

Otari sometimes has to carry a workaround while any-llm catches up. That is a
legitimate PR here, with conditions: keep the shim as small as the problem
allows, comment it with a link to the any-llm issue, and open an Otari issue to
remove it once the SDK pin moves. `service_tier` in
`src/gateway/api/routes/chat.py` is the worked example. File the upstream issue
first; a shim with no upstream issue is permanent by accident.

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
