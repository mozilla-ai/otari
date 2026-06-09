# Contributing to Otari

Thanks for your interest in contributing. Otari is open source and we welcome contributions, whether that's reporting a bug, improving the docs, or sending a pull request.

## Before you start

For anything beyond a small fix, open a [GitHub issue](https://github.com/mozilla-ai/otari/issues) or start a [discussion](https://github.com/mozilla-ai/otari/discussions) first. It's the best way to coordinate, avoid duplicate work, and get early feedback on the approach before you invest time in a change. For small fixes (typos, obvious bugs, doc corrections) feel free to open a pull request directly.

## Development setup

You'll need [uv](https://docs.astral.sh/uv/) and Python 3.13+.

Clone the repo and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync --dev
```

Copy the example config and add at least a master key and one provider credential:

```bash
cp config.example.yml config.yml
```

Run the gateway:

```bash
uv run gateway serve --config config.yml
```

The API is now at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Hot reload

For local development with auto-reload and a `.env` file:

```bash
cp .env.example .env
make dev
```

## Tests and checks

Before opening a pull request, make sure tests and checks pass:

```bash
make test
make lint
make typecheck
```

To run a single test file:

```bash
uv run pytest tests/unit/test_gateway_cli.py -v
```

## Database migrations

If your change touches the database schema, generate and check migrations:

```bash
uv run gateway migrate --config config.yml
```

If you modify the API surface, regenerate the OpenAPI spec and confirm it's in sync:

```bash
uv run python scripts/generate_openapi.py --check
```

## Pull requests

- Keep changes focused. One logical change per pull request is easier to review and merge.
- Make sure `make test`, `make lint`, and `make typecheck` all pass.
- If you've changed behavior, update the relevant docs in `docs/` and the `README` where needed.
- If you've added or changed an endpoint, regenerate the OpenAPI spec.
- Write a clear PR description: what changed, why, and anything reviewers should look at closely.

CI runs tests, linting, typechecking, and a Docker build on every pull request. All checks need to pass before review.

## Reporting bugs

Open an [issue](https://github.com/mozilla-ai/otari/issues) with:

- What you expected to happen and what actually happened
- Steps to reproduce
- Your environment (how you're running the gateway, standalone or connected, and any relevant config with secrets removed)
- Relevant logs, with API keys and tokens redacted

## Questions

For questions, ideas, or anything that isn't a bug report, use [GitHub Discussions](https://github.com/mozilla-ai/otari/discussions).

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE) that covers this project.