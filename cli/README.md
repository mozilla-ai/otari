# otari-agent

The laptop-side half of Otari: the `otari` command a developer runs next to a
coding agent. It talks HTTP to a running gateway and needs none of the server.

| Command | What it does |
|---|---|
| `otari hook` | The callback a coding agent's hook calls. Evaluates the repository's `.otari-guardrails.yml` in process; see `docs/agent-guardrails.md`. |
| `otari hook setup` | Registers that callback in the agent's own settings file. |
| `otari guardrails generate` | Proposes `.otari-guardrails.yml` gates from the repository's own `AGENTS.md` or `CLAUDE.md`, one at a time. |
| `otari import claude-code` | Backfills Claude Code usage from local transcripts into a gateway. |
| `otari --version` | The version this distribution was built as, or `OTARI_VERSION` where a deployment sets it (the Docker image does). |

## Install

```bash
brew install mozilla-ai/tap/otari
```

Every release publishes the formula to `mozilla-ai/homebrew-tap`
(`otari-homebrew.yml`), together with the sdist and its pinned requirements as
Release assets. Without Homebrew, install the sdist from a Release:

```bash
uv tool install https://github.com/mozilla-ai/otari/releases/download/vX.Y.Z/otari_agent-X.Y.Z.tar.gz
```

## Layout

This directory is a uv workspace member (`pyproject.toml` at the repository
root lists it). Distribution `otari-agent`, import package `otari_agent`:
the names `otari` and `otari-cli` on PyPI belong to other Mozilla AI projects.

It owns the `otari` console script. The gateway (`../src/gateway`) depends on
this package for the Agent Guardrails evaluator (`otari_agent.domain`) and, when
both are installed (Docker, a development checkout), attaches its server
commands (`serve`, `migrate`, ...) to the same `otari` group through
`gateway.cli.register`. On its own, `otari --help` lists only the commands
above.

Dependencies are click, httpx, python-dotenv and pyyaml, all pure Python.
`scripts/check_architecture.py` refuses an import of the gateway or of the
server stack from here, and `tests/unit/test_otari_agent_cli.py` checks that
importing the CLI loads none of it.

## Building it alone

```bash
uv build --package otari-agent --sdist
uv export --frozen --package otari-agent --no-dev --no-emit-workspace -o requirements.txt
```

The version in `src/otari_agent/__init__.py` is `0.0.0` in the tree; a release
stamps the tag into it before building. The Docker image installs from the tree
unstamped and sets `OTARI_VERSION` to its tag instead, which `otari --version`
reports whenever that variable is set; the gateway reads the same one, in
`src/gateway/version.py`.
