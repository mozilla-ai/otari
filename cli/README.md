# otari-agent

The laptop-side half of Otari: the `otari` command a developer runs next to a
coding agent. It talks HTTP to a running gateway and needs none of the server.

| Command | What it does |
|---|---|
| `otari hook` | The callback a coding agent's hook calls. Composes the repository's guardrail files under `.otari/`, and your own under `~/.otari/`, and evaluates them in process; see `docs/agent-guardrails.md`. |
| `otari hook setup` | Registers that callback in the agent's own settings file. |
| `otari guardrails check` | Runs the repository's gates against a change, such as a pull request in CI; see `docs/agent-guardrails.md`. |
| `otari guardrails generate` | Proposes guardrail gates from the repository's own `AGENTS.md` or `CLAUDE.md`, one at a time. |
| `otari guardrails validate` | Checks the composed guardrail offline, and dry-runs it against a command or a path. |
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

Dependencies are click, httpx and pyyaml, all pure Python.
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

## This repository's own guardrail CI

Two workflows run `otari guardrails check` on every pull request to `main`,
following the steps in `docs/agent-guardrails.md` ("Checking a pull request in
CI"). Both take the CLI, the guardrail files and the verifier scripts from the
base commit, and both skip with a notice when the base commit's CLI has no
`otari guardrails check`, rather than take the command from the pull request.

- `.github/workflows/otari-guardrails-check.yml` runs the `path` and `verifier`
  gates on `pull_request`. It checks out the base commit and the pull request
  side by side, stages the pull request against its merge base, and runs
  `--type path --type verifier --guardrails-from <base>` in the pull request's
  tree. It holds no secret: a pull request can edit it, and a verifier runs the
  pull request's own scripts (such as `scripts/check_architecture.py`), so a
  required status check and review of the workflow and those scripts are what
  keep it honest.
- `.github/workflows/otari-guardrails-judges.yml` runs the `judge` gates on
  `pull_request_target`, so it runs as it is on `main` and a pull request cannot
  edit the job that holds the credential. It checks out only the base commit,
  fetches the pull request as Git objects, and runs
  `--base <base> --head <head> --type judge` with `OTARI_HOOK_MAX_JUDGES=20`.
  Its credential, `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_API_KEY`, comes from
  the `agent-guardrails` environment, which can restrict its deployment branches
  to `main`; without one, or on a pull request from a fork, the judges skip with
  a notice. Findings go to the job summary, and a failing judge starts one
  comment on the pull request, naming the commit it describes, which later runs
  edit. A finding fails the job so it shows in the checks, but the job is not a
  required check, so it does not block the merge.
