# Agent framework end-to-end tests

Tests `pydantic-ai-otari` and `langchain-otari` (the two packages beside this
directory) from this checkout, and runs every Python block on [Use with Pydantic AI](../../docs/use-with-pydantic-ai.md),
[Use with LangChain](../../docs/use-with-langchain.md) and
[Use with LangGraph](../../docs/use-with-langgraph.md) against a real Otari.

`conftest.py` boots `otari serve` from the repository's environment on a throwaway
SQLite database, in front of `mock_provider.py`, a scripted OpenAI-compatible
provider that calls offered tools, answers JSON schema requests and streams. Each
block then runs in a fresh interpreter with only `OTARI_BASE_URL` and
`OTARI_API_KEY` set, which is what the pages tell a reader to export.

```bash
make test-frameworks
```

That runs the packages' own tests too. The `floors` job in the workflow also runs
them on Python 3.10 against the oldest framework releases the packages declare.

No Docker and no provider key are needed. CI runs it from
`.github/workflows/otari-frameworks.yml`.

This directory has its own `pyproject.toml` and `uv.lock`, installs the two
packages from their directories, and is not a workspace member, so the
frameworks never enter the gateway's lock. Releasing a package is described in
[RELEASE.md](../../RELEASE.md#integration-package-releases).

## Adding a block to a page

Every ```` ```python ```` block on those pages must run on its own and print
something. A block that calls a tool or asks for structured output needs the
mock to know the answer: add the tool's arguments to `TOOL_ARGUMENTS` in
`mock_provider.py`.
