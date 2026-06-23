# Otari

Otari is an OpenAI-compatible LLM gateway you run yourself. Put one endpoint in front of 40+ providers, then manage API keys, enforce budgets, and track usage in one place. It's the proxy server at the heart of [otari.ai](https://otari.ai), open source and yours to deploy.

Your applications talk to Otari instead of directly to providers. Otari authenticates each request, enforces budgets before a call runs, resolves the right provider credential, and logs every request. Provider keys stay in one place, and usage is tracked across every model and app.

## Two ways to run it

- **Standalone** — Otari manages everything locally: its own database, your provider credentials, keys, budgets, and usage. Nothing leaves your environment. This is the default and the place to start.
- **Connected to otari.ai** — Otari delegates provider routing, authentication, and usage tracking to the platform, adding multi-provider fallback. Enabled by setting one environment variable.

See [Modes](modes.md) for the full comparison.

## New here?

Start with the [Quickstart](quickstart.md). It gets Otari running locally and walks you through your first request in about five minutes.

## Browse the docs

- [Quickstart](quickstart.md) — get running and make your first request.
- [Deployment](deployment.md) — hybrid mode, optional services, and deployment-specific configuration.
- [Modes](modes.md) — standalone vs hybrid, and what changes between them.
- [Built-in tools](tools.md) — sandboxed code execution and web search Otari runs itself.
- [Files](files.md) — file uploads and document understanding for local models.
- [Guardrails](guardrails.md) — request-level checks like prompt-injection detection.
- [Configuration](configuration.md) — the full config file and environment variable reference.
- [API reference](api-reference.md) — every endpoint, with auth and availability per mode.
- [Use with Claude Code](use-with-claude-code.md) — point the Claude Code CLI at Otari.
- [Use with opencode](use-with-opencode.md) — point the opencode CLI at Otari.
- [Supported models](supported-models.md) — providers, model format, and capabilities.
- [Hybrid-mode protocol](hybrid-mode-protocol.md) — Otari/platform wire contract, for platform builders.
- [SDK compatibility](sdk-compatibility.md) — how the language SDKs are released and which SDK version works with which Otari version.
