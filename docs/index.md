# Otari Gateway

Otari Gateway is an OpenAI-compatible LLM gateway you run yourself. Put one endpoint in front of 40+ providers, then manage API keys, enforce budgets, and track usage in one place. It's the proxy server at the heart of [otari.ai](https://otari.ai), open source and yours to deploy.

Your applications talk to the gateway instead of directly to providers. The gateway authenticates each request, enforces budgets before a call runs, resolves the right provider credential, and logs every request. Provider keys stay in one place, and usage is tracked across every model and app.

## Two ways to run it

- **Standalone** — the gateway manages everything locally: its own database, your provider credentials, keys, budgets, and usage. Nothing leaves your environment. This is the default and the place to start.
- **Connected to otari.ai** — the gateway delegates provider routing, authentication, and usage tracking to the platform, adding multi-provider fallback. Enabled by setting one environment variable.

See [Modes](modes.md) for the full comparison.

## New here?

Start with the [Quickstart](quickstart.md). It gets the gateway running locally and walks you through your first request in about five minutes.

## Browse the docs

- [Quickstart](quickstart.md) — get running and make your first request.
- [Deployment](deployment.md) — Docker setup, production configuration, connecting to otari.ai.
- [Modes](modes.md) — standalone vs connected, and what changes between them.
- [Built-in tools](tools.md) — sandboxed code execution and web search the gateway runs itself.
- [Guardrails](guardrails.md) — request-level checks like prompt-injection detection.
- [Configuration](configuration.md) — the full config file and environment variable reference.
- [API reference](api-reference.md) — every endpoint, with auth and availability per mode.
- [Supported models](supported-models.md) — providers, model format, and capabilities.
- [Platform protocol](platform-protocol.md) — the gateway/platform wire contract, for platform builders.