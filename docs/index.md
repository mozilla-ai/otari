# Otari

Otari is an OpenAI-compatible LLM gateway you run yourself. Put one endpoint in front of 40+ providers, then manage API keys, enforce budgets, and track usage in one place. It's the proxy server at the heart of [otari.ai](https://otari.ai), open source and yours to deploy.

Your applications talk to Otari instead of directly to providers. Otari authenticates each request, enforces budgets before a call runs, resolves the right provider credential, and logs every request. Provider keys stay in one place, and usage is tracked across every model and app.

## Two ways to run it

- **Standalone**: Otari manages everything locally: its own database, your provider credentials, keys, budgets, and usage. Nothing leaves your environment. This is the default and the place to start.
- **Connected to otari.ai**: Otari delegates provider routing, authentication, and usage tracking to the platform, adding multi-provider fallback. Enabled by setting one environment variable.

See [Modes](modes.md) for the full comparison.

## New here?

Start with the [Quickstart](quickstart.md). It gets Otari running locally and walks you through your first request in about five minutes.

## Browse the docs

The docs are grouped by what you are trying to do.

### Start here

- [Quickstart](quickstart.md): get running and make your first request.
- [Modes](modes.md): standalone vs hybrid, and what changes between them.

### For operators

Running and managing a gateway.

- [Deployment](deployment.md): Docker, Render, Railway, hybrid mode, and optional services.
- [Configuration](configuration.md): the full config file and environment variable reference.
- [Admin dashboard](dashboard.md): the operator dashboard, covering the two-key model, a first-run walkthrough, and a page-by-page reference.
- [Access control](access-control.md): users, API keys, and budgets, with the management endpoints that drive them.
- [Supported models](models.md): providers, model format, aliases, and capabilities.
- [Routing policies](routing.md): failover, budget-based tier-down, a weighted split that
  load balances across providers, a learned router that sends easy prompts to cheaper
  models, and enforced guardrails, all behind one model name.
- [OpenAI provider guide](providers/openai.md): configure OpenAI and route your first request through Otari.

### For integrators

Calling the gateway from your own code.

- [API reference](api-reference.md): every endpoint, with auth and availability per mode.
- [Built-in tools](tools.md): sandboxed code execution and web search Otari runs itself.
- [MCP](mcp.md): connect MCP servers to chat, messages, and responses requests.
- [Files](files.md): file uploads and document understanding for local models.
- [Guardrails](guardrails.md): request-level checks like prompt-injection detection.
- [Use with Claude Code](use-with-claude-code.md): point the Claude Code CLI at Otari.
- [Use with Codex](use-with-codex.md): route the Codex CLI through Otari over the Responses API, or import its usage without routing.
- [Use with opencode](use-with-opencode.md): point the opencode CLI at Otari.
- [Use with a ChatGPT subscription](chatgpt-subscription.md): route Otari at ChatGPT Plus/Pro models through a local Codex-OAuth proxy.
- [Importing external usage](external-usage.md): bring subscription-backed usage (Claude Code, Codex, any OTLP app) into your analytics.
- [SDK compatibility](sdk-compatibility.md): how the language SDKs are released and which SDK version works with which Otari version.

### For platform builders

- [Hybrid-mode protocol](hybrid-mode-protocol.md): the Otari/platform wire contract, for building a platform that Otari connects to.
- [Code-execution protocol](code-execution-protocol.md): the Otari/sandbox contract, for building a code-execution backend that Otari dispatches to.

### For contributors

- [Architecture](../ARCHITECTURE.md): the two-plane model and the extension seam (ports, adapters, and capability lines) that mark what Otari's core ships versus what an overlay can add.
