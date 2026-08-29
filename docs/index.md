# Otari

Otari is an OpenAI-compatible LLM gateway. It routes to 40+ providers, manages
API keys, enforces budgets, and records usage. It is also the gateway at the
heart of [otari.ai](https://otari.ai).

## Runtime modes

- **Standalone** serves management and inference with local storage.
- **Hosted** serves a multi-tenant control plane while inference runs elsewhere.
- **Hybrid** serves inference using a connected control plane such as otari.ai.

See [Modes](modes.md) for the full comparison.

## New here?

Start with the [Quickstart](quickstart.md).

## Browse the docs

The docs are grouped by what you are trying to do.

### Start here

- [Quickstart](quickstart.md): get running and make your first request.
- [Modes](modes.md): standalone, hosted, and hybrid deployments.

### For operators

Running and managing a gateway.

- [Deployment](deployment.md): Docker, Render, Railway, hybrid mode, and optional services.
- [Configuration](configuration.md): configuration sources, precedence, and common settings.
- [Admin dashboard](dashboard.md): sign-in, setup, and management surfaces.
- [Access control](access-control.md): identities, organizations, workspaces, keys, and budgets.
- [Models](models.md): selectors, providers, discovery, aliases, and capabilities.
- [Routing policies](routing.md): failover, conditions, weighted and learned routing, and mandatory guardrails.
- [OpenAI provider guide](providers/openai.md): configure OpenAI and route your first request through Otari.

### For integrators

Calling the gateway from your own code.

- [API reference](api-reference.md): authentication, mode availability, and links to generated schemas.
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
- [Code-execution protocol](code-execution-protocol.md): the Otari/sandbox contract, for building a code-execution backend that Otari dispatches to. Its machine-readable form is [`public/code-execution-openapi.yaml`](public/code-execution-openapi.yaml).

### For contributors

- [Architecture](../ARCHITECTURE.md): the two-plane model and the extension seam (ports, adapters, and capability lines) that mark what Otari's core ships versus what an overlay can add.
