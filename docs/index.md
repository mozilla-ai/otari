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

The docs are grouped by what you need from them: a first lesson, the steps for
a task, the facts about a part of Otari, or the ideas behind it.

### Getting started

- [Quickstart](quickstart.md): run Otari with Docker Compose, connect OpenAI, and send your first request.

### How-to guides

Steps for one task, for a reader who knows what they want done.

- [Deployment](deployment.md): run Otari in production with Docker, Railway, or connected to otari.ai.
- [Connect OpenAI](providers/openai.md) and [Connect Gemini](providers/gemini.md): add a provider and route your first request to it.
- [Use with a ChatGPT subscription](chatgpt-subscription.md): route Otari at ChatGPT Plus/Pro models through a local Codex-OAuth proxy.
- [Track spend per end user](end-users.md): give each end user of one application its own spend, budget and limits with a service key.
- [Workload identity](workload-identity.md): a backend on Google Cloud authenticates with its platform identity token instead of a key's secret.
- [Inference Guardrails](guardrails.md): check requests and responses with a guardrails service, per request or for a whole organization.
- [Agent Guardrails](agent-guardrails.md): repository-owned rules checked against what a
  coding agent actually did to the working tree. `otari hook` runs beside Claude Code or
  Codex and evaluates them locally, with no server and no credential.
- [Importing external usage](external-usage.md): bring subscription-backed usage (Claude Code, Codex, any OTLP app) into your analytics.
- [Move config.yml providers to hosted providers](migrate-providers.md): upgrade a deployment from the deprecated `providers:` block.

### Integrations

Point an agent or a framework at Otari.

- [Use with Claude Code](use-with-claude-code.md): point the Claude Code CLI at Otari.
- [Use with Codex](use-with-codex.md): route the Codex CLI through Otari over the Responses API, or import its usage without routing.
- [Use with opencode](use-with-opencode.md): point the opencode CLI at Otari.
- [Use with Pydantic AI](use-with-pydantic-ai.md), [Use with LangChain](use-with-langchain.md) and [Use with LangGraph](use-with-langgraph.md): point an agent framework's model at Otari, with tools, structured output and streaming.

### Reference

The facts about each part of Otari: fields, settings, endpoints, codes and limits.

- [API reference](api-reference.md): authentication, mode availability, error codes, and a page per endpoint group.
- [Configuration](configuration.md): configuration sources, precedence, and every setting.
- [Admin dashboard](dashboard.md): sign-in, setup, and what each page shows.
- [Models](models.md): selectors, providers, discovery, aliases, and capabilities.
- [Hosted providers and BYO credentials](hosted-providers.md): which credential serves a request, hosted providers, pricing, discovery, and deployment-paid billing.
- [Provider endpoints](provider-endpoints.md): model servers a workspace or a user brings, reached by name and kept out of budgets.
- [Routing policies](routing.md): failover, weighted, priority and learned routing, and mandatory guardrails.
- [Built-in tools](tools.md): sandboxed code execution, web fetch and web search Otari runs itself.
- [MCP](mcp.md): connect MCP servers to chat, messages, and responses requests,
  or drive them from your own application through the caller-orchestrated endpoints.
- [Files](files.md): file uploads and document understanding for local models.
- [Health checks](health-checks.md): a scheduled check of every model the deployment serves, and email when one stops answering.
- [Agent Guardrails reference](agent-guardrails-reference.md): every gate type and field, the
  command-line options, and what each harness can see at each event.
- [SDK compatibility](sdk-compatibility.md): which SDK version works with which Otari version.
- [Hybrid-mode protocol](hybrid-mode-protocol.md): the Otari/platform wire contract, for building a platform that Otari connects to.
- [Code-execution protocol](code-execution-protocol.md): the Otari/sandbox contract, for building a code-execution backend that Otari dispatches to. Its machine-readable form is [`public/code-execution-openapi.yaml`](public/code-execution-openapi.yaml).

### Learn

How Otari works and why.

- [Runtime modes](modes.md): standalone, hosted, and hybrid deployments, and who runs each.
- [Access control](access-control.md): identities, organizations, workspaces, keys, and budgets.
- [Agent traces](traces.md): each agent session as turns, LLM calls and tool calls, with their timing and cost.

### For contributors

- [Architecture](../ARCHITECTURE.md): the two-plane model and the extension seam (ports, adapters, and capability lines) that mark what Otari's core ships versus what an overlay can add.
- [Backend domains](../DOMAINS.md): what each backend domain owns, and the layer shape each domain moves toward.
- [Load test](../loadtest/README.md): compares two builds under load on one machine, route by route, and fails when the second got slower or runs more database statements per request. CI runs it on a PR against its base.
