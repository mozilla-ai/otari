# Otari

Otari is an open-source, OpenAI-compatible LLM gateway. Your applications send
requests to one endpoint with one key, and Otari routes them to 40+ providers,
enforces budgets, applies guardrails, runs tools, and records usage and traces.

You can use it two ways, and these docs cover both:

- **[otari.ai](otari-ai/overview.md)**: Otari run by Mozilla.ai as a service.
  Sign up and send a request; there is nothing to deploy.
- **Self-hosted**: run the same gateway and dashboard on your own
  infrastructure, as the [Self-hosting](self-hosting/deploy.md) section describes.

Everything else, from workspaces and budgets to routing, tools and integrations,
works the same on both. Where a step differs, the page says so in place.

## New here?

Start with the [Quickstart](get-started/quickstart.md), which has an otari.ai
path and a self-hosted one, then [Connect to Otari](get-started/connect.md) for
base URLs, keys and SDKs.

## Browse the docs

- **Get started**: [Quickstart](get-started/quickstart.md),
  [Connect to Otari](get-started/connect.md), and
  [Ways to run Otari](get-started/ways-to-run-otari.md).
- **Guides**, for what you do with Otari:
  [Organizations and workspaces](guides/organizations-and-workspaces.md),
  [Members and invitations](guides/members.md),
  [API keys](guides/api-keys.md),
  [Provider keys](guides/provider-keys.md),
  [Models](guides/models.md),
  [Budgets](guides/budgets.md),
  [Routing policies](guides/routing.md),
  [Inference Guardrails](guides/guardrails.md),
  [Built-in tools](guides/tools.md),
  [MCP](guides/mcp.md),
  [Files](guides/files.md),
  [Playground](guides/playground.md),
  [Agent traces](guides/traces.md),
  [Importing external usage](guides/external-usage.md),
  [Agent Guardrails](guides/agent-guardrails.md), and the
  [Admin dashboard](guides/dashboard.md) tour.
- **Integrations**, for pointing a tool or framework at Otari:
  [Claude Code](integrations/claude-code.md),
  [Codex](integrations/codex.md),
  [OpenCode](integrations/opencode.md),
  [Pydantic AI](integrations/pydantic-ai.md),
  [LangChain](integrations/langchain.md),
  [LangGraph](integrations/langgraph.md), and a
  [ChatGPT subscription](integrations/chatgpt-subscription.md).
- **Self-hosting**, for running your own:
  [Deployment](self-hosting/deploy.md),
  [Configure providers](self-hosting/configure-providers.md) (with step-by-step
  guides for [OpenAI](self-hosting/providers/openai.md) and
  [Gemini](self-hosting/providers/gemini.md)),
  [Operator administration](self-hosting/administration.md),
  [Tool services](self-hosting/tool-services.md),
  [Provider endpoints](self-hosting/provider-endpoints.md),
  [Health checks](self-hosting/health-checks.md), and
  [Workload identity](self-hosting/workload-identity.md).
- **otari.ai**, for what only the service has:
  [Overview](otari-ai/overview.md) and [Billing](otari-ai/billing.md).
- **Reference**:
  [API reference](reference/api.md),
  [Configuration](reference/configuration.md),
  [Agent Guardrails reference](reference/agent-guardrails.md),
  [SDK compatibility](reference/sdk-compatibility.md), and the
  [Hybrid-mode](reference/hybrid-mode-protocol.md) and
  [Code-execution](reference/code-execution-protocol.md) protocols, whose
  machine-readable form is [`public/code-execution-openapi.yaml`](public/code-execution-openapi.yaml).

## For contributors

- [Architecture](../ARCHITECTURE.md): the two-plane model and the extension seam (ports, adapters, and capability lines) that mark what Otari's core ships versus what an overlay can add.
- [Backend domains](../DOMAINS.md): what each backend domain owns, and the layer shape each domain moves toward.
- [Load test](../loadtest/README.md): compares two builds under load on one machine, route by route, and fails when the second got slower or runs more database statements per request. CI runs it on a PR against its base.
- [Writing docs](../CONTRIBUTING.md#writing-docs): how these pages are organized, and when a page says otari.ai or self-hosted.
