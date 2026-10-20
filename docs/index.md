# What is Otari?

Otari is an LLM gateway: one endpoint between your applications and the model
providers they call. Applications send it OpenAI- or Anthropic-style requests,
and Otari routes each one to any of 40+ providers through
[any-llm](https://github.com/mozilla-ai/any-llm).

On the way through, it:

- **Authenticates the caller** with an API key that can be scoped to a user, a
  workspace and a set of models, and revoked on its own.
- **Holds the provider credentials**, so applications never see them.
- **Enforces budgets before the request is sent**, so spend stops at the limit
  rather than after it.
- **Routes around failures**, with fallbacks, load balancing and spill-over
  between models.
- **Records what happened**: usage and cost per request, and each agent session
  as a trace.
- **Adds what a provider lacks**: built-in tools (code execution, web search and
  fetch), MCP servers, file uploads and guardrails.

## Two ways to run it

You can use the hosted gateway at [otari.ai](https://otari.ai), or run Otari
yourself with your own database and provider keys. The two differ in who runs
the gateway and where its data lives, not in the API your applications call.
[Runtime modes](modes.md) compares them.

## Where to go next

- **Home**: the [Quickstart](quickstart.md) runs Otari on your machine and sends
  a first request through it.
- **Guides**: how to do one thing with Otari, such as give each end user their
  own budget, add guardrails, or read agent traces.
- **Self-Hosting**: deploying and operating your own gateway, and connecting it
  to providers.
- **Reference**: every setting, endpoint, error code and limit.

### Integrations

Point an agent or a framework at Otari:
[Claude Code](use-with-claude-code.md), [Codex](use-with-codex.md),
[opencode](use-with-opencode.md), [Pydantic AI](use-with-pydantic-ai.md),
[LangChain](use-with-langchain.md) and [LangGraph](use-with-langgraph.md).

### For contributors

- [Architecture](../ARCHITECTURE.md): the two-plane model and the extension seam (ports, adapters, and capability lines) that mark what Otari's core ships versus what an overlay can add.
- [Backend domains](../DOMAINS.md): what each backend domain owns, and the layer shape each domain moves toward.
- [Load test](../loadtest/README.md): compares two builds under load on one machine, route by route, and fails when the second got slower or runs more database statements per request. CI runs it on a PR against its base.
