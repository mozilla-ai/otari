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

You can use the gateway mozilla.ai runs at [otari.ai](https://otari.ai), or
run Otari yourself with your own database and provider keys. The two differ in
who runs the gateway, its base URL, where its data lives and who bills you, not
in the API your applications call. The [Quickstart](quickstart.md) starts on
otari.ai, and the [Self-hosted quickstart](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting) runs your
own gateway.

## Where to go next

- **Home**: the [Quickstart](quickstart.md) sends a first request through
  otari.ai in a few minutes.
- **Guides**: how to do one thing with Otari, such as give each end user their
  own budget, add guardrails, or read agent traces.
- **Self-Hosting**: running your own gateway, starting with the
  [Self-hosted quickstart](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/self-hosting), and connecting it to providers.
- **Reference**: every setting, endpoint, error code and limit.

### Integrations

Point an agent or a framework at Otari:
[Claude Code](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-claude-code), [Codex](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-codex),
[opencode](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-opencode), [Pydantic AI](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-pydantic-ai),
[LangChain](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-langchain) and [LangGraph](https://mozilla-ai.gitbook.io/otari-preview/1IBsGVIwnr70CsAaCBAO/guides/agents-and-frameworks/use-with-langgraph).

### For contributors

- [Architecture](https://github.com/mozilla-ai/otari/blob/main/ARCHITECTURE.md): one process over one database, and the extension seam (ports, adapters, and capability lines) that mark what Otari's core ships versus what an overlay can add.
- [Backend domains](https://github.com/mozilla-ai/otari/blob/main/DOMAINS.md): what each backend domain owns, and the layer shape each domain moves toward.
- [Load test](https://github.com/mozilla-ai/otari/blob/main/loadtest/README.md): compares two builds under load on one machine, route by route, and fails when the second got slower or runs more database statements per request. CI runs it on a PR against its base.
