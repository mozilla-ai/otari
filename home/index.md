# What is Otari?

Otari is an LLM gateway: one endpoint between your applications and the model
providers they call. 

Your applications cand send requests (in Anthropic, Gemini and/or Openai),
and Otari routes each one to any of the 40+ supported providers.

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

You can use our hosted version at [otari.ai](https://otari.ai), or
run Otari yourself (see [Self-hosted quickstart](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/self-hosting)). 


### Integrations

Point an agent or a framework at Otari:
[Claude Code](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/agents-and-frameworks/use-with-claude-code), [Codex](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/agents-and-frameworks/use-with-codex),
[OpencCde](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/agents-and-frameworks/use-with-opencode), [Pydantic AI](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/agents-and-frameworks/use-with-pydantic-ai),
[LangChain](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/agents-and-frameworks/use-with-langchain) and [LangGraph](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/agents-and-frameworks/use-with-langgraph)...

Missing an option? Don't hesitate to [open an issue on github](https://github.com/mozilla-ai/otari/issues/new/choose).

## Where to go next

- **Home**: the [Quickstart](quickstart.md) sends a first request through
  otari.ai, from curl and from your code, in a few minutes.
- **Guides**: how to do one thing with Otari, such as give each end user their
  own budget, add guardrails, or read agent traces.
- **Self-Hosting**: running your own gateway, starting with the
  [Self-hosted quickstart](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/self-hosting), and its
  [configuration](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/self-hosting/configuration) and providers.
- **Reference**: the API, models, routing, tools and protocols: every endpoint,
  field, error code and limit.
