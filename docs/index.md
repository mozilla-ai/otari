---
description: Otari is an open-source AI gateway by Mozilla AI (Apache-2.0). Run it yourself for free, or use the hosted version at otari.ai, see pricing.
icon: circle-info
---

# What is Otari?

Otari is an open-source AI gateway by Mozilla AI (Apache-2.0). Run it yourself
for free, or use the hosted version at [otari.ai](https://otari.ai/),
[see pricing](https://otari.ai/#pricing).

Otari is an AI gateway: one endpoint between your applications and the model
providers they call.

Your applications can send requests (in Anthropic, Gemini and/or OpenAI),
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
run Otari yourself (see [Self-hosted quickstart](self-hosted-quickstart.md)). 


### Integrations

Point an agent or a framework at Otari:
[Claude Code](use-with-claude-code.md), [Codex](use-with-codex.md),
[Junie](use-with-junie.md), [opencode](use-with-opencode.md), [Pydantic AI](use-with-pydantic-ai.md),
[LangChain](use-with-langchain.md) and [LangGraph](use-with-langgraph.md)...

Missing an option? Don't hesitate to [open an issue on github](https://github.com/mozilla-ai/otari/issues/new/choose).

## Where to go next

- **Otari**: the [Quickstart](quickstart.md) sends a first request through
  otari.ai, from curl and from your code, in a few minutes. Then, in the
  dashboard: [manage workspaces](manage-workspaces.md),
  [invite members](invite-and-manage-organization-members.md), add
  [provider keys](manage-provider-keys.md),
  [set a budget](set-a-budget-and-ceiling.md),
  [route across providers](create-a-multi-provider-routing.md) and
  [add guardrails](manage-guardrails.md).
- **Guides**: how to do one thing with Otari, such as give each end user their
  own budget, add guardrails, or read agent traces.
- **Self-Hosting**: running your own gateway, starting with the
  [Self-hosted quickstart](self-hosted-quickstart.md), and its
  [configuration](configuration.md) and providers.
- **Reference**: the API, models, routing, tools and protocols: every endpoint,
  field, error code and limit.
