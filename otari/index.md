---
description: Otari is an open-source AI gateway by Mozilla AI (Apache-2.0). Run it yourself for free, or use the hosted version at otari.ai, see pricing.
icon: circle-info
---

# What is Otari?

Otari is an open-source AI gateway by Mozilla AI (Apache-2.0) and a policy layer that provides a unified control plane for AI infrastructure. Run it yourself for free, or use the hosted version at [otari.ai](https://otari.ai/) ([see pricing](https://otari.ai/#pricing)).

Use Otari, the hosted platform, when you want workspace-scoped API tokens, provider access, usage reporting, and budgets without running the gateway yourself.

If you need the runtime in your own infrastructure instead, read the [Self-hosted quickstart](https://docs.otari.ai/self-hosting).

## What you can do

* **Use models from multiple providers.** Find available models in the **Models** tab.
* **Control access with workspace keys.** Create scoped API keys for your applications.
* **Track usage in one place.** Review requests and usage from the **Overview**.

## How it works

Your workspace is the control plane. It holds your provider connections, model access, API keys, and usage data. Your application sends requests to `https://api.otari.ai/api/v1` (`https://eu.api.otari.ai/api/v1` for an account in the EU) with a workspace API key and a model ID, in the OpenAI, Anthropic, or Gemini format, and Otari routes each one to any of 40+ supported providers.

This lets you keep application integrations consistent while managing model access centrally.

## What the hosted platform gives you

**Workspaces:** Separate apps, environments, or teams into distinct boundaries for provider access, usage, budgets, and members.

**Provider access**: Manage provider keys at the organization level, then decide which workspaces can use them.

**Usage and spend controls**: Track requests, inspect usage, and add budgets before costs become surprises.

**Routing**: Fall back to another model when one fails, or spread traffic across providers.

**Tools**: Give requests web search, code execution, and MCP servers, whatever the provider.

**Governance**: Add guardrails to enforce consistent safety and compliance checks on prompts and responses across every provider and model.

## Integrations

Point an agent or a framework at Otari: [Claude Code](https://docs.otari.ai/guides/agents-and-frameworks/use-with-claude-code), [Codex](https://docs.otari.ai/guides/agents-and-frameworks/use-with-codex), [Junie](https://docs.otari.ai/guides/agents-and-frameworks/use-with-junie), [opencode](https://docs.otari.ai/guides/agents-and-frameworks/use-with-opencode), [Pydantic AI](https://docs.otari.ai/guides/agents-and-frameworks/use-with-pydantic-ai), [LangChain](https://docs.otari.ai/guides/agents-and-frameworks/use-with-langchain) and [LangGraph](https://docs.otari.ai/guides/agents-and-frameworks/use-with-langgraph).

Missing one? [Open an issue on GitHub](https://github.com/mozilla-ai/otari/issues/new/choose).

## Start here

Create a workspace API key, select a model, and send a request. The [Quickstart](quickstart.md) takes about five minutes.

The rest of the docs are organized as **Guides** (how to do one thing with Otari), **Self-Hosting** (running your own gateway, from the [Self-hosted quickstart](https://docs.otari.ai/self-hosting) to its [configuration](https://docs.otari.ai/self-hosting/configuration)), and **Reference** (every endpoint, field, error code and limit).
