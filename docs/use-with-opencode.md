# Use with opencode

[opencode](https://opencode.ai) lets you register any OpenAI-compatible backend
as a provider. The gateway exposes an OpenAI-compatible endpoint
(`POST /v1/chat/completions`) in both standalone and connected modes, so you can
route opencode through Otari to get budgets, usage tracking, and traces without
changing how you code.

## Quick start

Add Otari as a provider in your `opencode.jsonc`:

```jsonc
{
  "provider": {
    "otari": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "{env:OTARI_API_KEY}"
      }
    }
  }
}
```

`baseURL` is the gateway root plus `/v1` (opencode appends `/chat/completions`
itself). Point it at your gateway: `http://localhost:8000/v1` for local
development, or `https://api.otari.ai/v1` when connected to otari.ai.

Export your key so opencode reads it from the environment instead of the config
file, then run with a model:

```bash
export OTARI_API_KEY=<your-token>          # standalone API key, or tk_ user token
opencode --model otari/openai:gpt-4o
```

The `apiKey` is sent as `Authorization: Bearer <token>`, which the gateway
accepts for both standalone API keys and connected user tokens.

## Choosing a model

The provider id is `otari` (from your config) and the model selector is whatever
your deployment expects:

- **Standalone:** `provider:model`, e.g. `otari/openai:gpt-4o`,
  `otari/anthropic:claude-sonnet-4-6`, `otari/mistral:mistral-large-latest`.
- **Connected to otari.ai:** the platform selector, e.g.
  `otari/mzai:openai/gpt-4o` or `otari/mzai:moonshotai/Kimi-K2.6`.

Any model in the catalog works; the gateway routes the request to the right
provider and records usage and cost for it the same way as any other client.

## See also

- [Use with Claude Code](use-with-claude-code.md) — drive the Claude Code CLI
  through the same gateway via the Anthropic Messages API.
