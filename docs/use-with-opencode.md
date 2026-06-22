# Use with opencode

[opencode](https://opencode.ai) lets you register any OpenAI-compatible backend
as a provider. Otari exposes an OpenAI-compatible endpoint
(`POST /v1/chat/completions`) in both standalone and hybrid modes, so you can
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

`baseURL` is the Otari root plus `/v1` (opencode appends `/chat/completions`
itself). Point it at your Otari: `http://localhost:8000/v1` for local
development, or `https://api.otari.ai/v1` when connected to otari.ai.

Export your key so opencode reads it from the environment instead of the config
file, then run with a model:

```bash
export OTARI_API_KEY=<your-token>          # standalone API key, or tk_ user token
opencode --model otari/openai:gpt-4o
```

The `apiKey` is sent as `Authorization: Bearer <token>`, which Otari
accepts for both standalone API keys and connected user tokens.

## Choosing a model

The provider id is `otari` (from your config) and the model selector is whatever
your deployment expects:

- **Standalone:** `provider:model`, e.g. `otari/openai:gpt-4o`,
  `otari/anthropic:claude-sonnet-4-6`, `otari/mistral:mistral-large-latest`.
- **Connected to otari.ai:** `otari/mzai:<catalog-id>` for a managed open-weight
  model (e.g. `otari/mzai:moonshotai/Kimi-K2.6`), or `otari/provider/model` for
  one of your own provider keys (e.g. `otari/openai/gpt-4o`). An `mzai:` prefix
  selects the managed catalog, so adding it to a proprietary model misroutes it.

Any model in the catalog works; Otari routes the request to the right
provider and records usage and cost for it the same way as any other client.

## See also

- [Use with Claude Code](use-with-claude-code.md) — drive the Claude Code CLI
  through the same Otari via the Anthropic Messages API.
