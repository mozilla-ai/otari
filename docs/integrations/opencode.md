# OpenCode

[OpenCode](https://opencode.ai) lets you register any OpenAI-compatible backend
as a provider, and Otari is one (`POST /api/v1/chat/completions`), so you can
route OpenCode through Otari to get budgets, usage tracking and traces without
changing how you code.

## Quick start

Add Otari as a provider in your `opencode.jsonc`. `baseURL` is your base URL from
[Connect to Otari](../get-started/connect.md) plus `/v1`, since OpenCode appends
`/chat/completions` itself:

```jsonc
{
  "provider": {
    "otari": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://api.otari.ai/api/v1",
        "apiKey": "{env:OTARI_API_KEY}"
      }
    }
  }
}
```

That is `https://eu.api.otari.ai/api/v1` for an otari.ai account in the European
Union, and `http://localhost:8000/api/v1` for a gateway on your own machine.

Export your key so OpenCode reads it from the environment instead of the config
file, then run with a model:

```bash
export OTARI_API_KEY=<your-otari-key>
opencode --model otari/openai:gpt-4o
```

The `apiKey` is sent as `Authorization: Bearer <key>`, which Otari accepts for
every key it issues.

## Choosing a model

The provider id is `otari` (from your config), and the model selector is any
model your organization can use, which `GET /api/v1/models` returns for
your key: for example `otari/openai:gpt-4o`, `otari/anthropic:claude-sonnet-4-6`,
or a catalog id such as `otari/moonshotai/kimi-k3`. `provider/model` works as
well as `provider:model`; see [Models](../guides/models.md#model-format).

Any model in the catalog works; Otari routes the request to the right
provider and records usage and cost for it the same way as any other client.

## See also

- [Claude Code](claude-code.md): drive the Claude Code CLI
  through the same Otari via the Anthropic Messages API.
- [Codex](codex.md): drive the Codex CLI through the
  same Otari via the OpenAI Responses API.
