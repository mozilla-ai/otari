# Use with opencode

[opencode](https://opencode.ai) lets you register any OpenAI-compatible backend
as a provider. Otari exposes an OpenAI-compatible endpoint
(`POST /api/v1/chat/completions`) in both standalone and hybrid modes, so you can
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
        "baseURL": "http://localhost:8000/api/v1",
        "apiKey": "{env:OTARI_API_KEY}"
      }
    }
  }
}
```

`baseURL` is the Otari API root, `/api/v1` (opencode appends `/chat/completions`
itself). Point it at the gateway you are actually using: `http://localhost:8000/api/v1`
for local standalone development, your self-hosted gateway URL plus `/api/v1` when
connected to otari.ai, or `https://api.otari.ai/api/v1` when using otari.ai's
own gateway.

Export your key so opencode reads it from the environment instead of the config
file, then run with a model:

```bash
export OTARI_API_KEY=<your-otari-key>      # from otari.ai, or from your own gateway
opencode --model otari/openai:gpt-4o
```

The `apiKey` is sent as `Authorization: Bearer <key>`, which Otari accepts for
every key it issues.

## Choosing a model

opencode names a model as `<provider id>/<model>`, and the provider id here is
`otari`, from your config. Everything after `otari/` is sent to Otari unchanged,
so write it as Otari spells it: `otari/openai:gpt-4o`,
`otari/anthropic:claude-sonnet-4-6`. [Model format](models.md#model-format) has
the spellings Otari accepts, and
[Listing available models](models.md#listing-available-models) shows which
models your key can use.

## See also

- [Use with Claude Code](use-with-claude-code.md): drive the Claude Code CLI
  through the same Otari via the Anthropic Messages API.
- [Use with Codex](use-with-codex.md): drive the Codex CLI through the
  same Otari via the OpenAI Responses API.
