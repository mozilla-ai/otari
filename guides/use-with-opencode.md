# Use with opencode

[opencode](https://opencode.ai) can use any OpenAI-compatible backend as a
provider, including Otari.

## Quick start

1. Create a key on the dashboard's **API keys** page.
2. Add Otari as a provider in your `opencode.jsonc`:

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

3. Export the key and run opencode with a model:

   ```bash
   export OTARI_API_KEY=<your-otari-key>
   opencode --model otari/openai:gpt-4o
   ```

`baseURL` ends in `/api/v1`; opencode adds `/chat/completions` itself. EU
accounts use `https://eu.api.otari.ai/api/v1`, and a gateway you run uses its
URL plus `/api/v1`. See [Base URL](https://docs.otari.ai/reference#base-url).

## Choosing a model

opencode names a model `<provider id>/<model>`, and everything after `otari/`
goes to Otari unchanged, so write it as Otari spells it:

- On otari.ai, a catalog id such as `otari/deepseek/deepseek-v4.1-flash` lets
  Otari choose the provider, and
  `otari/anthropic:claude-sonnet-4-6` pins one.
- On your own gateway, use `provider:model` for a provider it has, such as
  `otari/openai:gpt-4o`.

See [Model format](https://docs.otari.ai/reference/gateway/models#model-format) and
[Listing available models](https://docs.otari.ai/reference/gateway/models#listing-available-models).

To list only the open-weight models a key can use, ask the catalog for them:
`GET /api/v1/catalog/models?capability=open_weights`. Each entry's `selectors`
are the names to configure.

## See also

- [Use with Claude Code](use-with-claude-code.md)
- [Use with Codex](use-with-codex.md)
