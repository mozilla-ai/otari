# opencode

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
         "name": "Otari",
         "options": {
           "baseURL": "https://api.otari.ai/api/v1",
           "apiKey": "{env:OTARI_API_KEY}"
         },
         "models": {
           "openai:gpt-4o": {}
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

`models` lists the models opencode offers from this provider, keyed by the name
after `otari/`. A provider with none is not loaded, and opencode answers
`Provider not found: otari`. `baseURL` ends in `/api/v1`; opencode adds
`/chat/completions` itself. EU
accounts use `https://eu.api.otari.ai/api/v1`, and a gateway you run uses its
URL plus `/api/v1`. See [Base URL](api-reference.md#base-url).

## Choosing a model

opencode names a model `<provider id>/<model>`, and everything after `otari/`
goes to Otari unchanged, so write it as Otari spells it. opencode offers only the
models listed under `models`, so add that part as a key there before you use it:

- On otari.ai, a catalog id such as `otari/deepseek/deepseek-v4.1-flash` lets
  Otari choose the provider, and
  `otari/anthropic:claude-sonnet-4-6` pins one.
- On your own gateway, use `provider:model` for a provider it has, such as
  `otari/openai:gpt-4o`.

See [Model format](models.md#model-format) and
[Listing available models](models.md#listing-available-models).

To list only the open-weight models a key can use, ask the catalog for them:
`GET /api/v1/catalog/models?capability=open_weights`. Each entry's `selectors`
are the names to configure.

## See also

- [Claude Code](claude-code.md)
- [Codex](codex.md)
