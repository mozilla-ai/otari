# Codex

Codex can use Otari as a custom model provider over the Responses API.

## Route Codex through Otari

1. Create a key on the dashboard's **API keys** page.
2. Add a provider to `~/.codex/config.toml`:

   ```toml
   model_provider = "otari"
   model = "openai:gpt-5.4"

   [model_providers.otari]
   name = "Otari"
   base_url = "https://api.otari.ai/api/v1"
   wire_api = "responses"
   env_key = "OTARI_API_KEY"
   ```

3. Export the key and start Codex:

   ```bash
   export OTARI_API_KEY="<your-otari-key>"
   codex
   ```

- `base_url` must end in `/api/v1`, because Codex adds `/responses`. EU accounts
  use `https://eu.api.otari.ai/api/v1`, and a gateway you run uses its URL plus
  `/api/v1`. See [Base URL](api-reference.md#base-url).
- `model_provider` must select the custom provider, or Codex keeps using its
  built-in OpenAI provider.

The [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
lists every custom-provider option.

## Choosing a model

Codex sends `model` unchanged, so write it as Otari spells it, such as
`openai:gpt-5.4` ([Model format](models.md#model-format)).
[Listing available models](models.md#listing-available-models) shows which
models your key can use.

Codex works best on a provider with its own Responses API. Other providers are
served by translating each request to a chat completion, which refuses some
features Codex may use, such as tool types other than `function`; see
[Responses on providers without a Responses API](api-reference.md#responses-on-providers-without-a-responses-api).

To list only the open-weight models, ask the catalog for
`GET /api/v1/catalog/models?capability=open_weights`. Each entry's `selectors`
are the names to configure.

Codex looks up context windows, reasoning options and tool support by exact model
name. An Otari model name may not match, and then Codex uses fallback values. If
that matters, set `model_catalog_json` as the Codex configuration reference
describes, built for your Codex version.

## Troubleshooting

- Set `model_provider = "otari"` in your user-level config. Codex ignores it in a
  project-level config.
- Include `/api/v1` in `base_url`.
- Start a new Codex session after changing the provider.
- Set provider credentials and pricing in Otari, not in Codex.
- A `400` naming a Responses field means the provider is served through Chat
  Completions. Choose a provider with its own Responses API.

## Import Codex usage without routing

Codex can export the usage of sessions that go straight to OpenAI. This needs an
[import key](external-usage.md#create-an-import-key), budget-exempt and holding
`telemetry:write`, which only a deployment's operator can create, so it works on
a gateway you run, not on otari.ai today. See
[Enforcement boundary](external-usage-reference.md#enforcement-boundary).

Add the logs exporter to `~/.codex/config.toml`, with your gateway's URL and the
full logs path:

```toml
[otel]
environment = "otari"
log_user_prompt = false
exporter = { otlp-http = { endpoint = "http://localhost:8000/otlp/v1/logs", protocol = "binary", headers = { "Authorization" = "Bearer tk-your-import-key" } } }
```

- `binary` sends protobuf; `json` also works. gRPC does not.
- `log_user_prompt = false` keeps prompts out of the export. Otari does not
  store them either way.

Imported usage never counts toward budgets. Do not export a session you also
route, or its cost appears twice. See
[Import usage from outside Otari](external-usage.md).

## See also

- [Models](models.md)
- [Use with a ChatGPT subscription](chatgpt-subscription.md)
