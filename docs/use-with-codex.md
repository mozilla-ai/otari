# Use with Codex

Codex can call Otari as a custom model provider over the Responses API. Otari
serves `POST /api/v1/responses`.

## Route Codex through Otari

Create an API key on the **API keys** page of the otari.ai dashboard, then add a
provider to `~/.codex/config.toml`:

```toml
model_provider = "otari"
model = "openai:gpt-5.4"

[model_providers.otari]
name = "Otari"
base_url = "https://api.otari.ai/api/v1"
wire_api = "responses"
env_key = "OTARI_API_KEY"
```

```bash
export OTARI_API_KEY="<your-otari-key>"
codex
```

Codex appends `/responses`, so `base_url` must include `/api/v1`.
`model_provider` must select the custom provider; otherwise Codex continues
using its built-in OpenAI provider.

Accounts in otari.ai's EU region use
`base_url = "https://eu.api.otari.ai/api/v1"` instead: `api.otari.ai` answers an
EU key with `421 Misdirected Request`, naming the host to use. On a gateway you
run yourself, set `base_url` to its URL plus `/api/v1`, such as
`http://localhost:8000/api/v1`, and use a key it issued; the rest of the block
is the same.

The official [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
is the source of truth for custom-provider keys and supported options.

## Choosing a model

Codex sends the configured `model` string unchanged, so write it as Otari spells
it, such as `openai:gpt-5.4` ([Model format](models.md#model-format)).
[Listing available models](models.md#listing-available-models) shows which ones
your key can use.

A provider with its own Responses API is called natively. Any other provider is
served by translating each request into a chat completion, which refuses a few
Responses features Codex may use, such as tool types other than `function`; see
[Responses on providers without a Responses API](api-reference.md#responses-on-providers-without-a-responses-api).
Codex works best on a provider with a native Responses API.

## Model metadata

Codex associates context windows, reasoning options, and tool support with exact
model slugs. An Otari-prefixed selector may not match Codex's bundled metadata,
in which case Codex uses fallback metadata.

If exact metadata matters, configure `model_catalog_json` with a catalog built
for the same Codex version that consumes it. The schema is owned by Codex and can
change between releases; avoid copying a version-specific schema into Otari
documentation. See the official configuration reference above.

## Troubleshooting

- Confirm `model_provider = "otari"` is set at user level. Codex ignores custom
  provider selection in project-local configuration.
- Include `/api/v1` in `base_url`.
- Start a new Codex session after changing provider configuration.
- Configure provider credentials and pricing in Otari, not in Codex.
- If a request fails with a `400` naming a Responses field, the selected
  provider is being served through Chat Completions; choose a provider with a
  native Responses API.

## Import Codex usage without routing

Codex can export direct OpenAI usage to Otari over OTLP. This is for sessions
that do not already route through Otari. It needs a budget-exempt API key, and
only a deployment's operator creates one (**Exempt from budget** in the
deployment view's key form, or `exclude_from_budget: true` on
`POST /api/v1/keys`), so it works on a gateway you run and not on otari.ai
today: a key from otari.ai's **API keys** page is always budget-enforced, and
the import refuses it. See [Importing external usage](external-usage.md#enforcement-boundary).

With a dedicated budget-exempt key, configure the HTTP logs exporter with the
full Otari logs path, using your gateway's URL in place of
`http://localhost:8000`:

```toml
[otel]
environment = "otari"
log_user_prompt = false
exporter = { otlp-http = { endpoint = "http://localhost:8000/otlp/v1/logs", protocol = "binary", headers = { "Authorization" = "Bearer tk-your-import-key" } } }
```

`binary` sends protobuf; `json` also works. Otari does not accept OTLP over
gRPC. Keeping `log_user_prompt = false` avoids exporting prompt content; Otari
does not persist it in either case.

Imported events are priced for analytics and never count toward budgets. Do not
both route and export one session, or its cost will appear twice. See
[Importing external usage](external-usage.md).

## Related documentation

- [Models](models.md)
- [Use with a ChatGPT subscription](chatgpt-subscription.md)
- [API reference](api-reference.md)
