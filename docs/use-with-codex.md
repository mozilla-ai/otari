# Use with Codex

Codex can call Otari as a custom model provider over the Responses API. Otari
serves `POST /v1/responses` in standalone and hybrid modes.

## Route Codex through Otari

Create an Otari API key, then add a provider to `~/.codex/config.toml`:

```toml
model_provider = "otari"
model = "openai:gpt-5.4"

[model_providers.otari]
name = "Otari"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
env_key = "OTARI_API_KEY"
```

```bash
export OTARI_API_KEY="gw-your-otari-key"
codex
```

Codex appends `/responses`, so `base_url` must include `/v1`.
`model_provider` must select the custom provider; otherwise Codex continues
using its built-in OpenAI provider.

For otari.ai, change the URL and use an otari.ai user token:

```toml
model_provider = "otari"
model = "openai:gpt-5.4"

[model_providers.otari]
name = "Otari"
base_url = "https://api.otari.ai/v1"
wire_api = "responses"
env_key = "OTARI_API_KEY"
```

```bash
export OTARI_API_KEY="tk_your_otari_token"
codex
```

The official [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
is the source of truth for custom-provider keys and supported options.

## Choosing a model

Codex sends the configured model string unchanged. Use a model or named instance
served by the Otari deployment.

The upstream provider must implement the Responses API. Otari rejects unsupported
providers before dispatch. A hybrid attempt plan is rejected if any candidate
cannot serve Responses.

In standalone mode, inspect `GET /v1/models`. Hybrid model IDs come from the
connected control plane.

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
- Include `/v1` in `base_url`.
- Start a new Codex session after changing provider configuration.
- Configure provider credentials and pricing in Otari, not in Codex.
- Verify the selected provider supports the Responses API.

## Import Codex usage without routing

Codex can export direct OpenAI usage to Otari over OTLP. This is for sessions
that do not already route through Otari. Send it to a standalone or hosted
gateway; hybrid gateways do not serve the OTLP endpoints.

Create a dedicated API key with `exclude_from_budget: true`, then configure the
HTTP logs exporter with the full Otari logs path:

```toml
[otel]
environment = "otari"
log_user_prompt = false
exporter = { otlp-http = { endpoint = "https://otari.example.com/v1/logs", protocol = "binary", headers = { "Authorization" = "Bearer gw-your-import-key" } } }
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
