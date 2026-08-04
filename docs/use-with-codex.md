# Use with Codex

[Codex](https://developers.openai.com/codex) talks to models over the OpenAI
Responses API and lets you register any compatible endpoint as a custom model
provider. Otari exposes that surface (`POST /v1/responses`) in both standalone
and hybrid modes, so you can route Codex through Otari to get virtual keys,
budgets, and usage tracking without changing how you use the CLI.

If you would rather keep Codex on its own credentials, you can still get its
usage into Otari by pointing Codex's OpenTelemetry export at the gateway. That
path is covered in [Import Codex usage](#import-codex-usage-without-routing-through-otari)
at the end of this page.

> **Route or export, not both.** If a session both routes its API traffic through
> Otari and exports telemetry to it, every call lands twice: once as
> `source = gateway` (enforced, counts toward budget) and once as `source = codex`
> (exempt observability). The two rows are not correlated, so budgets and `spend`
> stay correct, but cost analytics count the same traffic twice. Pick one path per
> session.

## Quick start (standalone)

This is the primary flow: a self-hosted Otari with your own provider credentials.
It assumes a provider is already configured (see the
[OpenAI provider guide](providers/openai.md) and [Supported models](models.md));
this page does not repeat provider setup.

### 1. Create an Otari key

Codex sends the key as `Authorization: Bearer <token>`, which is the scheme Otari
accepts. In the dashboard: Keys -> create a key for a user. Or over the API with
the master key:

```bash
curl -sS "$OTARI_URL/v1/keys" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"key_name":"codex","user_id":"alice"}'
```

The response's `key` field (`gw-...`) is shown once. Export it under the name you
will reference from the Codex config:

```bash
export OTARI_API_KEY=gw-your-otari-key
```

### 2. Point Codex at Otari

Add a custom provider to `~/.codex/config.toml` and make it the default:

```toml
# ~/.codex/config.toml
model_provider = "otari"
model = "openai:gpt-5.4"

[model_providers.otari]
name = "Otari"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
env_key = "OTARI_API_KEY"
```

- `base_url` is the Otari root plus `/v1`; Codex appends `/responses` itself.
- `wire_api = "responses"` is what makes Codex speak the Responses API rather
  than chat completions.
- `env_key` names the environment variable Codex reads the key from, so the
  token stays out of the config file.
- `model_provider = "otari"` selects the entry above. Without it Codex keeps
  using its built-in OpenAI provider and never touches your gateway.

### 3. Run it

```bash
codex
```

Requests now land in Otari against your key. In the dashboard's Activity page
they show up with `endpoint = /v1/responses`, priced and counted against the
key's budget like any other traffic.

## Connected to otari.ai

The configuration is the same shape; only the base URL and the token change. Use
your self-hosted gateway's URL plus `/v1`, or `https://api.otari.ai/v1` for
otari.ai's hosted gateway, and a `tk_` user token instead of a local API key:

```toml
# ~/.codex/config.toml
model_provider = "otari"
model = "openai:gpt-5.4"

[model_providers.otari]
name = "Otari"
base_url = "https://api.otari.ai/v1"
wire_api = "responses"
env_key = "OTARI_API_KEY"
```

```bash
export OTARI_API_KEY=tk_your_otari_token
codex
```

Two differences from standalone:

- `GET /v1/models` is standalone-only, so a hybrid gateway will not list models
  for you. Take the model ids from otari.ai instead.
- Hybrid mode can try several providers for one request, and Otari checks that
  **every** attempt in the resolved route speaks the Responses API before
  dispatching. A route whose fallback lands on a provider that does not (see
  below) is rejected up front.

## Choosing a model

Codex sends the `model` string through unchanged, so it has to be a selector the
Otari deployment you are pointing at actually serves:

- **Standalone, any configured provider:** `provider:model`, for example
  `openai:gpt-5.4` or `openai:gpt-5-mini`.
- **Named provider instances:** the instance name replaces the provider, for
  example `chatgpt:gpt-5` for an instance declared as `chatgpt:` in `config.yml`.
  See [Named provider instances](models.md#named-provider-instances).

In standalone mode, `GET /v1/models` lists everything the gateway can resolve,
which is the authoritative source for valid ids:

```bash
curl -sS "$OTARI_URL/v1/models" -H "Authorization: Bearer $OTARI_API_KEY"
```

### The provider has to speak the Responses API

Otari serves `/v1/responses` only for providers that implement it, such as
OpenAI, Azure OpenAI, Groq, Fireworks, and HuggingFace, plus any
`provider_type: openai-compatible` instance (those run on the OpenAI
implementation). Providers with no Responses support, Anthropic and Mistral
among them, are rejected before the call goes out:

```json
{"detail": "Provider 'anthropic' does not support the Responses API"}
```

That is a 400, and it is a property of the endpoint rather than of Codex. To
drive a Claude model through Otari, use the Anthropic Messages surface with
[Claude Code](use-with-claude-code.md) instead.

## Model metadata for custom selectors

Codex looks up model metadata (context window, reasoning levels, tool support) by
exact slug. Otari selectors carry a provider prefix, so they miss Codex's bundled
metadata even when the underlying model is one Codex knows:

```text
warning: Model metadata for `openai:gpt-5.4` not found. Defaulting to fallback
metadata; this can degrade performance and cause issues.
```

Codex still runs, on conservative fallback metadata, and its model picker only
offers models it has metadata for. To get the real numbers back, hand Codex a
catalog whose slugs are your Otari selectors:

```toml
model = "openai:gpt-5.4"
model_provider = "otari"
model_catalog_json = "/absolute/path/to/otari-models.json"
```

Build that file from Codex's own bundled metadata, keeping only the models your
gateway actually serves:

```bash
codex debug models --bundled > bundled.json
curl -sS "$OTARI_URL/v1/models" -H "Authorization: Bearer $OTARI_API_KEY" > served.json

python3 - <<'PY'
import json

PREFIX = "openai:"  # the Otari provider or instance name, plus ":"

served = {m["id"] for m in json.load(open("served.json"))["data"]}
bundled = json.load(open("bundled.json"))["models"]
models = [dict(m, slug=PREFIX + m["slug"]) for m in bundled if PREFIX + m["slug"] in served]
json.dump({"models": models}, open("otari-models.json", "w"), indent=2)
print([m["slug"] for m in models])
PY
```

Filtering against `/v1/models` keeps the picker honest: it lists only models the
gateway can actually route. Reasoning level is a separate setting; Codex does not
infer one for you, so set `model_reasoning_effort` in `config.toml` if you want
something other than the default.

## Gotchas

- **Use a custom `model_providers` entry, not Codex's built-in OpenAI auth.**
  The built-in provider goes straight to `api.openai.com` (it even opens a
  WebSocket to `wss://api.openai.com/v1/responses`), so an `OPENAI_API_KEY` in
  your environment never reaches Otari. `model_provider` has to name your entry.
- **Include `/v1` in `base_url`.** Codex uses the URL as given and appends
  `/responses`; drop the `/v1` and every request 404s.
- **Start a new Codex session after changing provider configuration.** A running
  session keeps the provider it started with, so edits to `config.toml` look like
  they did nothing.
- **Configure the upstream provider in Otari, not in Codex.** Codex only needs
  the gateway URL and an Otari key. Credentials, pricing, and model availability
  come from Otari's [provider configuration](models.md#configuring-a-provider).
- **A `model_catalog_json` file is tied to the Codex version that reads it.**
  The schema changes between releases: `supports_reasoning_summaries` is required
  by `0.144.5` and absent from `0.145.0`, so a catalog generated by one fails to
  load in the other with `failed to parse model_catalog_json path ... as JSON:
  missing field ...`. Generate the catalog with the same Codex version that
  consumes it, and if one file is shared across environments (a pinned CI image
  and an auto-updating local CLI), generate it against the oldest one, since
  newer Codex tolerates extra fields.

## Import Codex usage (without routing through Otari)

If you keep Codex on its own credentials (directly against OpenAI, not routed
through Otari), Otari can still see that usage: point Codex's OTLP export at
Otari and each request lands in your usage analytics, priced at API-equivalent
rates. Nothing about how you run Codex changes.

Codex has native OpenTelemetry support: it emits a usage event per model call
carrying token counts, the model, and a session id, but no prompt or response
content. This is standalone-only and never affects budgets (a hybrid,
otari.ai-connected gateway does not serve the OTLP endpoints, so the export 404s
there). See [Importing external usage](external-usage.md) for the shared pricing,
idempotency, and budget-exempt rules; what follows is just the Codex setup. The
"route or export, not both" rule at the top of this page applies here: this path
is for sessions that do **not** proxy through Otari.

### 1. Get a budget-exempt import key (admin, once)

Imported usage is retrospective, so Otari can never block it, which is why an import
key must be **budget-exempt** (a budgeted key is refused). In the dashboard: Keys ->
create a key for the user, open **Advanced**, check **Exempt from budget**. Or over
the API with the master key:

```bash
curl -sS "$OTARI_URL/v1/keys" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"key_name":"codex-importer","user_id":"alice","exclude_from_budget":true}'
```

Treat that key as a secret: `exclude_from_budget` also exempts this key's **live**
gateway traffic from reservations, spend, and budget enforcement, so a key that leaks
or is reused for routing grants unmetered access. Use a key (and ideally a dedicated
user) reserved solely for imports, and rotate it if it is exposed.

### 2. Point Codex's telemetry at Otari

Codex configures OpenTelemetry in the `[otel]` section of `~/.codex/config.toml`.
Send it to Otari with the **full** `/v1/logs` path (Codex does not append the signal
path itself), an `http` protocol, and the exempt key:

```toml
# ~/.codex/config.toml
[otel]
environment = "otari"
log_user_prompt = false
exporter = { otlp-http = { endpoint = "https://otari.example.com/v1/logs", protocol = "binary", headers = { "Authorization" = "Bearer gw-your-exempt-key" } } }
```

- `endpoint` must include `/v1/logs`. Unlike the `OTEL_EXPORTER_OTLP_ENDPOINT`
  environment variable, Codex's configured endpoint is used as-is.
- `protocol = "binary"` sends protobuf; `"json"` also works. gRPC is not accepted by
  Otari's HTTP receiver.
- `log_user_prompt = false` keeps prompts out of the export. Otari never stores prompt
  or response content regardless, but there is no reason to send it.

Otari re-prices each event at its own configured rate for the event's timestamp,
records it as `source = codex`, and dedups per request so replays never double-count.
A model with no configured price still lands with `cost: null`; add pricing to see the
cost. Codex reports OpenAI-shaped token counts (cached tokens are a subset of the
input tokens), and Otari de-includes them so cache reads are not billed twice.

### 3. See it

In the dashboard, the Activity page shows each imported request with its **API key**
column, and (expanded) its Source (`Codex`) and session; the Usage page's **Tracked
cost** total separates priced from unpriced usage. Filter the Activity log by API key
to scope it to your importer key.

## See also

- [Use with a ChatGPT subscription](chatgpt-subscription.md): serve Codex from
  subscription-backed models through a local Codex-OAuth proxy.
- [API reference](api-reference.md): the Responses endpoint and its auth rules.
- [Supported models](models.md): provider configuration and the model selector
  format.
- [Importing external usage](external-usage.md): the ingestion contract, pricing,
  and budget behavior behind the telemetry path.
- [Use with Claude Code](use-with-claude-code.md): the equivalent Claude Code setup.
