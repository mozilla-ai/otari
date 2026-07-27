# Use with a ChatGPT subscription

You can route Otari at the models behind a ChatGPT Plus or Pro subscription,
even though a subscription ships no API key. This page walks through the one
extra hop that makes it work.

## Why this needs a proxy

A ChatGPT subscription and the OpenAI Platform API are separate products with
separate billing. The subscription has no standard `sk-...` API key, so Otari
cannot point at it directly. The subscription is only reachable programmatically
through **Codex OAuth**, the same login the Codex CLI uses.

The bridge is a small **Codex-OAuth proxy** you run on `localhost`. It logs in
with your subscription and re-exposes those models as a standard
OpenAI-compatible endpoint. Otari then treats that endpoint like any other
self-hosted OpenAI-compatible backend, and budgets, usage logging, and key
management layer on top unchanged.

```
your app  ->  Otari  ->  Codex-OAuth proxy  ->  Codex OAuth (ChatGPT subscription)
```

The whole stack can stay on `localhost`.

> **This rides an unofficial backend.** You are reaching the ChatGPT backend
> through a compatibility proxy, not the supported Platform API. It stays subject
> to your subscription's usage caps, and OpenAI can change the backend at any
> time and break the proxy. If you want a sturdy, supported path, use an OpenAI
> Platform API key with the [OpenAI provider guide](providers/openai.md)
> instead. This recipe is for getting extra mileage from a subscription you
> already pay for.

## Prerequisites

- Otari running locally (see the [Quickstart](quickstart.md)).
- A ChatGPT Plus or Pro subscription.
- A Codex-OAuth proxy. Several exist; any that speaks OpenAI-compatible
  `/v1/chat/completions` works:
  - [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI) (also published
    as `ai-cli-proxy-api`), confirmed working end to end in
    [#436](https://github.com/mozilla-ai/otari/issues/436).
  - Hermes' bundled `codex-proxy`.

## 1. Run the Codex-OAuth proxy

Follow the proxy's own docs to log in with your subscription and start it. The
two things you need out of that step:

- the local base URL it serves, for example `http://localhost:8317/v1`, and
- whether it requires a **downstream key** (a token you set on the proxy and
  then present to it) or serves keyless.

Confirm the proxy answers before wiring Otari to it:

```bash
curl -sS http://localhost:8317/v1/models \
  -H "Authorization: Bearer <downstream-key-if-any>"
```

You should get back a list of models the subscription exposes (Codex serves
GPT-5 class models; the exact ids come from the proxy).

## 2. Register the proxy as a provider in Otari

Add the proxy as a named OpenAI-compatible instance under `providers:` in your
`config.yml`. Give it a custom instance name and set `provider_type: openai`
(the aliases `openai-compatible` and `openai_compatible` also work):

```yaml
providers:
  chatgpt:                       # custom instance name; used in the model selector
    provider_type: openai-compatible
    api_base: "http://localhost:8317/v1"
    api_key: ${CHATGPT_PROXY_KEY}   # the proxy's downstream key
```

If the proxy is keyless, drop `api_key` entirely. The
[keyless-custom-endpoint fix (#423)](https://github.com/mozilla-ai/otari/pull/423)
lets Otari call a keyless custom `api_base` without a `MissingApiKeyError`, so a
local proxy that needs no key just works:

```yaml
providers:
  chatgpt:
    provider_type: openai-compatible
    api_base: "http://localhost:8317/v1"
    # no api_key: the proxy handles auth via its own Codex OAuth session
```

Models on this instance are addressed as `chatgpt:<model>`, so the selector
carries the instance name rather than the raw `openai:` provider. If the proxy
does not expose `/v1/models`, declare the served ids so they still show up in
`GET /v1/models`:

```yaml
providers:
  chatgpt:
    provider_type: openai-compatible
    api_base: "http://localhost:8317/v1"
    models:
      - gpt-5
      - gpt-5-codex
```

See [Named provider instances](models.md#named-provider-instances) for the full
behavior.

## 3. Price the model (optional but recommended)

Because the subscription bills a flat fee, there is no per-token cost to import.
If you want cost analytics anyway, add a **counterfactual** price: the rate the
equivalent Platform API model would charge. Otari then reports what the same
usage would have cost on the API, which is a useful way to see the value you are
getting from the subscription.

```yaml
pricing:
  chatgpt:gpt-5:
    input_price_per_million: 1.25
    output_price_per_million: 10.00
```

Pricing is keyed on the instance name, so price `chatgpt:gpt-5`, not
`openai:gpt-5`. If you would rather not price it, set `require_pricing: false`
so the unpriced model is still served and logged with `cost: null`. See
[Configuration](configuration.md) for both knobs.

## 4. Route a request

Use an Otari client key and the `chatgpt:<model>` selector:

```python
from openai import OpenAI

client = OpenAI(api_key="gw-...", base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="chatgpt:gpt-5",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
```

Budgets, per-key limits, and usage logging all apply exactly as they do for any
other provider.

## Streaming and usage logging

If you stream responses (for example from an agent runtime like Hermes), set the
usage log writer to `batch`:

```yaml
log_writer_strategy: batch
```

With the default inline (`single`) strategy, a streaming client can disconnect
the moment it sees the SSE `[DONE]` marker, before Otari finishes writing the
usage row, so some records go uncommitted. The background `batch` writer decouples
the database write from the response stream and records usage reliably. See
[Configuration](configuration.md) for the field.

## Caveats

- **Unofficial backend.** This reaches the ChatGPT backend through Codex OAuth
  and a compatibility proxy, not the supported Platform API. The proxy can break
  when the backend changes.
- **Subscription caps apply.** Usage still counts against your ChatGPT
  subscription's limits; Otari's budgets sit on top and do not raise those caps.
- **An API key is sturdier.** For anything you depend on, an OpenAI Platform API
  key via the [OpenAI provider guide](providers/openai.md) is the supported
  route.
- **Keep it local.** The proxy holds your subscription's OAuth session. Run it on
  `localhost` and do not expose it publicly.

## See also

- [OpenAI provider guide](providers/openai.md) for the supported API-key route
- [Supported models](models.md) for the provider list and named-instance selectors
- [Configuration](configuration.md) for `log_writer_strategy`, `require_pricing`, and pricing
