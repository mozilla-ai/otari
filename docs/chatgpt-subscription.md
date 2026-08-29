# Use with a ChatGPT subscription

A ChatGPT subscription is separate from the OpenAI Platform API and does not
provide a standard API key. Otari can reach subscription-backed models only
through a third-party Codex OAuth proxy that exposes an OpenAI-compatible API.

```text
application -> Otari -> local OAuth proxy -> ChatGPT subscription
```

This is an unofficial integration. OpenAI may change the backend, subscription
limits still apply, and a third-party proxy may create account or terms-of-service
risk. Use an OpenAI Platform API key for production workloads.

## Prerequisites

- A local Otari deployment
- A ChatGPT Plus or Pro subscription
- A Codex OAuth proxy that serves an OpenAI-compatible API, such as
  [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)

Run the proxy on a private address and follow its authentication instructions.
It holds your subscription session and should not be exposed publicly.

## Verify the proxy

Ask the proxy for its model catalog, including its downstream key if configured:

```bash
curl http://localhost:8317/v1/models \
  -H "Authorization: Bearer $CHATGPT_PROXY_KEY"
```

If it does not implement `/v1/models`, verify with a chat completion and declare
its model IDs explicitly in Otari.

## Register it in Otari

Create a named OpenAI-compatible provider instance:

```yaml
providers:
  chatgpt:
    provider_type: openai-compatible
    api_base: "http://localhost:8317/v1"
    api_key: ${CHATGPT_PROXY_KEY}
    models:
      - gpt-5
```

Remove `api_key` when the proxy is intentionally keyless. The `models` list is
needed only when the proxy has no working model-listing endpoint.

Models use the instance name, for example `chatgpt:gpt-5`. See
[Named provider instances](models.md#named-provider-instances).

## Pricing

A subscription has no per-token invoice. You can either:

- Leave the model unpriced and set `require_pricing: false`.
- Configure a counterfactual API price for analytics.

```yaml
pricing:
  chatgpt:gpt-5:
    input_price_per_million: 1.25
    output_price_per_million: 10.00
```

Configured pricing also drives budget enforcement. A counterfactual price can
therefore exhaust a real Otari budget even though the subscription charged no
per-request fee.

## Route a request

```python
from openai import OpenAI

client = OpenAI(api_key="gw-...", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="chatgpt:gpt-5",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

Provider routing, API-key access, usage logging, and configured budgets work as
they do for any other named instance.

## Caveats

- The proxy may break when its upstream backend changes.
- Subscription usage caps remain in force.
- Keep the proxy and its OAuth session private.
- Model IDs and capabilities come from the proxy, not from Otari.
- API pricing attached to the instance is an estimate, not the subscription's
  actual charge.

For the supported API-key path, use the [OpenAI provider guide](providers/openai.md).
