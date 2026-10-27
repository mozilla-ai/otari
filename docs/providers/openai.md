# Connect OpenAI

Serve OpenAI models (GPT-4o, GPT-4.1, the o-series, and others) through Otari
with your OpenAI API key. See any-llm's
[supported providers](https://docs.mozilla.ai/any-llm/providers/) for the
current provider list and [Models](../models.md) for Otari's model-name format.

## Before you start

- A running Otari with `OTARI_SECRET_KEY` set, and its master key. The
  [Self-hosted quickstart](../self-hosted-quickstart.md) sets up both.
- An OpenAI API key from <https://platform.openai.com/api-keys>.

## Add the key

Add OpenAI as a hosted provider, on **Hosted providers** in the dashboard's
deployment view or through the API:

```bash
curl -X POST http://localhost:8000/api/v1/hosted-providers \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "api_key": "sk-..."}'
```

Otari stores the key encrypted and offers every model OpenAI lists on it; each
model with a price is switched on, and one nothing prices arrives switched off.
To keep the provider in configuration instead, declare it under
`hosted_providers:` in `config.yml` (see
[Hosted providers in config.yml](../configuration.md#hosted-providers-in-configyml)).
Otari does not read `OPENAI_API_KEY` from its environment unless a declaration
names it; see [Configuring a provider](../models.md#configuring-a-provider).

### Optional settings

The same body takes `api_base`, for a proxy or another OpenAI-compatible
endpoint, and `client_args`, passed through to the underlying `any-llm`
provider client for options such as custom headers or timeouts:

```json
{
  "provider": "openai",
  "api_key": "sk-...",
  "client_args": {
    "timeout": 60,
    "custom_headers": {"OpenAI-Organization": "org-..."}
  }
}
```

## Choose a model

Models are addressed as `openai:<model>`:

```text
openai:gpt-4o
openai:gpt-4o-mini
openai:o4-mini
```

Everything after the colon is passed to OpenAI. A model is served once it is
switched on under the provider on the Hosted providers page.

## Verify

Make a request with your Otari API key (`tk-...`) through the Otari Python SDK
(`pip install otari`; any OpenAI-compatible client also works):

```python
from otari import OtariClient

client = OtariClient(api_base="http://localhost:8000", api_key="tk-...")
resp = client.completion(
    model="openai:gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
```

Expected (sample) output:

```text
Hello, nice to meet you!
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| HTTP 401 `Invalid master key` when adding the provider | You sent an API key (`tk-...`) or the wrong master key. `/api/v1/hosted-providers` takes the master key. |
| HTTP 424 (code `provider_not_configured`) | Nothing serves the model: no OpenAI provider is configured, it is switched off, or the model is not switched on under it. Switch the model on under OpenAI on the Hosted providers page. |
| HTTP 424 `The provider rejected the gateway's credentials` (code `provider_credential_rejected`) | OpenAI rejected the key Otari sent as invalid or lacking access. Replace the key on the Hosted providers page. |
| HTTP 404 with OpenAI's own "model not found" text | The name after `openai:` is misspelled or not a model your key can access; the detail carries OpenAI's message. `The requested model was not found on the provider` appears instead when OpenAI sent no message. |
| HTTP 502 `The upstream provider account is out of credit or over its billing limit.` | The OpenAI account has no credit left or hit its billing hard limit. Top it up, or point the model at a provider that has credit. The usage log keeps the status OpenAI returned. |
| HTTP 502 `The provider failed the request with HTTP 5xx`, `The provider closed the connection before answering`, or `The gateway could not connect to the provider` | OpenAI failed on its side or the connection broke. These are retryable. `LLM provider error` is the same answer for a failure Otari could not place. |
| HTTP 402 `No pricing configured for model ...` | The model has no price and `require_pricing` rejected the request. Set a price for it (below), or enable `default_pricing: true`. |

## Pricing (optional)

Set a model's price on the Hosted providers page, or add a `pricing:` entry in
`config.yml`:

```yaml
pricing:
  openai:gpt-4o-mini:
    input_price_per_million: 0.15
    output_price_per_million: 0.60
```

See [Configuration](../configuration.md) for all options.
