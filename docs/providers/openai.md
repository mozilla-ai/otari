# OpenAI

Route requests to OpenAI models (GPT-4o, GPT-4.1, the o-series, and others) through Otari. For the full provider list and the model-name format, see [Models](../models.md).

## What you'll set up

An OpenAI provider entry in your `config.yml` (or a single environment variable), then a first request that Otari routes to OpenAI.

## Prerequisites

- Otari running locally (see the [Quickstart](../quickstart.md))
- An OpenAI account and an API key from <https://platform.openai.com/api-keys>

## Configure

Add OpenAI under `providers:` in your `config.yml`:

```yaml
providers:
  openai:
    api_key: "sk-..."                        # your OpenAI API key
    # api_base: "https://api.openai.com/v1"  # optional; override for a proxy or compatible endpoint
```

To keep your provider key out of the file with environment interpolation:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

...then export it before starting Otari:

```bash
export OPENAI_API_KEY=sk-...
```

> Otari routes through [any-llm](https://pypi.org/project/any-llm-sdk/), so a standard `OPENAI_API_KEY` in the environment is picked up automatically even if you don't list the provider explicitly.

### Optional settings

`client_args` are passed through to the underlying `any-llm` provider client, for options such as custom headers or timeouts:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    client_args:
      timeout: 60
      custom_headers:
        OpenAI-Organization: "org-..."
```

## Choose a model

Models are addressed as `openai:<model>`:

```
openai:gpt-4o
openai:gpt-4o-mini
openai:o4-mini
```

Everything after the colon is passed straight to OpenAI, so any model your key can access works.

## Verify

If you have not already started Otari and created a client key, follow the [Quickstart](../quickstart.md) through step 3 first.

Then make a request with any OpenAI client, using an `openai:<model>` selector:

```python
from openai import OpenAI

client = OpenAI(api_key="gw-...", base_url="http://localhost:8000/v1")
resp = client.chat.completions.create(
    model="openai:gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
```

Expected (sample) output:

```
Hello, nice to meet you!
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| HTTP 502 `The provider rejected the gateway's credentials` | The `api_key` Otari sent to OpenAI is invalid or lacks access. Check the `api_key` (or `OPENAI_API_KEY`) configured on the gateway. |
| HTTP 404 `The requested model was not found on the provider` | The name after `openai:` is not a model your key can access, or the model name is misspelled. |
| HTTP 502 `LLM provider error` | Generic fallback: Otari reached the provider but the upstream call failed in a way it could not classify (for example a missing provider key, a provider-side 5xx, or a connection error). |
| HTTP 402 `No pricing configured for model ...` | Otari could not resolve pricing for that model and `require_pricing` rejected the request. Add a `pricing:` entry, enable `default_pricing: true` for bundled fallback pricing, or use a model that already has pricing coverage. |
| HTTP 401 `Invalid master key` on `/v1/keys` | You passed a client (`gw-...`) key, or the wrong master key, instead of the configured master key. |

## Pricing (optional)

To track cost for a model, add a `pricing:` entry in `config.yml`:

```yaml
pricing:
  openai:gpt-4o-mini:
    input_price_per_million: 0.15
    output_price_per_million: 0.60
```

See [Configuration](../configuration.md) for all options.
