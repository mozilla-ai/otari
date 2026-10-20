# Gemini

Route requests to Google's Gemini models through Otari, using a Gemini API key from Google AI Studio. For Gemini on Google Cloud, use the `vertex` provider instead. See any-llm's [supported providers](https://docs.mozilla.ai/any-llm/providers/) for the current provider list and [Models](../models.md) for Otari's model-name format.

## What you'll set up

A Gemini provider entry in your `config.yml` (or a single environment variable), then a first request that Otari routes to Gemini.

## Prerequisites

- Otari running locally (see the [Quickstart](../quickstart.md))
- A Gemini API key from <https://aistudio.google.com/apikey>

## Configure

Add Gemini under `providers:` in your `config.yml`:

```yaml
providers:
  gemini:
    api_key: "AIza..."   # your Gemini API key
```

To keep your provider key out of the file with environment interpolation:

```yaml
providers:
  gemini:
    api_key: ${GEMINI_API_KEY}
```

...then export it before starting Otari:

```bash
export GEMINI_API_KEY=AIza...
```

> Otari routes through [any-llm](https://pypi.org/project/any-llm-sdk/), so a standard `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment is picked up automatically even if you don't list the provider explicitly.

## Choose a model

Models are addressed as `gemini:<model>`:

```text
gemini:gemini-3-flash-preview
gemini:gemini-3.1-flash-lite
```

Everything after the colon is passed straight to Gemini, so any model your key can access works. Google retires older models for new keys, so check the [model list](https://ai.google.dev/gemini-api/docs/models) if a name is rejected.

Chat completions take only text-generation models. Embedding models such as `gemini-embedding-001` do not work there; send them to `POST /api/v1/embeddings` instead (see the [API reference](../api-reference.md)).

## Verify

If you have not already started Otari and created a client key, follow the [Quickstart](../quickstart.md) first.

Then make a request with the Otari Python SDK (`pip install otari`; any
OpenAI-compatible client also works), using a `gemini:<model>` selector:

```python
from otari import OtariClient

client = OtariClient(api_base="http://localhost:8000", api_key="tk-...")
resp = client.completion(
    model="gemini:gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
```

Expected (sample) output:

```text
Hello there, how are you?
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| HTTP 400 `API key not valid. Please pass a valid API key.` | Gemini rejected the `api_key` Otari sent. Check the `api_key` (or `GEMINI_API_KEY`) configured on the gateway. |
| HTTP 404 `This model ... is no longer available to new users` | Google has retired that model for new keys. Use the model the message suggests. |
| HTTP 429 `You exceeded your current quota` (code `upstream_rate_limited`) | Your key's quota for that model is used up or is zero, which is common for Pro models on the free tier. Wait, pick another model, or enable billing in Google AI Studio. |
| HTTP 502 `The provider failed the request with HTTP 503` | Gemini is overloaded or failed on its side. These are retryable. |
| HTTP 402 `No pricing configured for model ...` | Otari could not resolve pricing for that model and `require_pricing` rejected the request. Add a `pricing:` entry, enable `default_pricing: true` for bundled fallback pricing, or use a model that already has pricing coverage. |
