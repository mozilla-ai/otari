# Connect Gemini

Serve Google's Gemini models through Otari with a Gemini API key from Google AI
Studio. For Gemini on Google Cloud, use the `vertexai` provider instead. See
any-llm's [supported providers](https://docs.mozilla.ai/any-llm/providers/) for
the current provider list and [Models](../../reference/models.md) for Otari's model-name
format.

## Before you start

- A running Otari with `OTARI_SECRET_KEY` set, and its master key. The
  [Quickstart](../../home/quickstart.md) sets up both.
- A Gemini API key from <https://aistudio.google.com/apikey>.

## Add the key

Add Gemini as a hosted provider, on **Hosted providers** in the dashboard's
deployment view or through the API:

```bash
curl -X POST http://localhost:8000/api/v1/hosted-providers \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "gemini", "api_key": "AIza..."}'
```

Otari stores the key encrypted and offers every model Gemini lists on it; each
model with a price is switched on, and one nothing prices arrives switched off.
To keep the provider in configuration instead, declare it under
`hosted_providers:` in `config.yml` (see
[Hosted providers in config.yml](../../reference/configuration.md#hosted-providers-in-configyml)).
Otari does not read `GEMINI_API_KEY` from its environment unless a declaration
names it; see [Configuring a provider](../../reference/models.md#configuring-a-provider).

## Choose a model

Models are addressed as `gemini:<model>`:

```text
gemini:gemini-3-flash-preview
gemini:gemini-3.1-flash-lite
```

Everything after the colon is passed to Gemini. A model is served once it is
switched on under the provider on the Hosted providers page. Google retires
older models for new keys, so check the
[model list](https://ai.google.dev/gemini-api/docs/models) if a name is
rejected.

Chat completions take only text-generation models. Send an embedding model such
as `gemini-embedding-001` to `POST /api/v1/embeddings` instead (see the
[API reference](../../reference/api-reference.md)).

## Verify

Make a request with your Otari API key (`tk-...`) through the Otari Python SDK
(`pip install otari`; any OpenAI-compatible client also works):

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
| HTTP 424 (code `provider_not_configured`) | Nothing serves the model: no Gemini provider is configured, it is switched off, or the model is not switched on under it. Switch the model on under Gemini on the Hosted providers page. |
| HTTP 400 `API key not valid. Please pass a valid API key.` | Gemini rejected the key Otari sent. Replace the key on the Hosted providers page. |
| HTTP 404 `This model ... is no longer available to new users` | Google has retired that model for new keys. Use the model the message suggests. |
| HTTP 429 `You exceeded your current quota` (code `upstream_rate_limited`) | Your key's quota for that model is used up or is zero, which is common for Pro models on the free tier. Wait, pick another model, or enable billing in Google AI Studio. |
| HTTP 502 `The provider failed the request with HTTP 503` | Gemini is overloaded or failed on its side. These are retryable. |
| HTTP 402 `No pricing configured for model ...` | The model has no price and `require_pricing` rejected the request. Set a price for it on the Hosted providers page or in a `pricing:` entry, or enable `default_pricing: true`. |
