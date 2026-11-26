# Connect otari.ai

Serve models from many providers through one key by connecting your Otari to
[otari.ai](https://otari.ai), the hosted Otari. otari.ai is a paid service, and
a request served on it leaves your gateway for otari.ai. For what a connection
stores and how its models are priced, see
[Connecting otari.ai](../hosted-providers.md#connecting-otariai).

## Before you start

- A running Otari with `OTARI_SECRET_KEY` set, and its master key. The
  [Self-hosted quickstart](../self-hosted-quickstart.md) sets up both.
- An otari.ai account. You can create one while connecting.

## Connect

1. Open the dashboard and go to **Hosted providers** in the deployment view.
2. Choose **Connect otari.ai**. Otari opens otari.ai's approval page in a new
   tab and shows the code it is waiting on.
3. Sign in to otari.ai and approve the code.

Otari stores the key otari.ai issues as a hosted provider named `otari-ai`,
lists its models and prices them, so they are ready to use straight away.

### Without the button

Create a key on otari.ai's [API keys page](https://otari.ai/dashboard/#/keys)
and add it as a hosted provider:

```bash
curl -X POST http://localhost:8000/api/v1/hosted-providers \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"provider": "otari", "name": "otari-ai", "api_key": "<your otari.ai key>", "api_base": "https://api.otari.ai"}'
```

## Choose a model

otari.ai names its models `<vendor>/<model>`. Address one as
`otari-ai:<vendor>/<model>`, which always reaches otari.ai:

```text
otari-ai:openai/gpt-4o-mini
```

The **Models** page lists every model the connection offers.

## Verify

Make a request with your Otari API key (`tk-...`) through the OpenAI client
(`pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/api/v1", api_key="tk-...")
resp = client.chat.completions.create(
    model="otari-ai:openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(resp.choices[0].message.content)
```

## Disconnect

Remove the `otari-ai` provider on the Hosted providers page, then revoke its
key on otari.ai's [API keys page](https://otari.ai/dashboard/#/keys): removing
the provider deletes only your gateway's copy of the key.
