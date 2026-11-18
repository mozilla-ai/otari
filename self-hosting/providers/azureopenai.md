# Connect Azure OpenAI

Serve the models you deploy on Azure OpenAI through Otari. The provider id is
`azureopenai`. Azure AI Inference is a different provider, `azure`, and Claude
on Microsoft Foundry is `azureanthropic`. See any-llm's
[supported providers](https://docs.mozilla.ai/any-llm/providers/) for the
current provider list and [Models](https://docs.otari.ai/reference/gateway/models) for Otari's model-name format.

## Before you start

- A running Otari with `OTARI_SECRET_KEY` set, and its master key. The
  [Self-hosted quickstart](../self-hosted-quickstart.md) sets up both.
- An Azure OpenAI or Microsoft Foundry resource with at least one model
  deployment, and a credential: either a resource key from that resource's
  Keys and Endpoint page, or a Microsoft Entra access token for the scope
  `https://ai.azure.com/.default` issued to an identity with the Cognitive
  Services OpenAI User role on the resource. Microsoft's
  [v1 API](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle)
  names that role and scope, and its
  [chat completions quickstart](https://learn.microsoft.com/azure/ai-foundry/openai/chatgpt-quickstart)
  shows where the key and the resource endpoint are.

## Add the key

Azure OpenAI has no default endpoint, so the provider takes the resource
endpoint as well as the key. Add it as a hosted provider, on **Hosted
providers** in the dashboard's deployment view (the endpoint goes in
**Endpoint**) or through the API:

```bash
curl -X POST http://localhost:8000/api/v1/hosted-providers \
  -H "Authorization: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "azureopenai",
    "api_key": "...",
    "api_base": "https://YOUR-RESOURCE-NAME.openai.azure.com"
  }'
```

`https://YOUR-RESOURCE-NAME.openai.azure.com` and
`https://YOUR-RESOURCE-NAME.services.ai.azure.com` are both resource endpoints.
The provider calls the v1 route: a URL that already ends in `/openai/v1` is left
as it is, and `/openai/v1` is appended to any other URL. Give the resource
endpoint. A legacy deployment URL such as
`https://YOUR-RESOURCE-NAME.openai.azure.com/openai/deployments/gpt-5-mini` has
`/openai/v1` appended as well, and the call misses the resource.

A Microsoft Entra token goes in `api_key` the same way a resource key does.
Otari stores it and sends it as given, so when it expires the caller sees HTTP
424 with code `provider_credential_rejected` until you replace it.

To keep the provider in configuration instead, declare it under
`hosted_providers:` in `config.yml` (see
[Hosted providers in config.yml](../configuration.md#hosted-providers-in-configyml)):

```yaml
hosted_providers:
  azureopenai:
    api_key: ${AZURE_OPENAI_API_KEY}
    api_base: https://YOUR-RESOURCE-NAME.openai.azure.com
```

Otari reads `AZURE_OPENAI_API_KEY` only if a declaration names it; see
[Configuring a provider](https://docs.otari.ai/reference/gateway/models#configuring-a-provider).

### Optional settings

`client_args` are passed through to the underlying client, for options such as
a timeout:

```json
{
  "provider": "azureopenai",
  "api_key": "...",
  "api_base": "https://YOUR-RESOURCE-NAME.openai.azure.com",
  "client_args": {"timeout": 60}
}
```

Leave `api_version` and `azure_deployment` out of `client_args`. The provider
always calls `/openai/v1/` and reads the deployment name from the model on the
request. A dated `api_version` is rejected, and `OPENAI_API_VERSION` is ignored
when it is set to anything other than `v1`. See
[API version lifecycle](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle).

## Choose a model

Models are addressed as `azureopenai:<deployment>`. Everything after the colon
is the deployment name you chose for that model:

```text
azureopenai:gpt-5-mini
azureopenai:gpt-5.4
azureopenai:prod-chat
```

The v1 route takes the deployment name in `model` (see Microsoft's
[Endpoints for Microsoft Foundry Models](https://learn.microsoft.com/azure/foundry/foundry-models/concepts/endpoints)).
Azure's model list names the models the resource offers, not your deployments,
so the provider's roster in Otari holds model ids:

- A deployment named after its model id (`gpt-5-mini`) is on the roster, and is
  served once that model is switched on.
- A deployment with a name of its own (`prod-chat`) is not. Add it to the roster
  by hand, or turn on **Serve models not on its list**
  (`serves_unlisted_models`), and give it a price (see Troubleshooting).

### Messages through the Responses API

To serve `/v1/messages` through Azure's Responses API, keep the provider in the
`providers:` block of `config.yml` with `messages_api: responses`, a setting a
hosted provider cannot hold (see
[Messages through the Responses API](../configuration.md#messages-through-the-responses-api)):

```yaml
providers:
  azureopenai:
    api_key: ${AZURE_OPENAI_API_KEY}
    api_base: https://YOUR-RESOURCE-NAME.openai.azure.com
    messages_api: responses
```

Otari then sends reasoning, and carries it across turns, only when the
deployment name looks like a reasoning model: the `o` series (`o1`, `o3`,
`o4-mini`), `gpt-5`, or a name containing `codex`. For the `o` series and
`gpt-5`, the name has to start with the model family (`gpt-5-prod` works,
`prod-gpt-5` does not), and names containing `-chat` are excluded. Name such a
deployment after its model.

## Verify

Make a request with your Otari API key (`tk-...`) through the OpenAI client
(`pip install openai`). Replace `gpt-5-mini` with a deployment on your resource:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/api/v1", api_key="tk-...")
resp = client.chat.completions.create(
    model="azureopenai:gpt-5-mini",
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
| HTTP 424 (code `provider_not_configured`) | Nothing serves the model: no Azure OpenAI provider is configured, it is switched off, or the deployment is neither switched on on its roster nor served as an unlisted model. |
| HTTP 424 `The provider rejected the gateway's credentials` (code `provider_credential_rejected`) | Azure rejected the key Otari sent, or an Entra token expired. Check the key and that the endpoint is the resource that key belongs to. |
| HTTP 404, with Azure's own message, or `The requested model was not found on the provider` when Azure sent no message | The name after `azureopenai:` is not a deployment on that resource. |
| HTTP 400 `'api_version' is not supported for azureopenai. Azure OpenAI now uses /openai/v1/. Remove the dated version or set api_version="v1".` | `client_args` set a dated API version. Remove it, or set `api_version` to `v1`. |
| HTTP 400 `'azure_deployment' is not supported for azureopenai` | The deployment name was set on the client. Pass it as the model on the request, `azureopenai:<deployment>`, instead. |
| HTTP 502 `LLM provider error` | The provider has no endpoint, so it fails before it calls Azure. Set `api_base`. |
| HTTP 429 (code `upstream_rate_limited`) | Azure rate-limited the deployment. The response detail is Azure's own message. |
| HTTP 402 `No pricing is configured for model ...` | The deployment has no price and `require_pricing` rejected the request. A deployment named after a model id can take its default price with `default_pricing: true`; any other needs a price on the Hosted providers page or a `pricing:` entry keyed by the selector, such as `azureopenai:prod-chat`. |
