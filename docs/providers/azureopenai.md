# Azure OpenAI

Route requests to models you deploy on Azure OpenAI through Otari. The provider id is `azureopenai`. Azure AI Inference is a different provider, `azure`, and Claude on Microsoft Foundry is `azureanthropic`. See any-llm's [supported providers](https://docs.mozilla.ai/any-llm/providers/) for the current provider list and [Models](../models.md) for Otari's model-name format.

## What you'll set up

An `azureopenai` entry in your `config.yml` that reads a key from `AZURE_OPENAI_API_KEY` (or the key stored on the dashboard's Providers page) and names your resource endpoint, then a first request that Otari routes to one deployment on that resource.

## Prerequisites

- Otari running locally (see the [Quickstart](../quickstart.md))
- An Azure OpenAI or Microsoft Foundry resource with at least one model deployment, and a key from that resource's Keys and Endpoint page. Microsoft's [chat completions quickstart](https://learn.microsoft.com/azure/ai-foundry/openai/chatgpt-quickstart) shows where the key and the resource endpoint are.

## Configure

Azure OpenAI has no default endpoint. Add the resource under `providers:`:

```yaml
providers:
  azureopenai:
    api_key: ${AZURE_OPENAI_API_KEY}
    api_base: https://YOUR-RESOURCE-NAME.openai.azure.com
```

`api_base` is the resource endpoint from that page. `https://YOUR-RESOURCE-NAME.openai.azure.com` and `https://YOUR-RESOURCE-NAME.services.ai.azure.com` are both resource endpoints. The provider calls the v1 route: a URL that already ends in `/openai/v1` is left as it is, and `/openai/v1` is appended to any other URL. Give the resource endpoint. A legacy deployment URL such as `https://YOUR-RESOURCE-NAME.openai.azure.com/openai/deployments/gpt-5-mini` has `/openai/v1` appended as well, and the call misses the resource.

A bearer token belongs in `api_key` the same way a resource key does. The client sends that value as its API key. A token-refresh callback cannot be written in `config.yml`, so a token set here is used as given until it expires.

Export the key before starting Otari:

```bash
export AZURE_OPENAI_API_KEY=...
```

> `AZURE_OPENAI_API_KEY` alone is not enough, and neither is a bare `azureopenai:` entry: Otari reads the variable only where an entry names it, as the `api_key: ${AZURE_OPENAI_API_KEY}` line above does. The dashboard's Providers page and an organization's own key work too. When the entry sets no `api_base`, the provider reads `AZURE_OPENAI_ENDPOINT`.

### Optional settings

`client_args` are passed through to the underlying client:

```yaml
providers:
  azureopenai:
    api_key: ${AZURE_OPENAI_API_KEY}
    api_base: https://YOUR-RESOURCE-NAME.openai.azure.com
    client_args:
      timeout: 60
```

Leave `api_version` and `azure_deployment` out of `client_args`. The provider always calls `/openai/v1/` and reads the deployment name from the model on the request. A dated `api_version` is rejected. `OPENAI_API_VERSION`, when it is set to anything other than `v1`, is ignored. See [API version lifecycle](https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle).

## Choose a model

Models are addressed as `azureopenai:<deployment>`. Everything after the colon is the deployment name you chose for that model:

```text
azureopenai:gpt-5-mini
azureopenai:gpt-5-nano
azureopenai:gpt-5.4
```

Those names match catalog model ids, so bundled pricing can price a deployment that uses the model id as its name. A deployment with a name of its own is called by that name (`azureopenai:prod-chat`). Microsoft describes this on [Endpoints for Microsoft Foundry Models](https://learn.microsoft.com/azure/foundry/foundry-models/concepts/endpoints): the v1 route takes the deployment name in `model`.

With `messages_api: responses`, Otari sends reasoning and carries it across turns only when the deployment name looks like a reasoning model: the `o` series (`o1`, `o3`, `o4-mini`), `gpt-5` except names containing `-chat`, or a name containing `codex`. Name that deployment after the model (`gpt-5-mini`). See [Messages through the Responses API](../configuration.md#messages-through-the-responses-api).

## Verify

If you have not already started Otari and created a client key, follow the [Quickstart](../quickstart.md) first.

Then make a request with the Otari Python SDK (`pip install otari`; any
OpenAI-compatible client also works), using an `azureopenai:<deployment>` selector.
Replace `gpt-5-mini` with a deployment on your resource:

```python
from otari import OtariClient

client = OtariClient(api_base="http://localhost:8000", api_key="tk-...")
resp = client.completion(
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
| HTTP 424 `The provider rejected the gateway's credentials` (code `provider_credential_rejected`) | Azure rejected the `api_key` Otari sent. Check the key (or `AZURE_OPENAI_API_KEY`) and that `api_base` is the resource that key belongs to. |
| HTTP 424 `No credential is configured for provider 'azureopenai'` (code `provider_not_configured`) | The entry names no key. Set `api_key` on the entry. A bare `AZURE_OPENAI_API_KEY` does not configure the provider. |
| HTTP 404, with Azure's own message, or `The requested model was not found on the provider` when Azure sent no message | The name after `azureopenai:` is not a deployment on that resource. |
| HTTP 400 `'api_version' is not supported for azureopenai. Azure OpenAI now uses /openai/v1/. Remove the dated version or set api_version="v1".` | `client_args` set a dated API version. Remove it, or set `api_version` to `v1`. |
| HTTP 400 `'azure_deployment' is not supported for azureopenai` | The deployment name was set on the client. The rest of the message says to pass it as the model on the request, which is `azureopenai:<deployment>`. |
| HTTP 502 `LLM provider error` | The entry has no `api_base` and `AZURE_OPENAI_ENDPOINT` is unset, so the provider raises before it calls Azure. The response detail stays the generic provider error. |
| HTTP 429 (code `upstream_rate_limited`) | Azure rate-limited the deployment. The response detail is Azure's own message. |
| HTTP 402 `No pricing is configured for model ...` | `require_pricing` is on and nothing prices that deployment name. Bundled pricing matches a name in the catalog, such as `gpt-5-mini`. A name of your own needs a `pricing:` entry. |

## Pricing (optional)

With `default_pricing: true`, a deployment whose name is a catalog model id is priced from the bundled list. In that list, `gpt-5-mini` is 0.25 USD per million input tokens and 2.00 USD per million output tokens. To set a price yourself, add a `pricing:` entry keyed by the same selector the request uses:

```yaml
pricing:
  azureopenai:gpt-5-mini:
    input_price_per_million: 0.25
    output_price_per_million: 2.00
```

See [Configuration](../configuration.md) for all options.
