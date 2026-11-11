# Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is Nous Research's
agent. The [Otari provider plugin](https://github.com/mozilla-ai/hermes-provider-otari)
adds Otari as a model provider, so one Otari key reaches every model your
workspace serves.

## Quick start

You need Hermes Agent 0.22.5 or later and an Otari API key.

1. Install the plugin:

   ```bash
   hermes plugins install otari
   ```

2. Set your key, replacing `YOUR_OTARI_KEY`, and chat through Otari:

   ```bash
   hermes config set OTARI_API_KEY YOUR_OTARI_KEY
   hermes chat --provider otari -m anthropic:claude-sonnet-4-6
   ```

`hermes model` lists Otari too, with the models your key can use.

## Which gateway

| Gateway | `OTARI_BASE_URL` |
| --- | --- |
| otari.ai | Not needed |
| otari.ai, EU region | `https://eu.api.otari.ai/api/v1` |
| A gateway you run | Its URL plus `/api/v1`, for example `http://localhost:8000/api/v1` |

An EU key sent to the default host gets `421 Misdirected Request`; set the EU
URL.

## Choosing a model

`-m` takes any model your workspace's catalog lists: a `provider:model` id, a
routed id, or the name of a [routing policy](routing.md). See
[Model format](models.md#model-format).

Hermes's `--reasoning` level is sent as `reasoning_effort`, with `max` sent as
`xhigh`.

## See also

- [Pydantic AI](pydantic-ai.md)
- [LangChain](langchain.md)
