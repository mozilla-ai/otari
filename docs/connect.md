# Connect to Otari

Every request to Otari needs two things: the base URL of the Otari you use, and
an API key. This page is where the docs give the base URLs; other pages link
here or use `$OTARI_BASE_URL`.

## Base URL

| Where Otari runs | `OTARI_BASE_URL` |
| --- | --- |
| otari.ai, an account in the United States | `https://api.otari.ai/api` |
| otari.ai, an account in the European Union | `https://eu.api.otari.ai/api` |
| Your own gateway | `https://<your-otari-host>/api`, or `http://localhost:8000/api` on your own machine |

```bash
export OTARI_BASE_URL=https://api.otari.ai/api
export OTARI_API_KEY=<your-api-key>
```

The base URL ends in `/api`. Every endpoint lives under it, so an OpenAI-style
request goes to `$OTARI_BASE_URL/v1/chat/completions`, an Anthropic-style one to
`$OTARI_BASE_URL/v1/messages`, and so on through the [API reference](api-reference.md).

Leaving `/api` off is the most common mistake. On otari.ai, a request to
`https://api.otari.ai/v1/...` never reaches Otari: a small one answers `404 Not
Found`, and a large one, such as any Claude Code request, answers `403
Forbidden`, which Claude Code reports as "Failed to authenticate".

An otari.ai key belongs to one region. Sent to the other region's URL, it
answers `421` naming the host it belongs to.

## API key

Create an API key in the workspace's **API keys** tab, as
[API keys](api-keys.md) describes. A key from otari.ai starts with `otk_`; one
from your own gateway starts with `tk-`. Send it in any of the header forms in
[Authentication](api-reference.md#authentication), most commonly:

```text
Authorization: Bearer <your-api-key>
```

## Point an SDK at Otari

SDKs append their own paths, so each takes the base URL in the form it expects:

| Client | Base URL setting |
| --- | --- |
| OpenAI SDKs and OpenAI-compatible tools | `$OTARI_BASE_URL/v1`, for example `https://api.otari.ai/api/v1` |
| Anthropic SDKs and Claude Code | `$OTARI_BASE_URL`, for example `https://api.otari.ai/api`, since they append `/v1/messages` |

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.otari.ai/api/v1", api_key="<your-api-key>")
response = client.chat.completions.create(
    model="openai:gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(response.choices[0].message.content)
```

Models are named `provider:model`, or by catalog id; see [Models](models.md).
Setup for one tool or framework is under Guides, starting with
[Use with Claude Code](use-with-claude-code.md).
