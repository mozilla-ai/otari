# Use with n8n

[n8n](https://n8n.io) ships OpenAI and Anthropic credentials whose base URL can
be overridden, and Otari serves both wire formats
(`POST /api/v1/chat/completions` and `POST /api/v1/messages`) in standalone and
hybrid modes. Pointing an n8n credential at Otari routes every workflow that
uses it through the gateway, so budgets, usage tracking, and traces apply to
your automations without a custom node.

## Quick start

In n8n, create an **OpenAI** credential (Credentials, Add credential, OpenAI):

| Field    | Value                                   |
|----------|-----------------------------------------|
| API Key  | your Otari API key or `tk_` user token  |
| Base URL | `http://localhost:8000/api/v1`          |

Leave Organization ID empty. The base URL is the Otari API root, `/api/v1`
(n8n appends `/chat/completions` and `/models` itself). Point it at the gateway
you are actually using: `http://localhost:8000/api/v1` for local standalone
development, your self-hosted gateway URL plus `/api/v1` when connected to
otari.ai, or `https://api.otari.ai/api/v1` when using otari.ai's own gateway.

Saving the credential makes n8n test it with `GET /models` under that base
URL, which Otari answers with the models your key can reach. A failed test
means the key or URL is wrong, not that the integration needs more setup.

Then add an **OpenAI Chat Model** node to an AI Agent or chain, select the
credential, and pick a model from the dropdown. The key is sent as
`Authorization: Bearer <token>`, which Otari accepts for both standalone API
keys and connected user tokens.

## Choosing a model

The model dropdown lists what `GET /api/v1/models` returns, in Otari's
`provider:model` spelling. When the base URL is not `api.openai.com`, n8n
skips its GPT-only filter, so every provider your key can reach appears.

- **Standalone, any configured provider:** `openai:gpt-4o`,
  `anthropic:claude-sonnet-4-6`, `mistral:mistral-large-latest`.
- **Connected to otari.ai, managed models:** `mzai:<catalog-id>`, for example
  `mzai:moonshotai/Kimi-K2.6`. These run only through otari.ai's own gateway.
- **Connected to otari.ai, your own provider keys:** `openai:gpt-4o` or
  `openai/gpt-4o`, plus the equivalent Anthropic or Mistral forms. An `mzai:`
  prefix selects the managed catalog, so adding it to a proprietary model
  misroutes it.

n8n warns that on a non-OpenAI base URL not every listed model supports tools
or JSON mode. Otari rejects a model and endpoint combination it knows the
provider cannot serve before dispatch; see [Models](models.md).

## Tool calling

The OpenAI Chat Model node has a **Supports strict tool calling** option, on by
default. Leave it on for OpenAI models. Turn it off when the model you route to
rejects strict tool schemas, which is the usual cause of a tool-enabled agent
failing on the first call against a non-OpenAI provider.

## Anthropic Messages instead

To drive the **Anthropic Chat Model** node through Otari, create an
**Anthropic** credential with your Otari token as the API key and the base URL
set to the gateway root plus `/api`, for example `http://localhost:8000/api`.
The Anthropic client appends `/v1/messages` and `/v1/models` itself, so this
base URL ends in `/api`, not `/api/v1`. The key travels in the `x-api-key`
header, which Otari accepts the same way as a Bearer token.

Use this when a workflow depends on Anthropic-specific request features;
otherwise the OpenAI credential reaches Anthropic models too, through
`anthropic:<model>`.

## Other endpoints

The n8n **OpenAI** node (as opposed to the chat model sub-node) uses the same
credential for embeddings, images, and audio, which Otari serves at
`/api/v1/embeddings`, `/api/v1/images/*`, and `/api/v1/audio/*`. Anything the
built-in nodes do not cover is reachable from the **HTTP Request** node with the
same credential selected under Authentication, Predefined Credential Type,
OpenAI; see the [API reference](api-reference.md).

## See also

- [Use with opencode](use-with-opencode.md): register Otari as an
  OpenAI-compatible provider in the opencode CLI the same way.
- [Use with Claude Code](use-with-claude-code.md): drive the Claude Code CLI
  through the same Otari via the Anthropic Messages API.
- [Models](models.md): the `provider:model` format and catalog spellings.
