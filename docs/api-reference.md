# API reference

Otari serves an OpenAPI document at `/openapi.json` and interactive API docs at
`/docs` by default. The repository also commits the generated
[OpenAPI specification](public/openapi.json) and
[Postman collection](public/otari.postman_collection.json). Those generated
artifacts are the source of truth for paths, parameters, and schemas.

The default server address is `http://localhost:8000`. OpenAI-compatible clients
normally use `http://localhost:8000/v1` as their base URL.

## Authentication

In standalone and hosted mode, Otari accepts a local API key or the master key in
any of these forms:

```text
Authorization: Bearer <token>
Otari-Key: <token>
Otari-Key: Bearer <token>
x-api-key: <token>
```

Use API keys for inference. The master key is a deployment-wide administrative
credential. Dashboard sessions authenticate browser requests, but deployment-wide
operations also require operator authority.

In hybrid mode, the completion APIs accept an otari.ai user token through
`Authorization: Bearer <token>`. Local API keys and management APIs are not used.

## Availability by mode

| Surface | Standalone | Hosted | Hybrid |
| --- | --- | --- | --- |
| Health and `/v1/bootstrap` | Yes | Yes | Yes |
| Chat, Messages, and Responses | Yes | No | Yes |
| Other inference APIs | Yes | No | No |
| `/v1/models` | Yes | Yes | No |
| Management APIs | Yes | Yes | No |

Hosted mode is a control plane. Its inference paths return a descriptive `404`
and, when configured, the data-plane URL to use instead. See [Modes](modes.md).

## Core inference APIs

Otari implements three completion surfaces:

- `POST /v1/chat/completions`, OpenAI Chat Completions
- `POST /v1/messages` and `/v1/messages/count_tokens`, Anthropic Messages
- `POST /v1/responses`, OpenAI Responses

Standalone mode also serves embeddings, images, audio, files, batches,
moderations, rerank, and search. Provider support differs by endpoint, so use
`GET /v1/models` and the OpenAPI document for the deployment you are calling.

## Search

`POST /v1/search` and `POST /v1/search/{search_tool_name}` run a configured
search tool directly. This is separate from `otari_web_search`, which lets a
model request searches during a completion. Both are described in
[Built-in tools](tools.md).

Search-tool management lives under `/v1/search-tools`. The generated OpenAPI
document describes the supported providers, filters, and management schemas.

## Routing policies

Routing-policy management lives under `/v1/routing/policies`; learned-routing
examples and status live under `/v1/routing/preferences` and `/v1/routing/status`.
See [Routing policies](routing.md) for configuration and behavior, and OpenAPI for
the request schemas.

## Provider error details

Otari may return a short, sanitized provider diagnostic when the upstream
provider rejects something the caller can fix, such as a model name or request
parameter. Credentials, URLs, account identifiers, and reflected payloads are
removed.

Gateway-side failures use fixed public messages. Diagnose them with protected
logs and safe metadata such as request ID, provider, model, and status. Do not
log provider keys, prompts, responses, or raw upstream bodies.

## Keeping generated clients current

API changes must regenerate both committed artifacts:

```bash
uv run python scripts/generate_openapi.py
make postman
make openapi-check
make postman-check
```
