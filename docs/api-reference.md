# API reference

Otari serves an OpenAPI document at `/api/v1/openapi.json` and interactive API
docs at `/api/v1/docs` by default. The repository also commits the generated
[OpenAPI specification](public/openapi.json) and
[Postman collection](public/otari.postman_collection.json). Those generated
artifacts are the source of truth for paths, parameters, and schemas.

The default server address is `http://localhost:8000`. The API is mounted at
`/api/v1`. OpenAI-compatible clients use `http://localhost:8000/api/v1` as their
base URL; Anthropic-compatible clients use `http://localhost:8000/api`, because
their SDK appends `/v1/messages` itself. OTLP ingest is a sibling at `/otlp`:
point `OTEL_EXPORTER_OTLP_ENDPOINT` at `http://localhost:8000/otlp` and the
exporter appends `/v1/traces`, `/v1/logs` and `/v1/metrics`.

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
operations also require operator authority. Where a tenant needs one of those
operations, a separately scoped endpoint serves it to the caller's own
organization: `/api/v1/organizations/me/usage` for usage, and
`/api/v1/organizations/me/keys` for a member's own API keys.

In hybrid mode, the generation APIs and the `/api/v1/mcp` endpoints accept an
otari.ai user token through `Authorization: Bearer <token>`. Local API keys and
management APIs are not used.

## Availability by mode

| Surface | Standalone | Hosted | Hybrid |
| --- | --- | --- | --- |
| Health and `/api/v1/bootstrap` | Yes | Yes | Yes |
| Chat, Messages, and Responses | Yes | No | Yes |
| Caller-orchestrated MCP | Yes | No | Yes |
| Other inference APIs | Yes | No | No |
| `/api/v1/models` | Yes | Yes | No |
| Management APIs | Yes | Yes | No |

Hosted mode is a control plane. Its inference paths return a descriptive `404`
and, when configured, the data-plane URL to use instead. See [Modes](modes.md).

## Core inference APIs

Otari implements three completion surfaces:

- `POST /api/v1/chat/completions`, OpenAI Chat Completions
- `POST /api/v1/messages` and `/api/v1/messages/count_tokens`, Anthropic Messages
- `POST /api/v1/responses`, OpenAI Responses

Standalone mode also serves embeddings, images, audio, files, batches,
moderations, rerank, and search. Provider support differs by endpoint, so use
`GET /api/v1/models` and the OpenAPI document for the deployment you are calling.

## Search

`POST /api/v1/search` and `POST /api/v1/search/{search_tool_name}` run a configured
search tool directly. This is separate from `otari_web_search`, which lets a
model request searches during a completion. Both are described in
[Built-in tools](tools.md).

Search-tool management lives under `/api/v1/search-tools`. The generated OpenAPI
document describes the supported providers, filters, and management schemas.

## Routing policies

Routing-policy management lives under `/api/v1/routing/policies`; learned-routing
examples and status live under `/api/v1/routing/preferences` and `/api/v1/routing/status`.
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

## Caller-orchestrated MCP

Two stored-server endpoints let an application own its own MCP tool loop, as an
alternative to sending `mcp_servers` with a completion and letting Otari own it.
`GET /api/v1/mcp/servers/{mcp_server_id}/tools` returns the tool definitions a stored
MCP server exposes to the authenticated workspace, and `POST /api/v1/mcp/execute`
runs one exact caller-authorized call and returns the remote server's native MCP
result.

Otari executes a caller-authorized call; it does not verify a user approval and
does not claim to. The calling application is the authorization boundary and owns
any human approval, argument editing, cancellation, and action history. Otari
enforces authentication, stored-server access, the stored tool allowlist, URL
safety, and its own execution bounds.

`POST /api/v1/mcp/execute` must never be retried automatically, including by a
reverse proxy or service mesh: an `outcome_unknown` response means the tool may
already have run. See [MCP](mcp.md#caller-orchestrated-mcp) for the request
shapes, the error and execution-state contract, and the limits.

## Keeping generated clients current

API changes must regenerate both committed artifacts:

```bash
uv run python scripts/generate_openapi.py
make postman
make openapi-check
make postman-check
```
