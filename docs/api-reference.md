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

Otari accepts a credential in any of these forms, whatever the mode: a local API
key or the master key in standalone and hosted mode, an otari.ai user token in
hybrid mode:

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

In hybrid mode, the generation APIs and the `/api/v1/mcp` and `/api/v1/hooks`
endpoints accept an otari.ai user token in the same header forms. Local API
keys and management APIs are not used.

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

### Request ID and inline cost

Every Chat, Messages, and Responses response carries an `Otari-Request-ID`
header, streaming or not. In hybrid mode it is the platform's id for the
request; a standalone gateway mints its own.

A priced response also carries its cost on the usage object it already returns,
as `usage.cost_usd` (a six-decimal USD string) and `usage.pricing_source`. On a
stream the fields ride the terminal usage event: the last usage chunk for Chat
Completions, `message_delta` for Messages, and `response.completed` for
Responses. The two fields always appear together, and an unpriced or
unreported request carries neither.

In standalone mode the amount is the one the gateway wrote to its own usage
record, including any gateway-run tool charges, and `pricing_source` names the
rate that priced the model: `organization` (an organization's override),
`deployment` (a rate stored on this gateway), or `defaults` (the bundled
genai-prices dataset). Hybrid mode attaches the platform's settlement instead;
see [Hybrid mode protocol](hybrid-mode-protocol.md#inline-response-fields).

### Cost of a failed or interrupted request

A stream that fails mid-response ends in an error event, and one the client
disconnects from ends with nothing, so neither delivers `usage.cost_usd`. The
provider may still have charged for the tokens it reported before the stream
ended (an Anthropic stream reports its input tokens in `message_start`), and a
standalone gateway records and bills those tokens rather than treating the
request as free.

To recover that amount, look the request up by its `Otari-Request-ID`:

```
GET /api/v1/usage/requests/{request_id}
```

```json
{
  "request_id": "5f0c…",
  "status": "error",
  "cost_usd": "0.012400",
  "prompt_tokens": 4100,
  "completion_tokens": 0,
  "total_tokens": 4100,
  "row_count": 1
}
```

The response sums every usage row the request wrote: a request routed through a
policy writes one per attempt, all sharing that id as their `request_group_id`,
so the total covers the attempts it fell over from as well as the one that
served. `cost_usd` uses the inline format and is `null` when nothing was priced.
An API key sees only its own requests and the master key sees any. The endpoint
answers 404 until the request has settled, since usage rows are written in the
background, and for an id that is unknown or belongs to another key.

This lookup is standalone only. In hybrid mode the platform owns settlement, and
a failed stream reports no usage to it; see
[Hybrid mode protocol](hybrid-mode-protocol.md).

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
