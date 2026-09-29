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

### Retrying safely

A request the provider or the gateway refused (a 429, a 529, any other error) is
not billed: its budget hold is refunded, so a client can retry it as it is. So is
a stream the client disconnected from. The case that does bill twice is a
non-streaming request that succeeded while its response was lost on the way
back, through a dropped connection or a client timeout, because the retry calls
the provider again.

Send an `Idempotency-Key` header on a non-streaming Chat, Messages, or Responses
request to make that retry safe. The value is any unique string of 1 to 255
printable ASCII characters; a UUID is the usual choice. A retry with the same key
and the same body then gets the original response, with its original
`Otari-Request-ID` and `usage.cost_usd`, and an `Otari-Idempotent-Replayed: true`
header, without calling the provider or billing again. While the original is
still running the retry waits for it, and if the original fails the retry runs
in its place.

- A key belongs to the API key that sent it (or, for the master key, to the
  billed user), so two callers never see each other's responses.
- A retry has to be the same request: the same body, and the same
  `Otari-Code-Execution`, `Otari-Web-Search`, `Otari-Router` and
  `anthropic-beta` headers, since those change what the request does. The same
  key with a different body or different values for those headers is refused
  with 422, so send a new key for a new request. Key order and whitespace in the
  JSON body do not count as a difference.
- A retry that waits longer than `idempotency_wait_sec` for the original is
  answered 409 with `Retry-After`; retry again with the same key. A waiting
  retry checks on the original less often the longer it waits.
- The request holding a key renews its claim while it runs, so a retry does not
  run it a second time however long it takes. If the worker running it dies, its
  key frees up within `idempotency_lease_sec` (a minute by default).
- A response is kept for `idempotency_retention_sec` (a day by default),
  generated content included, and then deleted. It is stored encrypted with
  `OTARI_SECRET_KEY`, so a deployment without that key ignores the header, and
  a response no configured key can decrypt (after the key was rotated away)
  runs again. Responses larger than 8 MiB are not kept, so a retry of one runs
  again. Expired responses are still deleted after the header is turned off.
- Only a successful response is kept. On a retry the request is still
  authenticated and checked against the key's model access, and a user who has
  since been blocked is refused rather than given the stored response.

A retry runs again, and is billed again, whenever the original's response was
not stored or can no longer be read. The cases above are the ones a deployment
chooses: the response was larger than 8 MiB, its retention passed, the header was
turned off, or `OTARI_SECRET_KEY` was rotated away. Two more come from failures:
the gateway stops after the provider answers and before the response is stored,
or the database stays unreachable for about `idempotency_lease_sec` while the
original runs, so its claim lapses and a retry takes it over.
- Streaming requests ignore the header, and so does hybrid mode, which has no
  local database to keep the response in.

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
