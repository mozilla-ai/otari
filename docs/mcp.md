# MCP

Otari lets `/api/v1/chat/completions`, `/api/v1/messages`, and `/api/v1/responses` use
tools exposed by MCP servers.

Add MCP as a top-level request field, not a `tools` entry.

Use either or both of:

- `mcp_servers`: inline MCP server configs the gateway should connect to directly
- `mcp_server_ids`: ids of MCP servers your workspace has configured

Otari resolves `mcp_server_ids` first and appends the resulting server configs
to any inline `mcp_servers`. Where it resolves them depends on the mode:
standalone reads the servers the workspace configured on this gateway, hybrid
resolves them through otari.ai.

When the model emits an MCP tool call, Otari:

- executes it
- appends the tool result to the conversation
- calls the model again

The loop stops when the model returns a normal assistant response or hits
`max_tool_iterations`.

## Caller-orchestrated MCP

Otari supports two MCP integration modes, and they are alternatives rather than
layers.

**Managed MCP** is the loop above: you send `mcp_servers` or `mcp_server_ids`
with a completion, and Otari owns the model/tool loop.

**Caller-orchestrated MCP** hands the loop back to your application. Otari
provides two stored-server endpoints, and your application decides whether a
proposed call may run:

1. `GET /api/v1/mcp/servers/{mcp_server_id}/tools` returns the tool definitions a
   stored MCP server exposes to the authenticated workspace.
2. Your application exposes those tools to its model or workflow, and applies
   its own authorization policy to whatever gets proposed. That may be a user
   approval prompt, an administrator rule, or trusted read-only
   auto-authorization.
3. `POST /api/v1/mcp/execute` runs one exact authorized call and returns the remote
   server's native MCP result.

**Otari executes a caller-authorized call. It does not verify a user approval
and does not claim to.** The calling application is the authorization boundary
and owns any human approval, argument editing, cancellation, and action history.
What Otari enforces independently is authentication, that the stored server is
one the authenticated workspace may reach, the stored tool allowlist, URL
safety, and its own execution bounds.

Between the two requests Otari holds nothing open: no model stream, no MCP
session, no database session, no worker-local state. That is why your
application can pause for a person for as long as it needs to.

Both endpoints refer to a server by the id it was registered under. There is no
inline server field: registering a remote MCP server through the control plane
once, rather than transmitting a URL and a credential per request, is what keeps
authorization, credential storage, revocation, and allowlist policy on Otari's
side of the boundary. Local `stdio` servers stay outside this: a hosted Otari
deployment cannot reach a machine-local process.

### Discovery

```http
GET /api/v1/mcp/servers/2c948a61-dc96-4cd8-96bb-8e1434bf424e/tools
Authorization: Bearer <token>
```

```json
{
  "server_id": "2c948a61-dc96-4cd8-96bb-8e1434bf424e",
  "server_revision": "9f2c41ae7b0d4e5c8a13d6f0b27e5c91",
  "tools": [
    {
      "name": "create_issue",
      "description": "Create an issue",
      "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
      "annotations": {"readOnlyHint": false}
    }
  ],
  "warnings": []
}
```

Call this once per server when you prepare a run, and reuse the answer for every
tool from that server for the length of the run. There is no cross-run cache, so
a later run rediscovers.

Where the stored server has an allowlist, `tools` is its intersection with the
live catalog, so a tool the server added later is not exposed just because it
was listed. Where it has none, the whole live catalog is returned, the same
meaning the column already carries in the managed loop. An explicit empty
allowlist is a deny-all: `tools` is empty and Otari opens no connection.

The response never contains the server URL, its credential, or an allowlist entry
the live catalog did not return.

`annotations` is the server's own metadata, passed through as untrusted data.
Otari never turns `readOnlyHint` into an authorization decision, and never infers
safety from a tool's name or description. Your risk policy is yours.

`tools` is either the complete authorized catalog or absent: a pagination
failure or a discovery ceiling returns `502 mcp_discovery_limit_exceeded` with no
tools at all, because a short catalog would be read as the complete input to
your authorization policy. The one exception is `warnings`, which names a tool
whose descriptor Otari could not carry, for example a schema that is too large or
one with a `$ref` pointing outside itself:

```json
{"tool_name": "create_issue", "code": "mcp_tool_schema_unsupported"}
```

A tool name outside the 1–256-character execution limit is omitted with
`mcp_tool_name_unsupported`, so every tool returned by discovery can be sent
back to the execution endpoint.

Otari does not validate an `inputSchema` against a JSON Schema dialect, and never
truncates, rewrites, or drops a keyword. It bounds the size, depth and property
count it will carry, and it refuses a schema that would have to be fetched over
the network to be understood at all. An unknown dialect or vendor keyword survives
the round trip unchanged.

### Execution

```http
POST /api/v1/mcp/execute
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "mcp_server_id": "2c948a61-dc96-4cd8-96bb-8e1434bf424e",
  "tool_name": "create_issue",
  "arguments": {"title": "Approved title", "body": "Approved body"},
  "server_revision": "9f2c41ae7b0d4e5c8a13d6f0b27e5c91",
  "client_execution_id": "6e51b3bc-6f48-4c68-a61e-0786bc80cd67"
}
```

Otari resolves the stored server, applies its allowlist to `tool_name`, compares
`server_revision`, validates the resolved URL, and then calls that one tool
directly. It never calls `tools/list` here: MCP permits calling a known tool, and
what authorizes the call is the stored allowlist and the authenticated workspace,
not a live catalog.

The successful body is the native MCP result:

```json
{
  "content": [{"type": "text", "text": "Created issue #42"}],
  "structuredContent": {"issue_number": 42},
  "isError": false
}
```

`isError: true` is still an HTTP 200. The server handled the call and classified
the outcome itself, so it is a definitive result rather than a transport failure,
and it is not a retry signal.

`server_revision` is the value discovery returned. Persist it with the proposed
call and send it back. It is an opaque revision of the stored server's URL,
credential, enabled state and allowlist, so a configuration change between your
authorization decision and this request is refused with
`409 mcp_server_changed` rather than executed against something else. It is not
an approval credential.

It detects Otari-side and platform-side configuration changes only. A remote
server that changes its own catalog, or a tool's behavior, behind an unchanged
URL will not move it; catching that would need another live `tools/list` at
execution time, which this version does not do. The revision binds the request
only to the stored server configuration returned by discovery. The calling
application is responsible for binding the server id,
revision, remote tool name, and final arguments to its authorization or approval
record. Neither mechanism binds the call to an immutable remote implementation.

`client_execution_id` is a canonical UUID you generate, for correlation across
your service and Otari. It is not proof of approval and **not an idempotency
key.**

### Never retry an execution automatically

Repeating a request with the same `client_execution_id` may execute the tool
again. Otari implements no deduplication, because a worker-local one would not
survive a second process or a restart.

Send exactly one HTTP attempt for `POST /api/v1/mcp/execute`, and make sure nothing
on the path adds attempts of its own: your SDK, your own retry policy, and the
reverse proxies, service meshes and ingress controllers in front of Otari, on
connection resets and 5xx responses alike.

`execution_state` on every error tells you what a retry would mean:

- `not_started`: Otari knows the call was not dispatched. Retrying is as safe as
  your own execution claim allows.
- `outcome_unknown`: dispatch began and no definitive result came back, so the
  tool may have mutated external state. Surface this as indeterminate. Do not
  turn it into an ordinary failure, and do not fall back to executing the call
  yourself.

The boundary is deliberately conservative. Once the transport has begun writing
the `tools/call` request, every timeout, cancellation, protocol error, parse
error, size-limit breach and dropped connection is `outcome_unknown`, even where
the client cannot prove the server received the whole request.

A caller-side network failure that produces no typed Otari response at all is
always indeterminate.

### Errors

Both endpoints share one error body:

```json
{
  "detail": "MCP execution outcome is unknown",
  "code": "mcp_outcome_unknown",
  "execution_state": "outcome_unknown",
  "request_id": "req_..."
}
```

`detail` is a fixed safe message per category; `code` and `execution_state` are
stable enums. `request_id` is an opaque Otari request id, present in every error
body and returned in `X-Otari-Request-ID` on success and failure alike, which is
why a successful body stays the plain MCP result. No error carries a server URL,
a credential, a header, your arguments, a result, a platform detail, or raw
exception text.

| Condition | Status | Code | Execution state |
|---|---:|---|---|
| Invalid request body | 422 | `invalid_request` | `not_started` |
| Authentication failure | 401 | `authentication_failed` | `not_started` |
| Payment required or insufficient funds | 402 | `payment_required` | `not_started` |
| Platform authorization refused | 403 | `forbidden` | `not_started` |
| Authenticated request rate exceeded | 429 | `rate_limit_exceeded` | `not_started` |
| Server inaccessible to caller, or disabled | 404 | `mcp_server_not_found` | `not_started` |
| Stored server revision changed | 409 | `mcp_server_changed` | `not_started` |
| Tool absent from stored allowlist | 403 | `mcp_tool_not_allowed` | `not_started` |
| Unsafe resolved URL | 400 | `unsafe_mcp_url` | `not_started` |
| Stored-server resolution unavailable | 502 | `mcp_resolution_failed` | `not_started` |
| Stored credential unavailable | 500 | `mcp_credentials_unavailable` | `not_started` |
| Discovery bound or pagination failure | 502 | `mcp_discovery_limit_exceeded` | `not_started` |
| Discovery admission deadline exceeded | 503 | `mcp_discovery_capacity_unavailable` | `not_started` |
| Execution admission deadline exceeded | 503 | `mcp_capacity_unavailable` | `not_started` |
| Local service temporarily unavailable | 503 | `service_unavailable` | `not_started` |
| Connection or initialization failure | 502 | `mcp_connection_failed` | `not_started` |
| Tool-call deadline exceeded after dispatch | 504 | `mcp_outcome_unknown` | `outcome_unknown` |
| Transport, cancellation, protocol or parse failure after dispatch | 502 | `mcp_outcome_unknown` | `outcome_unknown` |
| Result exceeds the response bound | 502 | `mcp_result_too_large` | `outcome_unknown` |

Only the two capacity codes and `rate_limit_exceeded` carry `Retry-After`. No
other failure advertises an automatic retry.

### Limits

Gateway-owned ceilings, not caller-controlled request fields:

| Scope | Limit | Value |
|---|---|---|
| Request | `tool_name` length | 1 to 256 characters |
| Request | `arguments` encoded size, nesting depth | 256 KiB, depth 32 |
| Discovery | `tools/list` pages followed | 20 |
| Discovery | live descriptors examined | 1,000 |
| Discovery | returned tools | 200 |
| Discovery | per description / schema / annotations | 4 KiB / 64 KiB at depth 32 / 16 KiB |
| Discovery | serialized response | 1 MiB |
| Discovery | total request, concurrency, admission wait | 15 s, 4 per process, 5 s |
| Both | connection plus initialization | 5 s |
| Execution | tool call, from dispatch | 30 s |
| Execution | serialized result | 1 MiB |
| Execution | concurrency, admission wait, total request | 8 per process, 5 s, 45 s |

Discovery and execution hold separate concurrency ceilings, so slow or hostile
discovery cannot starve calls an application has already authorized.

The 1 MiB transport ceiling is enforced while each response stream is read, so
chunked JSON and SSE responses cannot bypass it by omitting `Content-Length`.
Otari requests identity encoding and refuses compressed responses before
reading their bodies, preventing a small encoded response from expanding past
the ceiling during decompression. The decoded MCP result is measured again
before it is returned.

Redirects are disabled and never forwarded to the caller. A redirect during
connection or initialization becomes `502 mcp_connection_failed`; after tool
dispatch it becomes `502 mcp_outcome_unknown`. No credential or call data is
sent to the redirect destination.

### Availability

Both endpoints are data-plane surfaces. Standalone resolves stored servers from
the authenticated API key's workspace; a master-key request holds no workspace of
its own and reaches no stored server, so it gets the `404`. Hybrid resolves them
through otari.ai using the authenticated user token. Hosted mode is a control
plane and serves neither: both paths return the standard hosted-mode `404`
naming the configured data-plane URL.

## Messages streaming activity

For streaming `/api/v1/messages` requests with the standard Anthropic header
`anthropic-beta: mcp-client-2025-11-20`, Otari emits server-owned activity
around each gateway-run MCP call. The body form
`betas: ["mcp-client-2025-11-20"]` is also accepted. An `mcp_tool_use` block
starts immediately before execution with an opaque call id, tool and server
names, and parsed input. A matching
`mcp_tool_result` block follows with the result content and `is_error` value.
Each block uses a `content_block_start` / `content_block_stop` pair. Without an
MCP client beta, Otari still runs the call but omits these activity blocks.

The blocks report execution without transferring it: Otari runs the call, feeds
the result back to the model, and returns one logical Messages stream. It strips
the blocks from history echoed on later turns and never includes the server URL,
authorization token, or headers. Chat Completions and Responses streams continue
to hide this activity because they have no equivalent server-owned MCP vocabulary.

## Inline MCP servers

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "List open issues in mozilla-ai/otari"}],
  "mcp_servers": [
    {
      "name": "github",
      "url": "https://mcp.example.com/github",
      "authorization_token": "ghp_...",
      "purpose_hint": "Use for repository and issue lookups",
      "allowed_tools": ["list_issues", "get_issue"]
    }
  ]
}
```

- `name`: label for the server
- `url`: streamable HTTP MCP endpoint, reachable from the gateway
- `authorization_token`: optional bearer token; when set, the `url` must use `https://`
- `purpose_hint`: optional hint Otari prepends to the system message to help the model choose the tool
- `allowed_tools`: optional allow-list with three states: omit it or send `null` to expose every listed tool, send `[]` to expose none, or send a non-empty list to expose only those named tools

## Workspace-scoped servers

Reference servers your workspace has configured by id, instead of inlining
their configs:

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Summarize the open PRs"}],
  "mcp_server_ids": ["11111111-1111-1111-1111-111111111111"]
}
```

The workspace is the one your API key belongs to; it is never read from a
header. An id that names no server in that workspace returns `404`, and a
server that is configured but disabled is skipped rather than refusing the
request.

The same stored servers are what
[caller-orchestrated MCP](#caller-orchestrated-mcp) refers to by id, so a server
registered once serves both integration modes. The two read a disabled server
differently, and deliberately: the loop above resolves a list and skips one
entry, while an endpoint asked about exactly that server answers `404` rather
than returning nothing.

### Configuring them (standalone)

The dashboard manages them under **Tools → MCP servers**, which acts on the
workspace the switcher has selected. Everything below is the same thing over the
API.

Manage a workspace's servers with the master key, under
`/api/v1/workspaces/{workspace_id}/mcp-servers`:

```bash
curl -X POST http://localhost:8000/api/v1/workspaces/$WORKSPACE_ID/mcp-servers \
  -H "Otari-Key: $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "name": "github",
        "url": "https://mcp.example.com/github",
        "authorization_token": "ghp_...",
        "purpose_hint": "Use for repository and issue lookups",
        "allowed_tools": ["list_issues", "get_issue"]
      }'
```

`GET` lists them, `PATCH /{server_id}` updates one, and `DELETE /{server_id}`
removes it. The list is readable by any member who can see the workspace;
creating, updating, or deleting a server needs an organization owner/admin or
an owner/admin of that workspace.

- The `authorization_token` is encrypted at rest with `OTARI_SECRET_KEY` and is
  never returned. Responses carry `has_token` instead. On a `PATCH`, omit the
  field to leave the stored token alone, send `""` to clear it, or send a value
  to rotate it.
- `name` is unique within a workspace; a duplicate is refused with `409`.
- `enabled: false` keeps the row and its token but takes the server out of
  every request that names it.
- The URL is checked for SSRF safety when it is stored as well as when a
  request uses it, and must be `https://` when a token is set. A stored URL that
  passes the first check and fails the second, because the host it resolves to
  moved, refuses the request with a `500`: the endpoint is a workspace setting
  the caller cannot see or fix, so the reason goes to the log rather than into
  the response. A URL sent in the request body is yours to fix and still fails
  with a `400` naming what was wrong with it.
- A workspace may configure up to 50 servers.

In hybrid mode these routes are not mounted: the servers live in otari.ai and
are managed there.

## Limits and safety

- `mcp_servers` and `mcp_server_ids` cannot be combined with `otari_code_execution` or `otari_web_search` in the same request yet
- `max_tool_iterations` optionally caps the loop; default is `10`, max is `25`
- `mcp_server_ids` accepts at most 50 ids, which is also the most servers a workspace can have; a repeated id resolves once
- Server names must be unique across everything one request uses, because Otari routes each tool call to its server by name. Two `mcp_servers` entries sharing a name are refused with a `400` naming it, the way a bad request-body URL is. An `mcp_servers` name colliding with one of your workspace's stored servers is a `400` too, but a fixed one that repeats neither name, since a stored server is not yours to read. Two stored servers sharing a name is a `500` with the names in the log, on the same grounds as a stored URL that fails its safety check
- MCP URLs are validated to reduce SSRF risk; by default, private and reserved addresses are blocked, loopback is allowed, and `http://` is rejected when `authorization_token` is present
- The managed MCP loop follows redirects only when the scheme, host, and port stay the same, or when a default-port `http://` URL upgrades to `https://` on the same host. Caller-orchestrated discovery and execution do not follow redirects. Configure the final URL when a managed-loop redirect changes host or port
- `OTARI_MCP_ALLOW_LOOPBACK=false` disables loopback; `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` relaxes the private-host restriction

For the hybrid platform contract behind `mcp_server_ids`, see
[Hybrid-mode protocol](hybrid-mode-protocol.md).
