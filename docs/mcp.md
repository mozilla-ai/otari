# MCP

Otari lets `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` use
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

## Messages streaming activity

A streaming `/v1/messages` client that includes
`betas: ["mcp-client-2025-04-04"]` receives live server-owned activity around
each gateway-run MCP call. Otari emits a `content_block_start` /
`content_block_stop` pair containing `mcp_tool_use` immediately before the call,
then another pair containing the matching `mcp_tool_result` immediately after it.
The blocks carry an opaque call id, tool and server names, parsed input, result
content, and an explicit `is_error` value. Without that beta declaration, Otari
still runs the MCP call but returns a stable stream without those activity blocks.

These blocks report execution; they do not transfer it. Otari still owns the MCP
call, feeds the result into the internal model loop, and returns one logical
Messages stream. The server URL, authorization token, and headers are never
included. If a client echoes the accumulated assistant message on a later turn,
Otari removes its activity pairs before forwarding the history upstream.

Chat Completions and Responses streams continue to hide gateway-run MCP activity
because those formats have no equivalent server-owned MCP vocabulary.

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
- `allowed_tools`: optional allow-list; only these tools are exposed from that server

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

### Configuring them (standalone)

The dashboard manages them under **Tools → MCP servers**, which acts on the
workspace the switcher has selected. Everything below is the same thing over the
API.

Manage a workspace's servers with the master key, under
`/v1/workspaces/{workspace_id}/mcp-servers`:

```bash
curl -X POST http://localhost:8000/v1/workspaces/$WORKSPACE_ID/mcp-servers \
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
- `OTARI_MCP_ALLOW_LOOPBACK=false` disables loopback; `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` relaxes the private-host restriction

For the hybrid platform contract behind `mcp_server_ids`, see
[Hybrid-mode protocol](hybrid-mode-protocol.md).
