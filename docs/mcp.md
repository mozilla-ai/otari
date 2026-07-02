# MCP

Otari lets `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` use
tools exposed by MCP servers.

Add MCP as a top-level request field, not a `tools` entry.

Use either or both of:

- `mcp_servers`: inline MCP server configs the gateway should connect to directly
- `mcp_server_ids`: workspace-scoped MCP server ids resolved through otari.ai (hybrid mode only)

In hybrid mode, Otari resolves `mcp_server_ids` first and appends the resulting
server configs to any inline `mcp_servers`.

When the model emits an MCP tool call, Otari:

- executes it
- appends the tool result to the conversation
- calls the model again

The loop stops when the model returns a normal assistant response or hits
`max_tool_iterations`.

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

## Workspace-scoped servers (hybrid only)

In hybrid mode, you can reference servers stored in otari.ai by id instead of
inlining their configs:

```json
{
  "model": "anthropic:claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Summarize the open PRs"}],
  "mcp_server_ids": ["11111111-1111-1111-1111-111111111111"]
}
```

Otari resolves those ids through the platform before the request runs. In
standalone mode, `mcp_server_ids` returns `400`.

## Limits and safety

- `mcp_servers` and `mcp_server_ids` cannot be combined with `otari_code_execution` or `otari_web_search` in the same request yet
- `max_tool_iterations` optionally caps the loop; default is `10`, max is `25`
- MCP URLs are validated to reduce SSRF risk; by default, private and reserved addresses are blocked, loopback is allowed, and `http://` is rejected when `authorization_token` is present
- `OTARI_MCP_ALLOW_LOOPBACK=false` disables loopback; `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` relaxes the private-host restriction

For the hybrid platform contract behind `mcp_server_ids`, see
[Hybrid-mode protocol](hybrid-mode-protocol.md).
