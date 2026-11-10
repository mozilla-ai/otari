# MCP

Otari lets `/api/v1/chat/completions`, `/api/v1/messages`, and `/api/v1/responses` use
tools exposed by MCP servers.

Add MCP as a top-level request field, not a `tools` entry.

Use either or both of:

- `mcp_servers`: inline MCP server configs the gateway should connect to directly
- `mcp_server_ids`: ids of MCP servers your workspace has configured

Otari resolves `mcp_server_ids` first and appends the resulting server configs
to any inline `mcp_servers`, reading the servers the workspace configured on
this gateway.

When the model emits an MCP tool call, Otari:

- executes it
- appends the tool result to the conversation
- calls the model again

The loop stops when the model returns a normal assistant response. Its last
allowed round (the one `max_tool_iterations` names) offers the model none of
Otari's tools and tells it to answer from the results it already has. Your own
tools stay on offer in that round. The request fails only if the model calls
one of Otari's tools anyway.

That loop is managed MCP. To run the tool loop in your own application instead,
see [Caller-orchestrated MCP](#caller-orchestrated-mcp).

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

Which stored servers a key reaches, and which of their tools, is each server's
[access list](#server-access). A server that denies the key answers `404` exactly
as an unknown id does, and the tools it may not run are never offered to the
model.

A stored server sends either one token for the whole workspace or each user's
own, as its [credential mode](#credential-mode) says. A request that reaches a
server its user has not signed in to is not refused: the model is given a
[sign-in link](#signing-in-from-the-conversation) to show instead.

A server can also be [attached to every request](#attached-to-every-request), so
a client that never sends `mcp_server_ids`, such as a coding agent, gets it too.

[Caller-orchestrated MCP](#caller-orchestrated-mcp) refers to the same stored
servers by id. It treats a disabled server differently: its endpoints answer
`404` for it, where the managed loop skips it.

### Configuring them

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
- The URL is checked for SSRF safety when it is stored and again when a
  request uses it, and must be `https://` when a token is set. A stored URL
  that fails the second check (because the host it resolves to moved) refuses
  the request with a `500`, with the reason in the log only. A URL sent in the
  request body fails with a `400` naming what was wrong with it.
- A workspace may configure up to 50 servers.
- `everyone_tools` (default `["*"]`) is the server's default policy, and
  `read_tools` (default `[]`) the tools labeled read; see
  [Server access](#server-access).
- `GET /{server_id}` reads one server, for any member who can see the workspace.
- `credential_mode` (default `shared`) says whose credential a call carries;
  see [Credential mode](#credential-mode).
- `attach_by_default` (default `false`) adds the server to every model request
  from the workspace's keys; see [Attached to every request](#attached-to-every-request).

#### Attached to every request

Claude Code, Pi and most other clients never send `mcp_server_ids`. Set
`attach_by_default: true` on a server and Otari adds it to every chat
completions, Messages and Responses request made with one of the workspace's API
keys, as if the request had named it, and runs its tools in its own loop.

```bash
curl -X PATCH http://localhost:8000/api/v1/workspaces/$WORKSPACE_ID/mcp-servers/$SERVER_ID \
  -H "Otari-Key: $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"attach_by_default": true}'
```

In the dashboard it is **Attach to every request**, on the server's page, on its
row under **Tools → MCP servers**, and in the form that adds one.

- **The access list still decides.** An attached server joins only the requests
  whose key it grants something, with the tools that key holds and under the
  server's `allowed_tools`. A key it denies is left without it, silently. To keep
  one key off an attached server, give that key a rule with no tools.
- **A key with no user gets no per-user server.** An attached per-user server is
  left out for a key that belongs to nobody, rather than offering a sign-in no
  one can complete.
- **It adds to what the request names.** The servers in `mcp_server_ids` come
  first, then the attached ones it did not name. A named server keeps its strict
  answers (`404` for one the key cannot reach); an attached one never refuses a
  request.
- **The client's own tools stay the client's.** A request that declares its own
  tools, as a coding agent does with dozens, keeps them all. A call to one of
  them goes back to the client as before, and where a server's tool has the same
  name as one of the client's, the client's wins and the server's is not offered.
- **A server that is down is left out.** An attached server that cannot be
  reached, does not answer within 10 seconds, or whose URL fails its safety check
  is dropped from that request, with the reason in the log, and the request goes
  on without it. A server the request named still fails it.
- **Streaming works the same way.** The loop runs gateway tool calls between
  rounds and streams only what the model writes for the user.
- **Loop limits and the bill.** The loop stops after `max_tool_iterations` rounds
  (10 unless the request sets it, at most 25), the same as for a named server.
  Each extra round is an ordinary model call billed to the key and held against
  its budget, so a request that uses an attached server costs what it would have
  cost had the client named it.
- **What it costs on every request.** The server's tool definitions are sent with
  every request from the workspace, so they cost input tokens even when the model
  uses none of them, and Otari connects to the server to list its tools each
  time. The list is not cached, because each request needs its own live session
  to call a tool and a per-user server's list is per user; the definitions are
  sent in the same order every time, so a provider's prompt cache absorbs most of
  the token cost. Use `allowed_tools` to send only the tools people need.
- **Only API keys.** The Playground names the servers it uses, and the master key
  belongs to no workspace, so neither gets attached servers.

#### Credential mode

A server in `shared` mode sends its `authorization_token` for every caller, so
everyone acts as the account that token belongs to. A server in `per_user` mode
holds no shared token. Each user signs in to it with their own account, and a
request carries the token of the user its API key belongs to.

```bash
curl -X POST http://localhost:8000/api/v1/workspaces/$WORKSPACE_ID/mcp-servers \
  -H "Otari-Key: $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "drive", "url": "https://mcp.example.com/drive", "credential_mode": "per_user"}'
```

- **Who the user is.** The user an API key belongs to (its `user_id`). In the
  Playground it is the signed-in person, who is the same user their own keys
  belong to. A sign-in made in the dashboard connects the signed-in person's
  own user, so a key can use it only when its `user_id` is that person's:
  create the key through the dashboard's key page for that person, not under
  another user id. A key created without a user belongs to the shared
  `default` user, which is nobody's account: it cannot use a per-user server
  and is told it does not exist.
- **How a user signs in.** Through the
  [MCP authorization flow](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization):
  Otari reads the server's protected-resource metadata and its authorization
  server's metadata, registers itself as a client, and runs the authorization
  code flow with S256 PKCE. Where the authorization server does not register
  clients on its own, set `oauth_client_id` (and `oauth_client_secret` if it
  has one) on the server, with
  `<public_base_url>/api/v1/mcp-connections/callback` as the client's redirect
  URI. [Connect your account to an MCP server](connect-your-account-to-an-mcp-server.md)
  walks through it in the dashboard.
- **What it needs.** `public_base_url`, because the redirect URI is derived
  from it, `OTARI_SECRET_KEY`, because the tokens are encrypted with it, and an
  `https://` server URL. Every URL the server and its authorization server
  publish passes the same SSRF check the server's own URL does.
- **A connection belongs to one user on one server.** A server is a workspace's,
  so the same person signs in once per workspace, and one user's connection is
  never used for another's request.
- **Tokens are renewed for you.** An access token close to its expiry is
  refreshed before the request uses it. When the authorization server ends the
  grant, the connection is removed and the user is asked to sign in again. Otari
  learns this only when it next renews: a grant revoked at the provider early,
  or one whose token states no expiry, still shows as connected until a call
  fails, and the user then reconnects from the server's page.
- **What ends a connection.** The user disconnecting, a manager removing it,
  deleting the server, deleting the user, the user leaving the workspace, and a
  change to the server's URL, its OAuth client or its credential mode.
- **The access list still comes first.** A key the ACL denies is told the
  server does not exist, whether or not its user has signed in.

A per-user server's connection check runs with the connection of the person
asking, so `health` stays `null` until a manager who has signed in checks it.

The routes, under `/api/v1/workspaces/{workspace_id}`, take a dashboard session
or the master key:

| Route | Does | Who |
|---|---|---|
| `POST /mcp-connections/{server_id}` | Starts your sign-in and returns `authorization_url`. Takes an optional `return_to`, a dashboard path to land on afterwards | Any member who can see the workspace |
| `GET /mcp-connections` | Lists your own connections in the workspace | Any member who can see the workspace |
| `DELETE /mcp-connections/{server_id}` | Ends your connection, and asks the authorization server to revoke its tokens | Any member who can see the workspace |
| `GET /mcp-servers/{server_id}/connections` | Lists who is connected, a page at a time (`skip`, `limit`), with the total in `count` | Workspace or organization owners and admins |
| `DELETE /mcp-servers/{server_id}/connections/{user_id}` | Ends one user's connection | Workspace or organization owners and admins |

A sign-in started here is single-use, expires after ten minutes, and finishes
only for the person who started it. The authorization server redirects the
browser to `<public_base_url>/api/v1/mcp-connections/callback`, which passes
`state` and `code` on to the dashboard and spends nothing. A client other than
the dashboard finishes the sign-in by posting both, as JSON, to
`POST /api/v1/mcp-connections/complete` as the same person. No route returns a
token.

#### Signing in from the conversation

A request whose tool loop reaches a per-user server its user has not signed in
to is not refused. The model gets one stand-in tool for that server,
`sign_in_to_<name>`, whose description says the server exists and needs the
user's own sign-in. When the user asks for something that server does, the model
calls it, and the result carries a link to show:
`<public_base_url>/api/v1/mcp-connections/sign-in?ticket=...`. The person opens
it, signs in with their own account, and lands on a plain Otari page that tells
them to go back to the conversation. The next request uses the new connection.

- **The link is minted for the key's own user, and only when the model reaches
  for the server.** Making one stores nothing: it carries the server and the user
  sealed with `OTARI_SECRET_KEY` and works for 30 minutes. Opening it starts a
  sign-in that finishes on its state alone, since the person may have no
  dashboard session; opening it again replaces the sign-in it started before.
- **Treat a link like a short-lived invitation.** Whoever opens it within its 30
  minutes connects the account they sign in with to that user. It never carries
  a token or the sign-in's state, and the log masks it.
- **Only Otari makes the link.** It always starts with the deployment's own
  `public_base_url`, and the stand-in tool tells the model to show no sign-in
  link that came from anywhere else. A server's name reaches the model with
  anything but letters, digits, spaces, `.`, `_` and `-` removed.
- **A key with no user gets no link.** A server it names explicitly gets the
  stand-in, which tells the model the key belongs to nobody and to use a personal
  key. An attached one is simply left out.
- **A token that cannot be renewed right now** gets the same stand-in, telling
  the user to try again shortly and offering a fresh sign-in if it keeps failing.
- **Needs** `public_base_url` and `OTARI_SECRET_KEY`, as any per-user server does.
  Without them the stand-in says the deployment cannot make sign-in links.

A stand-in tool is used rather than a note in the system message because a tool
reaches the model the same way in all three API shapes, sits beside the many
tools a coding agent already sends without rewriting its system prompt, and lets
the link be made only when it is wanted.

#### Connection checks

Creating a server connects to it once, lists its tools, and returns the outcome
as `health`. So does a `PATCH` that changes the URL, the token, or
`allowed_tools`; other edits leave `health` as it was.
`POST /{server_id}/check` runs the check again on demand, with the same
permissions as an update.

A server that fails its check is still stored. `health.status` is `failing` and
`health.error` names the cause: `unreachable`, `tls_failed`, `timeout`,
`redirected`, `unauthorized` (401), `forbidden` (403), `not_found` (404 or
405), `server_error` (5xx), `http_error` (another status), `not_mcp` (the URL
answered, but not as a streamable HTTP MCP server), `limits_exceeded`,
`unsafe_url`, `credentials_unavailable` (the stored token will not decrypt), or
`connection_failed`. `health.http_status` carries the status where the server
refused with one. A passing check reports `tools`, every tool name the server
offers whatever the allow-list says (up to 200), `tool_count`, the number of
those the allow-list admits, and `missing_allowed_tools`, the allow-listed names
the server does not offer.

`POST /check` (on the collection, with no server id) checks connection details
without storing anything, which is what the dashboard's **Test connection**
button calls before a save. It takes `url`, `authorization_token` and
`allowed_tools`, and answers with what `health` would say. Pass `server_id` and
omit `authorization_token` to test with that server's stored token; `""` tests
with none. An unsafe URL is refused with the same `400` a create gets.

`health` is `null` until a check has run. It is also `null` after a connection
change when every check slot on the gateway was busy; `POST /{server_id}/check`
answers `429` in that case and records nothing. The value describes the server
as of `health.checked_at`; a request still checks the URL again, and nothing is
skipped because a check once passed.

### Server access

Each stored server has an access list (ACL) that decides which keys reach it and
which of its tools they may run. The dashboard steps are in
[Control access to an MCP server](control-access-to-an-mcp-server.md).

An ACL has two parts:

- **Rules**, each for one key or one user. A user's rule covers every key that
  user owns, including keys created later. A rule's `tools` are the tool patterns
  it allows; `[]` denies. The API calls a rule a grant.
- **The default policy**, the server's `everyone_tools`, for every key no rule
  names. `["*"]`, the default, allows every tool; `[]` denies.

A key's rule wins, then its user's rule, then the default policy. There is no
order to manage: "deny by default, allow these keys" and "allow by default,
except this key" are each a policy plus rules.

A tool pattern is one of:

| Pattern | Matches |
|---|---|
| `create_issue` | that tool |
| `*` | every tool |
| `get_*` | every tool whose name starts with `get_` |
| `@read` | the server's read tools |

Any other use of `*` or `@` is refused.

The master key is not bound by any ACL. A dashboard session, which is what the
Playground runs as, has no key, so it is bound as its user: the user's rule, else
the default policy. The Playground offers only the servers that user reaches.

#### Read tools

`read_tools` on a server lists the tools a manager labeled read, as exact names,
set with `PATCH` on the server. `@read` stands for exactly that list, so a rule of
`["@read"]` keeps meaning "read only" as the labels change. A tool nobody labeled
counts as write, and a new server has none labeled, so `@read` allows nothing
until someone labels some.

Otari never takes the labels from the server. A tool may declare MCP's
`readOnlyHint`, and a connection check reports those tools as
`health.read_only_hinted_tools`, which the dashboard offers as suggestions. A
server can mark any tool read-only, so a hint counts only once a manager saves it.

#### What a caller sees

- **A denied key cannot tell the server exists.** Every route answers it as it
  would an unknown id: `GET /api/v1/mcp/servers` leaves the server out, a
  completion naming it in `mcp_server_ids` gets a `404`, and discovery and
  execution answer `404 mcp_server_not_found`. To see why a key was turned
  away, use `GET .../access/check` (below).
- **A key the ACL admits, on a per-user server its user has not signed in to,
  is told so, with a link to sign in.** `GET /api/v1/mcp/servers` lists the
  server with `sign_in_required: true` and a `sign_in_url`, a completion that
  reaches it hands the model that link through a stand-in tool, and discovery
  and execution answer `403 mcp_sign_in_required` with the same `sign_in_url`.
  See [Signing in from the conversation](#signing-in-from-the-conversation).
- **A tool outside the key's rule is never offered to the model.** On
  `POST /api/v1/mcp/execute` it is refused with `403 mcp_tool_not_allowed`.
- **In a completion, the model is told what it may not use.** The server's line in
  the system message names the tools the caller holds and the ones it does not,
  so the model can say the caller lacks access instead of trying. A name that is
  not plain letters, digits, `_`, `.` and `-` is left out, since the server wrote
  it. If the model calls a withheld tool anyway, the call never reaches the server
  or the client: the model gets `[tool error] mcp_tool_not_allowed: ...` and the
  trace records the call as blocked, with the rule that refused it. A tool the
  request declares itself under the same name stays the client's.
- **The server's `allowed_tools` comes first.** A tool it leaves out is
  unreachable whatever a rule says, and a disabled server is unreachable to
  everyone.

Rules name the server by id, so renaming a server keeps them. Deleting the
server, the key or the user deletes them.

#### Servers named in a request

`mcp_servers` in a completion request names a server by URL, which no ACL covers.
A workspace refuses such a request with a `400` unless
`request_mcp_servers_allowed` is on, so its ACLs cannot be stepped around. A new
workspace starts with it off; one that existed before ACLs, including the default
workspace an install migrates into, keeps it on. Change it with
`PATCH /api/v1/workspaces/{workspace_id}`.

The switch also covers asking a provider to dial a server itself: a Responses tool
of `type: "mcp"`, or a Gemini tool with a non-empty `mcp_servers`
(`mcpServers`). The master key may always name servers; a key or a dashboard
session follows the workspace's switch.

#### Managing the ACL

These routes live under `/api/v1/workspaces/{workspace_id}/mcp-servers/{server_id}`
and need an owner or admin of the organization or the workspace:

| Route | Does |
|---|---|
| `PATCH /` with `everyone_tools` | Change the default policy. |
| `GET /access` | List the default policy and every rule, each naming its key (with name, owner and whether it is active) or user. |
| `POST /access/grants` | Give one rule to several `api_key_ids` and `user_ids`. A key or user that has a rule gets this one instead. |
| `PATCH /access/grants/{grant_id}` | Change one rule's `tools`. |
| `DELETE /access/grants/{grant_id}` | Delete one rule; its key or user falls back to the next match. |
| `GET /access/check?api_key_id=&tool=` | Say whether a key may call a tool, and what decided: `key`, `user`, `everyone` (the default policy) or `server` (disabled, or the tool is not exposed). |

`GET /api/v1/workspaces/{workspace_id}/mcp-servers/keys/{api_key_id}/access`
answers the other way round: what one key reaches on each of the workspace's
servers, and which rule matched.

A rule's key must belong to the workspace. A rule's user must own a key in it, or
be someone the Playground admits (an active member of the workspace, or an owner
or admin of its organization), so a manager can restrict a member's Playground
before they hold a key; such a user is named by their identity id in lowercase. A
deleted user is refused. A user's rules on a workspace's servers are deleted when
they leave the workspace or its organization, so inviting them again restores
none. A server holds at most 1000 rules, so write a rule for a user rather than
one per key.

## Limits and safety

- `mcp_servers` and `mcp_server_ids` may be combined with Otari's built-in tools in one request. [Combining tools](tools.md#combining-tools) describes how, including when a server tool named like a built-in is refused
- `max_tool_iterations` optionally caps the loop; default is `10`, max is `25`. The last round answers rather than calling Otari's tools (see above)
- `mcp_server_ids` accepts at most 50 ids, which is also the most servers a workspace can have; a repeated id resolves once
- Server names must be unique across everything one request uses, because Otari routes each tool call to its server by name:
  - two `mcp_servers` entries sharing a name: `400` naming it;
  - an `mcp_servers` name colliding with a stored server of your workspace: `400` with a fixed detail that names neither server (rename or remove the inline entry);
  - two stored servers sharing a name: `500`, with the names in the log
- A URL whose host Otari cannot read is refused as invalid with a `400` before any address check, so an IPv6 host needs brackets (`http://[2400:8901::1]:8787/mcp`) and a port has to be a number
- MCP URLs are validated to reduce SSRF risk; by default, private and reserved addresses are blocked, loopback is allowed, and `http://` is rejected when `authorization_token` is present
- The managed MCP loop follows redirects only when the scheme, host, and port stay the same, or when a default-port `http://` URL upgrades to `https://` on the same host. Caller-orchestrated discovery and execution do not follow redirects. Configure the final URL when a managed-loop redirect changes host or port
- `OTARI_MCP_ALLOW_LOOPBACK=false` disables loopback; `OTARI_MCP_ALLOW_PRIVATE_HOSTS=true` relaxes the private-host restriction

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

The blocks report execution only: Otari runs the call, feeds the result back to
the model, and returns one logical Messages stream. It strips the blocks from
history echoed on later turns and adds no server URL, authorization token, or
headers to them, though a result carries whatever the tool returned. Chat Completions and Responses streams do not carry this
activity.

## Caller-orchestrated MCP

Otari has two alternative MCP integration modes:

- **Managed MCP** is the loop the sections above describe: you send
  `mcp_servers` or `mcp_server_ids` with a completion, and Otari owns the
  model/tool loop.
- **Caller-orchestrated MCP** hands the loop back to your application. Otari
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

Between the two requests Otari holds nothing open (no model stream, MCP
session, database session or worker-local state), so your application can pause
for a person for as long as it needs to.

Both endpoints refer to a server by the id it was registered under; there is no
inline server field. Local `stdio` servers are not supported, since a remote
Otari deployment cannot reach a machine-local process.

### Listing the servers a key can reach

```http
GET /api/v1/mcp/servers
Authorization: Bearer <token>
```

```json
{
  "servers": [
    {
      "id": "2c948a61-dc96-4cd8-96bb-8e1434bf424e",
      "name": "github",
      "purpose_hint": "Use for repository and issue lookups",
      "sign_in_required": false
    }
  ]
}
```

The enabled stored servers of the key's workspace whose
[access list](#server-access) grants it something, which is what to send in
`mcp_server_ids`. `sign_in_required` is true for a
[per-user server](#credential-mode) the key's user has not signed in to yet, and
`sign_in_url` is then a link that starts their sign-in, good for 30 minutes.
Making the link stores nothing, so listing costs nothing. A key with no user does
not see such a server at all. The URL and the token are
never returned. A master key has no
workspace and is refused with `404`, as on the other routes of this section; the
route is rate limited and refuses a blocked user as they are.

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
tools at all, never a partial list. The one exception is `warnings`, which names a tool
whose descriptor Otari could not carry, for example a schema that is too large or
one with a `$ref` pointing outside itself:

```json
{"tool_name": "create_issue", "code": "mcp_tool_schema_unsupported"}
```

A tool name outside the 1–256-character execution limit is omitted with
`mcp_tool_name_unsupported`, so every tool returned by discovery can be sent
back to the execution endpoint. The same code omits a name that contains `*`,
`,`, whitespace or a control character, or starts with `@`: an access rule would
read it as a pattern, and the dashboard's allow-list field would split it. A
connection check leaves such a tool out of the tools it lists too.

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
  "isError": false,
  "resultType": "complete"
}
```

`isError: true` is still an HTTP 200: the server handled the call and
classified the outcome itself. It is a definitive result, not a retry signal.

`server_revision` is the value discovery returned. Persist it with the proposed
call and send it back. It is an opaque revision of the stored server's URL,
credential, enabled state and allowlist, so a configuration change between your
authorization decision and this request is refused with
`409 mcp_server_changed`. It is not an approval credential.

It detects Otari-side configuration changes only. A remote server that changes
its own catalog, or a tool's behavior, behind an unchanged URL does not change
it. The calling application is responsible for binding the server id, revision,
remote tool name, and final arguments to its authorization or approval record.

`client_execution_id` is a canonical UUID you generate, for correlation across
your service and Otari. It is not proof of approval and **not an idempotency
key.**

Each execution is recorded in the workspace's [agent traces](traces.md) as a
step with one tool span: the tool, its server, whether it ran, and which rule
decided the call (the key's rule, its user's rule, the server's default policy,
or the server's own tool list). A call the rules refuse is recorded too, as a
blocked span, so a manager can see who tried a tool they may not use; that
includes a call to a server the key cannot reach at all, which the caller is
still told does not exist. The
arguments and the result are never recorded here. Send the run's
`Otari-Conversation-Id` header on each call to file it under the agent's
session rather than as a session of its own. Discovery is not recorded.

### Never retry an execution automatically

Repeating a request with the same `client_execution_id` may execute the tool
again. Otari implements no deduplication.

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
body; the `Otari-Request-ID` header carries it on success and failure alike. No error carries a server URL,
a credential, a header, your arguments, a result, or raw
exception text.

| Condition | Status | Code | Execution state |
|---|---:|---|---|
| Invalid request body | 422 | `invalid_request` | `not_started` |
| Authentication failure | 401 | `authentication_failed` | `not_started` |
| API key belongs to another regional deployment | 421 | `misdirected_request` | `not_started` |
| Payment required or insufficient funds | 402 | `payment_required` | `not_started` |
| Authorization refused | 403 | `forbidden` | `not_started` |
| Authenticated request rate exceeded | 429 | `rate_limit_exceeded` | `not_started` |
| Server inaccessible to caller (including one whose ACL denies the key), or disabled | 404 | `mcp_server_not_found` | `not_started` |
| Per-user server the key's user has not signed in to (the body adds `sign_in_url`) | 403 | `mcp_sign_in_required` | `not_started` |
| That user's sign-in has expired and cannot be renewed right now | 503 | `mcp_credentials_unavailable` | `not_started` |
| Stored server revision changed | 409 | `mcp_server_changed` | `not_started` |
| Tool absent from stored allowlist, or outside the server's ACL for the key | 403 | `mcp_tool_not_allowed` | `not_started` |
| Unsafe resolved URL | 400 | `unsafe_mcp_url` | `not_started` |
| Authentication refused for any other reason | 502 | `mcp_resolution_failed` | `not_started` |
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
| Discovery | returned tools, counted after the caller's access list | 200 |
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

Both endpoints resolve stored servers from the authenticated API key's
workspace; a master-key request holds no workspace of its own and reaches no
stored server, so it gets the `404`.
