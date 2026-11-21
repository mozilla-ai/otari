---
icon: server
---

# Share an MCP server with every workspace

An organization can add an MCP server once for all of its workspaces, instead of
adding a copy to each one. Every workspace reaches it, each person signs in to it
once for all of them, and the organization decides how far each workspace may use
it. A server one workspace adds under **Tools** → **MCP servers** stays that
workspace's own, as described in [MCP servers](mcp-servers.md).

Only the organization's owners and admins add and change these servers.

## Add a server for the whole organization

1. Go to **Organization** → **MCP servers**.
2. Click **Add server**.
3. Fill in the server as you would for a workspace: its name, URL, sign-in mode,
   token or OAuth client, purpose hint, allowed tools, availability and default
   policy. The fields are the same as in [Connect a server](mcp-servers.md#connect-a-server).
4. Save it.

Its name has to be free across the organization: no workspace may already have a
server of that name, since a workspace cannot reach two servers of one name. For
the same reason, a workspace cannot later add a server of its own with that name.

What you set here is what every workspace starts from:

- **Allowed tools** is the most any workspace reaches on the server.
- **Availability** decides whether each workspace attaches it to every request.
- **Default policy** is what every key of each workspace may call, until the
  workspace or the organization says otherwise.
- **Off** turns it off in every workspace at once.

A shared account, or a token, serves every workspace. A server in personal mode
takes each person's own sign-in, and one sign-in works in every workspace that
reaches the server.

## Limit what one workspace reaches

The server's page lists every workspace of the organization under **Workspaces**,
with whether it reaches the server, the organization's limit there, and whether
the workspace's own admins have narrowed it or turned it off.

To limit one workspace, click **Set limit…** on its row and choose what it may
reach: any tool, the read tools only, specific tools, or nothing. Nothing turns
the server off for that workspace, and its admins cannot turn it back on. Click
**Save limit**. **Lift limit** takes the limit away again.

A limit is a ceiling. The workspace's own rules still apply under it and can only
narrow it: a workspace rule that allows every tool reaches only what the limit
allows.

## What a workspace sees

A workspace's requests reach the organization's servers with its own by id, in
`mcp_server_ids`, and attached ones join its requests as its own do. A server the
organization or the workspace turned off for it answers as an unknown id does.

Each person signs in to a personal server from a workspace that reaches it. Their
sign-in lasts while they belong to any workspace of the organization, or are one
of its owners or admins. Leaving one workspace ends only the access rules and
sign-ins started there.
