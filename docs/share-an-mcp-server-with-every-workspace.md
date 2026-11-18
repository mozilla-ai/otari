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

## Replace workspaces' own copies

Workspaces may already have added the same server on their own. When you add a
server for the organization, or make a workspace's server the organization's,
Otari lists the workspaces' servers at the same URL and offers to fold them in.
A copy you fold:

- gives its access rules, default policy, availability and on or off state to its
  workspace's settings on the organization's server, and its allowed tools become
  the organization's limit for that workspace. They are kept even where they
  match the organization server's today, so a later change to the organization's
  server does not change what that workspace gets;
- keeps its id working in its workspace, so agents and links that name it still
  reach the server;
- is then deleted.

Its workspace keeps what it reached before only within the organization server's
own allowed tools. If the copy allowed write tools and the organization's server
allows only read tools, the workspace loses the write tools.

You can fold one copy of each workspace: where a workspace has several, tick the
one to fold. An organization server that is off can replace only copies that are
off too, since no workspace can turn it back on, so a copy that is on is listed
but cannot be ticked.

What happens to the people signed in to a copy is counted before you confirm,
for each one:

- A person's own sign-in is kept when the organization's server also uses
  **Personal accounts**, has exactly the URL the sign-in was made for, and uses
  the same OAuth client, which can still renew it: the sign-in must hold a
  refresh token.
- A person who already has a sign-in on the organization's server keeps using
  that one. It may be a different account upstream from the one they used on
  the copy, so these people are counted separately.
- Every other sign-in ends. On an organization server that uses
  **Personal accounts**, the person signs in again the next time they use it;
  nothing breaks meanwhile, as they are handed the usual sign-in link. On one
  with a shared account or a token, that credential serves them and nobody
  signs in.
- A copy's own token or shared account is replaced by the organization
  server's credential, which may be a different account upstream.

A copy is ticked by default when folding it changes nobody's account: it signs
in the same way as the organization's server, neither the copy nor the
organization's server holds a token or shared account, and nobody signed in to
it would sign in again or switch to another sign-in. The others are listed but
not ticked, and you can tick them yourself. A copy you leave unticked stays its
workspace's own.

To make a workspace's own server the organization's, open it under **Tools** →
**MCP servers** as an organization owner or admin and choose to make it an
organization server. It keeps its id, its settings and every sign-in to it,
which then work in every workspace. Other servers in its own workspace at the
same URL are not offered for folding and stay that workspace's own. Members who
asked for the server in another workspace that now reaches it have their request
approved; a server that is off reaches no workspace, so their requests stay
pending.

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

A workspace's **Tools** → **MCP servers** lists the organization's servers beside
its own, marked as the organization's. Their definition is read-only there: the
organization's owners and admins change it.

A workspace's own admins decide how the workspace uses one:

- turn it off for the workspace, and back on unless the organization's limit
  turned it off there;
- set the workspace's own default policy, what every key of the workspace may
  call;
- choose whether the workspace attaches it to every request;
- add access rules for keys and members of the workspace, as on a workspace
  server.

The organization's limit, when it set one, is shown above these settings: it is
the most the workspace reaches, whatever they say. Using the organization's
defaults again takes the workspace's own settings away and keeps the limit.

A workspace's requests reach the organization's servers with its own by id, in
`mcp_server_ids`, and attached ones join its requests as its own do. A server the
organization or the workspace turned off for it answers as an unknown id does,
except by the id of a folded copy that was off: a request naming that id skips
the server, as it skipped the copy before.

Each person signs in to a personal server from a workspace that reaches it. Their
sign-in lasts while they belong to any workspace of the organization, or are one
of its owners or admins. Leaving one workspace ends only the access rules and
sign-ins started there. Turning a server off for a workspace ends no sign-in.
