# MCP server catalog

The catalog is a list of remote MCP servers a workspace can add without anyone
hunting for a URL. Otari fills it by copying the official
[MCP Registry](https://registry.modelcontextprotocol.io), and each organization
can mark the servers it recommends on a featured shelf.

The catalog permits nothing. A workspace reaches an MCP server only once the
server is added to that workspace, so a workspace's own servers stay its allow
list. See [Add an MCP server](add-an-mcp-server.md).

## In the dashboard

Open **Organization → MCP catalog**.

- **Featured servers** is the organization's shelf. Owners and admins of the
  organization add to it with **Feature a server**, which searches the
  catalog, and take an entry off with **Remove**. Every member can read it.
- **Registry sync** is shown to operators. It says what the last sync did and
  how many servers the catalog holds, and **Sync now** starts one within a
  minute. The button is disabled while `mcp_catalog_sync` is `off`.

People add servers from the catalog on **Tools → MCP servers**, with
**Browse catalog**. See [Add an MCP server](add-an-mcp-server.md).

## Settings

Set these in `config.yml`, or as `OTARI_<NAME>` environment variables.

| Setting | Meaning |
| --- | --- |
| `mcp_catalog_enabled` | Offer the catalog. Defaults to `true`. `false` removes its routes and its sync worker; startup setting. |
| `mcp_catalog_sync` | Whether Otari contacts the MCP Registry: `off`, `manual` (the default), or `scheduled`. |
| `mcp_catalog_sync_interval_hours` | Hours between syncs when `mcp_catalog_sync` is `scheduled`. Defaults to 24, from 1 to 720. |

The three sync modes:

- `off`: Otari never contacts the registry, and a sync request is refused with
  a 409. Use it on a deployment with no outbound access. The catalog keeps
  whatever it already holds, so a fresh deployment has an empty one.
- `manual`: a sync runs only when an operator asks for one.
- `scheduled`: a sync also runs on its own, every
  `mcp_catalog_sync_interval_hours`.

Everything else works with an empty catalog. Adding an MCP server by URL does
not depend on it.

## What a sync does

A sync reads the registry a page at a time, 100 servers to a page, and stores
each page before it asks for the next. It keeps a server only if Otari can
connect to it:

- the registry does not mark it deprecated or deleted;
- it has a streamable HTTP remote, at a fixed `https` URL with a public host
  name, with no credentials in the URL;
- it needs no header other than `Authorization`, which is the one credential a
  stored server sends.

A package to run locally, an SSE-only server, a URL with a placeholder to fill
in, and a deprecated entry are left out. A server with no title gets one from
its name: `com.stripe/mcp` reads as Stripe.

A sync that reaches the registry's last page is `complete`, and removes the
servers the registry no longer lists. A sync that stops earlier is `partial`:
it keeps the pages it read, removes nothing, and the next sync resumes at the
page where it stopped, for up to six hours. After that the next sync starts
over. A sync that could not read even its first page is `failed` and leaves the
catalog as it was.

A sync stops early when the registry fails three attempts at a page, when a
page takes more than three minutes, after 5,000 pages, or after 30 minutes. A
gateway that shuts down mid-sync ends it as `partial` too.

Two answers from the registry are not trusted, because a registry cut short is
more likely than one that emptied overnight, and both remove nothing:

- a last page reached with no server kept at all, reported as `failed`;
- a last page reached with fewer than half the servers the catalog holds,
  reported as `partial`.

In `scheduled` mode, a sync that did not complete is tried again an hour later.

On a deployment with several replicas, every replica polls once a minute and
exactly one runs a due sync. If that replica dies mid-sync, another takes the
sync over after ten minutes without a sign of life and resumes it. The first
replica, should it come back, stops without storing anything more.

### Network

A sync calls `https://registry.modelcontextprotocol.io` and nothing else. It
follows no redirect and reads at most 4 MB per page. The registry's address
goes through the same check as an MCP server's, so it is refused if it resolves
to a private address, unless `mcp_allow_private_hosts` is on.

A sync never connects to the servers it lists. A server's address is resolved
and checked when someone adds it to a workspace, and again on every request
that uses it.

## Running and reading a sync

Both routes are for operators. The dashboard's **Sync now** calls the first.

```bash
# Start a sync within a minute.
curl -X POST "https://otari.example.com/api/v1/mcp-catalog/sync" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# Read what the sync in progress, or the last one, did.
curl "https://otari.example.com/api/v1/mcp-catalog/sync" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"
```

The status carries the mode, whether a sync is running or waiting to start,
the last `outcome` (`complete`, `partial` or `failed`) with its `error`, how
many registry entries were read (`fetched`), stored (`kept`) and deleted
(`removed`), and `server_count`, the size of the catalog now.

## Searching the catalog

Any signed-in caller can search:

```bash
curl "https://otari.example.com/api/v1/mcp-catalog/servers?q=sentry&limit=25" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"
```

`q` matches a server's title, name and description. A title that starts with
`q` comes first, then a title that contains it, then a match in the name or
description alone. Leave `q` out to page through the whole catalog with `skip`
and `limit` (at most 100).

Each server reports `requires_token`: the registry says a call needs an
`Authorization` header. `false` means the registry declared nothing, not that
the server is open. A server that signs users in with OAuth declares nothing.

## The featured shelf

An organization's owners and admins mark the catalog servers they recommend.
Every member of the organization sees the shelf. An organization features at
most 50 servers.

The shelf belongs to the organization, not to the deployment. Which servers are
vetted is a company's call, and on a deployment that serves several
organizations each one makes its own.

```bash
# Feature a catalog server, by the id the search returned.
curl -X POST "https://otari.example.com/api/v1/organizations/me/mcp-catalog/featured" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"server_id": "<id>"}'

# List the shelf.
curl "https://otari.example.com/api/v1/organizations/me/mcp-catalog/featured" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# Stop featuring an entry, by the id the shelf returned.
curl -X DELETE "https://otari.example.com/api/v1/organizations/me/mcp-catalog/featured/<featured_id>" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY"
```

A featured entry is the organization's own copy of the catalog server, taken
when it was featured. A sync never changes or removes it. Each entry reports
how it compares with the catalog today, as `catalog_status`:

| `catalog_status` | Meaning |
| --- | --- |
| `current` | The catalog lists the server at the URL that was featured. |
| `changed` | The catalog now lists another URL. The entry keeps the one that was featured. |
| `removed` | The catalog no longer lists the server. The entry stays as it was. |

To accept a changed URL, feature the server again, by the `catalog_server_id`
the entry reports: the entry takes a fresh copy. In the dashboard that is
**Update** on the entry's row.

Removing an entry from the shelf does not touch servers already added to a
workspace.
