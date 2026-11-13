# MCP server catalog

The catalog lists remote MCP servers you can add to a workspace without
looking up their URLs. Otari fills it by copying the official
[MCP Registry](https://registry.modelcontextprotocol.io), and adds a short
list of its own for well-known servers the registry does not list (see
[Servers the registry does not list](#servers-the-registry-does-not-list)).
You use it from
**Tools → MCP servers → Browse catalog**; the steps are in
[MCP servers](mcp-servers.md#add-servers-from-the-catalog).

Being listed permits nothing, and does not mean anyone checked the server. A
workspace reaches a server only once someone adds it there. A request can
still name a server by URL where the workspace allows it, and the master key
can do so even when the workspace does not.

## What you see

Before you search, the dialog shows **Popular**: about twenty well-known
servers, such as GitHub, Notion, Linear and Sentry, that the catalog holds
right now. The list ships with Otari and is matched to the catalog by URL, so
a server whose vendor moves it drops off until Otari's list is updated. Typing
in **Search the catalog** searches every server the catalog holds, a page at a
time.

Servers the workspace already has are not shown, on Popular or in a search.
What you tick moves to **Selected**, above both, and stays there when you
change or clear the search; untick it there to put it back.

If a server is not in the catalog, **Add it by URL** at the bottom of the
dialog opens the form that takes a URL.

A member who cannot add servers reads the same catalog from **Request a
server**, with **Request** on each server in place of the tick box, and asks
the workspace's admins for one. See
[Ask for a server](connect-your-account-to-an-mcp-server.md#ask-for-a-server)
and
[Answer members' requests](mcp-servers.md#answer-members-requests).

Each server shows its logo, or its first letter when none was found. The same
logo appears on the MCP servers lists, in **Connect a server** and on a
server's page. See [Logos](#logos).

An **Official** label marks a server its maker runs on its own domain, so
Figma's own server stands out among the gateways named after Figma. Hover it
to see the host. Among servers that match a search equally well, Official ones
come first. A server is Official when all of this holds:

- its endpoint lives on a domain of its publisher: its registry namespace, its
  GitHub organization or its website;
- that domain is not a hosting platform's, such as `vercel.app` or
  `workers.dev`, where anyone can take a name;
- its title is that domain's name and nothing more. "Stripe" on `stripe.com`
  and "Figma MCP Server" on `figma.com` are. "Fudge for Shopify" on `fudge.ai`
  is not, so nobody's helper for Shopify reads as Shopify's own.

Two servers the rule cannot recognize are listed by hand: GitHub's, which
lives on `githubcopilot.com`, and Atlassian's, whose title says more than
"Atlassian". Every Popular server is Official.

The label is read from names. Nobody at Otari checked the servers, and the
rule cannot tell a maker from someone who registered the maker's name under
another ending (`example.dev` for `example.com`). Check the host before you
trust a server with a credential. A server with no label is not suspect: many
are a company's own server for a product with a longer name.

## Loading the catalog

A new deployment starts with an empty catalog. Whoever can add MCP servers to
the workspace sees **Load catalog** in the dialog. Loading copies the registry,
which takes a couple of minutes, and the list fills in as it goes. Once the
catalog is more than a day old, the dialog offers **Refresh** instead.

To keep the registry from being called over and over, a workspace admin can
ask again only an hour after a load completes, or five minutes after one that
failed. That limit covers the whole deployment, however many people click.
An operator (the master key, or a deployment superuser) is not held to it.

## Settings

Set these in `config.yml`, or as `OTARI_<NAME>` environment variables.

| Setting | Meaning |
| --- | --- |
| `mcp_catalog_enabled` | Offer the catalog. Defaults to `true`. `false` removes **Browse catalog**, its routes and its sync worker; startup setting. Servers keep their logos. |
| `mcp_catalog_sync` | Whether Otari contacts the MCP Registry: `off`, `manual` (the default), or `scheduled`. |
| `mcp_catalog_sync_interval_hours` | Hours between syncs when `mcp_catalog_sync` is `scheduled`. Defaults to 24, from 1 to 720. |

The three sync modes:

- `off`: Otari never contacts the registry. The dialog says the catalog is
  turned off on this deployment, and a sync request is refused with a 409. Use
  it on a deployment with no outbound access. The catalog keeps whatever it
  already holds, so a fresh deployment has an empty one.
- `manual`: a sync runs only when someone loads or refreshes the catalog.
- `scheduled`: a sync also runs on its own, every
  `mcp_catalog_sync_interval_hours`.

Adding an MCP server by URL works in every mode, with or without a catalog.

## Servers the registry does not list

Some well-known remote servers are not in the MCP Registry. Otari ships a
short list of them, each confirmed against its maker's own documentation, and
a sync stores that list along with the registry's servers. Today it holds one:
Granola, at `https://mcp.granola.ai/mcp`. These report `source` as `curated`
where the registry's report `registry`.

A sync never deletes them or replaces them with a registry entry of another
name. Once the registry lists a server at the same URL, the registry's entry
takes over at the next sync. Like the rest of the catalog, they appear only
after a sync has run, so a deployment with `mcp_catalog_sync` set to `off`
does not get them.

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
servers the registry no longer lists. It also works out which servers are
Official, so a change to that rule in a new Otari release shows in the search
order after the next sync. A sync that stops earlier is `partial`:
it keeps the pages it read, removes nothing, and the next sync resumes at the
page where it stopped, for up to six hours. After that the next sync starts
over. It also starts over when the resumed sync cannot read its first page,
because the saved page may no longer be one the registry accepts. A sync that
could not read even its first page is `failed` and leaves the catalog as it
was.

A sync stops early when the registry fails three attempts at a page, when a
page takes more than three minutes, after 5,000 pages, or after 30 minutes. A
gateway that shuts down mid-sync ends it as `partial` too.

It also stops when the registry's answer points to the next page in a way it
cannot follow: a cursor that is not a non-empty string, or the same cursor it
was asked for. Reading the first as the last page would remove every server
the sync had not reached yet, and following the second would read the same
page forever. After a repeated cursor, the next sync starts over.

Two answers from the registry are not trusted, because a registry cut short is
more likely than one that emptied overnight, and both remove nothing:

- a last page reached with no server kept at all, reported as `failed`;
- a last page reached having seen fewer than half the servers the catalog held
  when the sync began, reported as `partial`. Servers the sync added along the
  way do not count toward that half, and a sync that resumes keeps the number
  from when it first began.

In `scheduled` mode, a sync that did not complete is tried again an hour later.

On a deployment with several replicas, every replica polls once a minute and
exactly one runs a due sync. If that replica dies mid-sync, another takes the
sync over after ten minutes without a sign of life and resumes it.

### Network

A sync calls `https://registry.modelcontextprotocol.io` and nothing else. It
follows no redirect and reads at most 4 MB per page. The registry's address
goes through the same check as an MCP server's, so it is refused if it resolves
to a private address, unless `mcp_allow_private_hosts` is on. Loopback
addresses pass by default (`mcp_allow_loopback`, on unless you turn it off),
even while private hosts are refused.

A sync never connects to the servers it lists. A server's address is resolved
and checked when someone adds it to a workspace, and again on every request
that uses it.

## Logos

The gateway finds each server's logo once, keeps it, and serves it from its own
address. Browsers never contact the vendor for it.

It looks in this order, and keeps the first image it finds:

1. A logo shipped with Otari, for seven Popular servers (Stripe, PayPal,
   Zapier, Cloudflare, Supabase, Sentry and Grafana). These come from
   [Simple Icons](https://simpleicons.org), whose icon data is CC0. The marks
   stay their owners' trademarks.
2. The image the server's registry entry names in its `icons`.
3. The publisher's and the website's `apple-touch-icon.png` and `favicon.ico`,
   then the icons the publisher's home page declares.
4. For an `io.github.*` name, the GitHub organization's avatar.

A server added by URL, and not from the catalog, gets the catalog's logo when
the catalog lists its URL. Otherwise the gateway looks at its host's site the
same way: `apple-touch-icon.png`, `favicon.ico`, then the icons the home page
declares, first on the domain the host belongs to and then on the host itself.

A logo is looked for the first time someone's browser asks for it, never while
the catalog syncs, and never while `mcp_catalog_sync` is `off`: then only the
logos shipped with Otari and those an admin uploads show. Each replica runs at most 4 looks at once and starts at most
30 a minute; past that, the logo shows its first letter and is tried on a later
view. A logo found is looked for again after a week; one not found is not
looked for again for a week, so a page view does not keep calling out.

Each fetch is held to the rules any outside address is: https only, every
address the name resolves to must be public, at most three redirects each
checked the same way, 5 seconds and 256 KB, 512 KB for a home page. The image
type is read from the bytes, not from the response headers, and an SVG with a
document type is refused. A logo is served only to a signed-in dashboard, as
an image that cannot run a script even when opened on its own.

The gateway also measures how bright a logo is. A dark logo gets a light tile
behind it on the dark theme, and a very light one a dark tile on the light
theme, so it never fades into the background.

### Your own logo for a server

On a server's page, an admin clicks the logo to open **Logo for** the server.
It shows the logo on the light and the dark theme side by side. From there:

- **Choose a PNG or SVG** to upload one. The browser crops it to a square and
  resizes it to 256 px, up to 512 KB, and the gateway checks it is a PNG again.
- **Tile behind the logo**: **Auto** adds a tile only on the theme where the
  logo would fade out, **Always** adds one on both, **Never** on neither.
- **Remove uploaded logo** goes back to the server's own logo.

An uploaded logo belongs to that server in that workspace. It is stored apart
from the logos the gateway fetches, so no sync or later fetch replaces it.

## API

Every route takes a session or the master key.

```bash
# Search, or page through everything with no q.
curl "https://otari.example.com/api/v1/mcp-catalog/servers?q=sentry&limit=25" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# The same, without the servers a workspace already has. Works on /popular too.
curl "https://otari.example.com/api/v1/mcp-catalog/servers?q=sentry&workspace_id=<workspace-id>" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# The Popular servers the catalog holds.
curl "https://otari.example.com/api/v1/mcp-catalog/popular" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# The logo of each of a workspace's servers, and whether it needs a tile.
curl "https://otari.example.com/api/v1/workspaces/<workspace-id>/mcp-servers/logos" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# An admin's own logo for a server, and the tile choice (auto, always or never).
curl -X PUT "https://otari.example.com/api/v1/workspaces/<workspace-id>/mcp-servers/<server-id>/logo" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -F tile=auto -F file=@logo.png

# Back to the server's own logo. The tile choice stays.
curl -X DELETE "https://otari.example.com/api/v1/workspaces/<workspace-id>/mcp-servers/<server-id>/logo" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# What the sync in progress, or the last one, did.
curl "https://otari.example.com/api/v1/mcp-catalog/sync" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"

# Start a sync within a minute. The master key needs no body.
curl -X POST "https://otari.example.com/api/v1/mcp-catalog/sync" -H "Otari-Key: Bearer $OTARI_MASTER_KEY"
```

Anyone else starting a sync sends `{"workspace_id": "<id>"}` for a workspace
they may add MCP servers to, and gets a 429 before `next_request_at`.

`q` matches a server's title, name and description. A title that starts with
`q` comes first, then a title that contains it, then a match in the name or
description alone. Within each, servers whose `publisher_kind` is `vendor`
come first.

`workspace_id` names a workspace the caller can see, and leaves out the
servers it already has, matched by URL with or without a trailing slash. A
workspace the caller cannot see is a 404.

The sync status carries the mode, whether a sync is running or waiting to
start, the last `outcome` (`complete`, `partial` or `failed`) with its `error`,
how many registry entries were read (`fetched`), stored (`kept`) and deleted
(`removed`), `server_count`, `is_stale` and `next_request_at`.

Each server reports `logo_url`, where this gateway serves its logo (null when
none was found), `logo_tone`, which says where the logo needs a tile,
`publisher_kind` (`vendor` is what the dialog labels Official, `community` is
everything else) and `source`, `registry` or `curated`.

Each server reports `requires_token`: the registry says a call needs an
`Authorization` header. `false` means the registry declared nothing, not that
the server is open. A server that signs users in with OAuth declares nothing.
