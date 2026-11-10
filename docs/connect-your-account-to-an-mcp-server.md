---
icon: user-check
---

# Connect your account to an MCP server

An MCP server can use one token for the whole workspace, or each person's own
account. With one token, everyone who uses the server acts as whoever that token
belongs to. With each person's own account, your API keys and your Playground
messages reach the server as you, and see what you are allowed to see there.

This page covers the second setup. [Credential mode](mcp.md#credential-mode)
describes how it works.

## Before you start

- The deployment needs `public_base_url` and `OTARI_SECRET_KEY` set. The first
  is the address the server sends you back to after you sign in, and the second
  encrypts the tokens Otari keeps for you.
- The MCP server needs an `https://` URL and has to support sign-in through
  OAuth, as the MCP authorization specification describes.

## Set a server to use each person's own account

You need to be an owner or admin of the workspace or of the organization.

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **MCP servers**, and click **Add MCP server**.
3. Enter a **Name** and the server's **URL**.
4. Under **Credentials**, choose **Each user signs in**.
5. Leave **OAuth client ID** blank. Otari registers itself with the server on
   the first sign-in.
6. Click **Add MCP server**.

To change a server you already have, open its page, click **Edit** under
**Connection**, and choose **Each user signs in** there. The shared token it had
is deleted.

### If the server does not let Otari register itself

Some servers accept only clients their owner has created. Starting a sign-in
then says to add an OAuth client ID.

1. Create an OAuth client with the server's provider. Give it this redirect
   URI, with your own `public_base_url` in front:
   `https://otari.example.com/api/v1/mcp-connections/callback`.
2. On the server's page, click **Edit** under **Connection**.
3. Enter the **OAuth client ID**, and the **OAuth client secret** if the
   provider gave you one.
4. Click **Save server**.

A client ID and secret belong to one host. If you later change the server's
**URL** to another host, saving clears them, and you enter the new host's
client or leave it blank for Otari to register itself.

## Connect your account

Anyone who can see the workspace can do this.

1. Go to **Tools**, then **MCP servers**, and click the server.
2. Under **Your account**, click **Connect your account**.
3. Sign in on the server's own page and approve the access it asks for.

You come back to the server's page, which now says **Connected**. From then on
every API key you own in this workspace, and the Playground, use the server with
your account.

Connecting does not change who may use the server. Its
[access list](configure-an-mcp-server.md#control-who-can-use-it) still decides that: a key the
access list denies is told the server does not exist, whether or not you have
connected.

The link you are sent to works once, for ten minutes, and only in the browser
session that started it.

A manager who has connected can then check the server and choose its tools: on
the server's page, click **Check connection** under **Exposed tools**.

## Connect from your coding agent instead

You do not have to open the dashboard. When a workspace manager has clicked
**Attach to every request** on the server, or your client names the server in
`mcp_server_ids`, just ask your agent for something the server does:

1. Ask, for example "check my open Jira tasks".
2. The agent answers with a sign-in link. Open it within 30 minutes and sign in
   with your own account.
3. A page says you are signed in. Go back to the conversation and ask again.

The second ask uses your account.

Before you open the link, check where it goes. A real one is on your own Otari
address, exactly, followed by `/api/v1/mcp-connections/sign-in?ticket=`, for
example `https://otari.example.com/api/v1/mcp-connections/sign-in?ticket=...`.
An address that only starts the same way, such as
`https://otari.example.com.other.site/...`, is not Otari: do not open it.

Keep the link to yourself. It is made for you, from the key your agent uses, and
whoever opens it within its 30 minutes connects the account they sign in with to
you.

## What your keys see before you connect

A key the access list allows, whose user has not connected, is told to sign in
instead of getting a generic failure:

- `GET /api/v1/mcp/servers` lists the server with `sign_in_required: true` and a
  `sign_in_url` to open.
- A model request that reaches the server is not refused. The model is given the
  sign-in link to show, as in [Connect from your coding agent
  instead](#connect-from-your-coding-agent-instead).
- Tool discovery and execution answer `403` with the code
  `mcp_sign_in_required` and the same `sign_in_url`.

A key created without a user cannot use a server set up this way. Listing does
not show it to such a key, and a model request that names it is told the key
belongs to nobody. Create the key for a user instead.

## Disconnect

- **Your own account.** On the server's page, click **Disconnect** under
  **Your account**. Otari deletes your tokens and asks the server to revoke
  them. That request is best effort: not every server offers it, and it can
  fail. To be sure the access has ended, also remove Otari from your account
  with the server's provider.
- **Somebody else's.** Owners and admins see **Connected users** on the server's
  page. Click the bin on a row, then **Remove connection**.

A connection also ends when the server is deleted, when the person leaves the
workspace or is deleted, and when a manager changes the server's URL, its OAuth
client or its **Credentials** setting. Everyone connects again afterwards.

## If a connection stops working

Otari renews your sign-in on its own for as long as the server allows it. When
the server ends it, for example because you revoked Otari's access there, your
connection is removed and the page says **Not connected** again. Click
**Connect your account** to sign in again.

Otari only learns this when it next renews your sign-in. Until then the page can
still say **Connected** while calls to the server fail. Click **Reconnect** to
sign in again straight away.
