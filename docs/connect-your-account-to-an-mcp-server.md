# Connect your account to an MCP server

An MCP server can use one account for the whole workspace, or each person's own
account. With one account, everyone who uses the server acts as whoever that
account belongs to. With each person's own account, your API keys and your Playground
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
4. Under **Sign-in mode**, choose **Personal accounts**.
5. Leave **OAuth client ID** blank. Otari registers itself with the server on
   the first sign-in.
6. Click **Add MCP server**.

To change a server you already have, open its page, click **Settings**, choose
**Personal accounts** under **Sign-in mode**, and click **Save**. The shared token
or shared account it had is removed.

### If the server does not let Otari register itself

Some servers accept only clients their owner has created. Starting a sign-in
then says what happened, and to add an OAuth client ID: either the server
offers no way to register, or it refused the registration, with the HTTP status
it answered.

Some of these servers read a client's details from a public `https://` address
instead of registering it. Otari offers its own when `public_base_url` is such
an address, so nothing is needed there. On a deployment reached at `localhost`
or over plain `http://` the server cannot read it, and the message says so.

1. Create an OAuth client with the server's provider. Give it this redirect
   URI, with your own `public_base_url` in front:
   `https://otari.example.com/api/v1/mcp-connections/callback`.
2. On the server's page, click **Settings**. A message that asks for a client
   ID also has an **Open Settings** button, which lands on the field.
3. Under **OAuth client**, enter the **OAuth client ID**, and the **Client secret** beside it if the
   provider gave you one. Some providers, GitHub among them, refuse to issue a
   token without it. The secret is stored encrypted and never shown again; the
   page says only whether one is stored. To drop it later, tick **Remove the
   stored secret**. Clearing the client ID clears the secret with it.
4. Click **Save**.

A client ID and secret belong to one host. If you later change the server's
**URL** to another host, saving clears them, and you enter the new host's
client or leave it blank for Otari to register itself.

## Connect your account

Anyone who can see the workspace can do this.

1. Go to **Tools**, then **MCP servers**. Click **Connect a server** and pick
   the server, or click a server you already use and, under **Your account**,
   click **Connect your account**. An owner or admin has no **Your account**
   section: click **Connect your account** at the top of the server's page.
   Once connected it reads **Your account: connected**, and opens **Reconnect**
   and **Disconnect**.
2. The server's own sign-in opens in a small window. Sign in and approve the
   access it asks for.

The window closes itself and the page you were on updates: the server appears
among **Your servers** as connected. If your browser blocks the window, the whole page goes to the
server's sign-in instead and comes back afterwards. From then on every API key
you own in this workspace, and the Playground, use the server with your account.
Connecting also lists the server's tools once with your sign-in, so its status
in the server list is current straight away.

**Your servers** shows a card for each server you use: a **Shared** chip on a
server that uses one account for everyone, or the date you connected your own,
and whether the access rules give you all of its tools, a restricted set, or
only the ones that read. A Shared server an admin has not set up yet is not
listed. Click a card to open the
server's page.

The server's page lists **Your tools**: the tools the access rules let you use
on it, with a line about each where the server describes it, a **Read-only**
mark on the tools that only read, and a search over both.

Connecting does not change who may use the server. Its
[access list](mcp-servers.md#control-who-can-use-it) still decides that: a key the
access list denies is told the server does not exist, whether or not you have
connected.

The link you are sent to works once, for ten minutes, and only in the browser
session that started it.

A manager checks the server and chooses its tools apart from any connection: on
the server's page, click **Check** under **Exposed tools**. See
[Check a server](mcp-servers.md#check-a-server).

## Connect from your coding agent instead

You do not have to open the dashboard. When a workspace manager has set the
server's **Availability** to **Every request**, or your client names the server in
`mcp_server_ids`, just ask your agent for something the server does:

1. Ask, for example "check my open Jira tasks".
2. The agent answers with a sign-in link. Open it within 30 minutes.
3. An Otari page names the server and the Otari user your account will be
   attached to. Check that the user is you, then click **This is me, continue**
   and sign in with your own account.
4. A page says you are signed in. Go back to the conversation and ask again.

The second ask uses your account.

Before you open the link, check where it goes. A real one is on your own Otari
address, exactly, followed by `/api/v1/mcp-connections/sign-in?ticket=`, for
example `https://otari.example.com/api/v1/mcp-connections/sign-in?ticket=...`.
An address that only starts the same way, such as
`https://otari.example.com.other.site/...`, is not Otari: do not open it.

Keep the link to yourself. It is made for you, from the key your agent uses, and
whoever opens it within its 30 minutes and continues connects the account they
sign in with to you.

Never open a sign-in link someone else sent you. It is made for their Otari
user, so signing in through it would hand them the use of your account on that
server. The page the link opens names the user it is for: if that is not you, or
you did not ask for the link yourself, close the page. If you are signed in to
the Otari dashboard as a different user, the page refuses to continue.

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

## Ask for a server

**Connect a server** lists the servers the workspace already has. For one it
does not have, ask its admins:

1. Go to **Tools**, then **MCP servers**. Click **Connect a server**, then
   **Request a server**. With no servers yet, **Request a server** is on the
   page itself.
2. Pick from **Popular**, or type in **Search the catalog**. Servers the
   workspace already has are not listed.
3. Click **Request** on the server. Under **What do you need it for?** you can
   leave a short note for the admins, up to 500 characters.
4. Click **Send request**.

The server then reads **Requested**, and the request is listed under **Your
requests** at the top of the same dialog as **Waiting for an admin**. Click
**Withdraw** there to take it back.

Asking adds nothing. An owner or admin of the workspace, or of the
organization, adds the server or declines:

- **Added.** A line at the top of your MCP servers page says the server was
  added. For a **Personal** server it has a **Connect** button, which starts
  the sign-in with your own account. For a **Shared** server it says the server
  is ready, or that an admin still has to set up its account.
- **Declined.** A line at the top of the page says so, with the admin's reason
  when they gave one. The request also reads **Declined** under **Your
  requests** until you click **Dismiss**. You can ask again, and the new
  request replaces the declined one.

Each line stays until you dismiss it with its **x**, on every device you sign
in from. Connecting your account to the server dismisses it too. Where the
deployment sends mail, you are also emailed either answer. Any member who
can see the workspace can ask, a viewer included. You can have 10 requests
waiting in a workspace and make 10 there in a day, and one per server at a
time; several members can ask for the same server. A request you have not
heard back on is withdrawn when you leave the workspace. Requests need the
catalog, so a deployment without it offers no **Request a server**.

## Disconnect

- **Your own account.** Click **Disconnect** on the server's card under
  **Tools → MCP servers**, or under **Your account** on its page. An owner or
  admin clicks **Your account: connected** at the top of the server's page,
  then **Disconnect**, or the bin on their own row under **Connected users**,
  which is marked **(you)**. Otari deletes your tokens and asks the server to revoke
  them. That request is best effort: not every server offers it, and it can
  fail. To be sure the access has ended, also remove Otari from your account
  with the server's provider.
- **Somebody else's.** Owners and admins see **Connected users** on the server's
  page. Click the bin on a row, then **Remove connection**.

A connection also ends when the server is deleted, when the person leaves the
workspace or is deleted, and when a manager changes the server's URL, its OAuth
client or its **Sign-in mode**. Everyone connects again afterwards.

## If a connection stops working

Otari renews your sign-in on its own for as long as the server allows it. When
the server ends it, for example because you revoked Otari's access there, your
connection is removed and the page says **Not connected** again. Click
**Connect your account** to sign in again.

Otari only learns this when it next renews your sign-in. Until then the page can
still say **Connected** while calls to the server fail. Click **Reconnect** to
sign in again straight away.
