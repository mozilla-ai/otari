# Read Otari from your agent

Otari serves an MCP server at `/mcp`. An agent connected to it reads what you
see on the dashboard (spend, usage, requests, budgets, keys and workspaces)
without anyone opening the dashboard. It reads as you, with your role, in one
organization, and it never changes anything.

The agent reads with an access token: a credential of your own, separate from
the API keys that call models. A client that supports MCP sign-in gets one by
asking you to approve it in the browser; any other client is given one you
create. Either way it reads the organization you were in when you approved or
created it, even after you switch the dashboard to another.

## Connect with sign-in

Point the client at your Otari address followed by `/mcp`. For Claude Code:

```bash
claude mcp add --transport http otari https://otari.example.com/mcp
```

Then run `/mcp` in Claude Code, select `otari` and choose to authenticate. A
browser opens on Otari: sign in if you are not, check the organization it
names, pick how long the access lasts (up to 90 days) and approve. Otari marks
the client as unverified, because it registered itself.

The client is given an access token that lasts an hour and works only through
`/mcp`, and a refresh token it uses to get the next one until the time you
picked runs out. The sign-in appears under **Access tokens** on your **Account**
page as `Claude Code (MCP)`, where deleting it signs the client out.

Other clients that support MCP sign-in work the same way: add the server URL
and approve in the browser when asked.

## Create an access token

For a client without MCP sign-in, or a script:

1. Open **Account** from the menu under your name.
2. Under **Access tokens**, click **Create token**.
3. Name it after where it will live, such as `CI report`, and pick when it
   expires, up to a year.
4. Copy the token. It starts with `otari-pat-` and is shown only once.

The token reads the organization you are in when you create it. You can hold up
to 50 tokens and sign-ins at once; ended ones do not count. Delete one on the
same page to revoke it at once.

## Connect with a token

Send the token as a bearer token. For Claude Code:

```bash
claude mcp add --transport http otari https://otari.example.com/mcp \
  --header "Authorization: Bearer otari-pat-..."
```

For a client configured with JSON:

```json
{
  "mcpServers": {
    "otari": {
      "url": "https://otari.example.com/mcp",
      "headers": { "Authorization": "Bearer otari-pat-..." }
    }
  }
}
```

Then ask it something like "what did we spend on each model last week?".

## What the agent can read

| Tool | What it answers |
| --- | --- |
| `get_context` | Who the token belongs to, the organization it acts in and your role there |
| `list_workspaces` | The workspaces you can see |
| `get_overview` | Active keys and members, and how budgets and spend ceilings stand |
| `get_usage_summary` | Spend, tokens and requests over a window, broken down by model, user, key and more |
| `list_requests` | Individual requests, newest first |
| `get_request` | The trace of one request, by its `Otari-Request-ID` |
| `list_budgets` | Budgets and spend ceilings, with what each has spent |
| `list_api_keys` | The API keys you can see, never the keys themselves |
| `api_get` | Any other `GET` endpoint of the [management API](api-reference.md) |

Each tool answers what the matching page answers you in the token's
organization: an owner sees the whole organization, a member sees the
workspaces they belong to. If you operate the deployment, that includes its
settings and every key, member and trace. A few reads that do more than read,
such as asking a provider which models it serves, refuse access tokens.

## Use a token without MCP

A token you created authenticates any `GET` request to the management API, so a
script can read the same data:

```bash
curl https://otari.example.com/api/v1/organizations/me/usage/summary \
  -H "Authorization: Bearer otari-pat-..."
```

A token is refused on every other method with `403`, and it is not an API key:
it cannot call models or tools. The access token an MCP client signs in with is
refused here with `401`, since it works only through `/mcp`.

## When a token stops working

A token stops at once when you delete it or when it expires. Every token and
sign-in you hold also ends whenever Otari ends your dashboard sessions:

- you change or reset your password;
- you prove your email address by signing in with a provider, after a password
  was set on it before the address was proven;
- an operator deactivates your account, or blocks your email address or domain;
- you leave, or are removed from, the organization it reads;
- your organization's single sign-on enforcement starts refusing you;
- the master key is rotated.

Reactivating an account does not bring its tokens back. Signing out of the
dashboard leaves tokens working. An MCP client's sign-in also ends when the
client revokes it, or when one of its refresh tokens is used a second time,
which only a copy someone else kept would do.

## Turn the server off

The server is on by default. To turn it off on a deployment you run, set
`mcp_server_enabled: false` in the config (or `OTARI_MCP_SERVER_ENABLED=false`)
and restart; MCP clients can then no longer register themselves either. Access
tokens still work against the management API. How sign-in for MCP clients works
is in [OAuth apps](oauth-apps.md#mcp-clients-register-themselves).
