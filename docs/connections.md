# Connections

A connection is a third-party account (Slack, GitHub, Google, Microsoft,
Notion, Linear, Atlassian, HubSpot, Typeform, Salesforce) that a user of
*your* application has authorized Otari to act on. You send the user to a
link; Otari runs the OAuth consent flow, stores the tokens encrypted, refreshes
them, and hands the user back to your page. From then on your application, or
anything Otari runs on its behalf, can act on that account without ever
touching an OAuth code.

This is the account-connection layer from Octonous, mozilla.ai's agent
product, brought into otari so that anyone building an agent on the gateway
gets it without writing OAuth. The protocol mechanics come from
[apron-auth](https://github.com/mozilla-ai/apron-auth); Otari adds the
deployment's part: which apps are configured, the pending state between the
two halves of a flow, the encrypted rows, and the identity of your users.

Standalone mode only.

## Who the user is

Your application talks to Otari with an API key and names its users with the
`user` string it already puts on completion requests. That string, scoped to
the API key's workspace, is what a connection belongs to. Two applications
naming a user `alice` never see each other's connections. Otari never needs to
know who alice is beyond that string.

## Configure the apps

Register an OAuth app with each provider, set its redirect URI to
`{public_base_url}/connected-accounts/{provider}/callback`, and put the client
credentials in `config.yml`:

```yaml
public_base_url: https://otari.example.com   # the redirect URI is derived from it

connected_apps:
  slack:
    client_id: ${SLACK_CLIENT_ID}
    client_secret: ${SLACK_CLIENT_SECRET}
    scopes: [chat:write, channels:join, im:write, reactions:write, users:read]   # bot scopes
    user_scopes: [channels:read, channels:history, chat:write, im:write, users:read, team:read]
  github:
    client_id: ${GITHUB_CLIENT_ID}
    client_secret: ${GITHUB_CLIENT_SECRET}
    scopes: [repo, read:org]
```

`scopes` (and, for Slack, `user_scopes`) are added to the provider preset's
base scopes, which cover identity so Otari can tell which account was
connected. Tokens are encrypted with `OTARI_SECRET_KEY`, so it must be set. An
app missing either credential, or a deployment without `public_base_url`, is
not offered: `GET /v1/connections/apps` does not list it and
`POST /{provider}/authorize` answers 400.

## Connect an account

```python
import httpx

otari = httpx.Client(base_url="https://otari.example.com", headers={"Authorization": f"Bearer {API_KEY}"})

# 1. Get a link and put it behind a "Connect Slack" button.
link = otari.post("/v1/connections/slack/authorize", json={
    "user": "alice@acme.com",
    "return_url": "https://myapp.com/settings",
}).json()["authorization_url"]

# 2. The user's browser visits the link, consents, and lands back on
#    https://myapp.com/settings?connection=ok&provider=slack&connection_id=…
#    (or ?connection=error&provider=slack&reason=access_denied).

# 3. Your settings page shows what alice has connected.
otari.get("/v1/connections", params={"user": "alice@acme.com"}).json()
# {"count": 1, "data": [{"provider": "slack", "account_label": "Alice (Acme Corp)", "scopes": [...], ...}]}
```

The link is good for ten minutes and one exchange. Otari records the state
with your user, the provider, the PKCE verifier (encrypted) and the return URL
when *you* start the flow, so the browser's callback carries nothing Otari has
to trust. `return_url` must be https, or http on localhost for development,
and is stored with the state rather than read back from the browser. Without
one, the browser lands on a page of this deployment.

## Use a connection

| Call | Does |
|---|---|
| `GET /v1/connections/apps?user=` | Apps this deployment can connect, the scopes each asks for with labels, and how many accounts the user holds per app. |
| `GET /v1/connections?user=&provider=` | The user's connected accounts. Tokens never appear. |
| `GET /v1/connections/{provider}/token?user=` | The live credential, refreshed first when it expires within a minute. Slack's user token comes alongside the bot token under `extra.user`. For code that calls the app itself. |
| `PATCH /v1/connections/{id}?user=` | Set the user's label ("Work Slack"). |
| `DELETE /v1/connections/{id}?user=` | Revoke at the provider where supported, then forget the tokens. |

The token endpoint is the one place a credential leaves Otari, and it goes
only to the application whose users these are, which is also the application
that registered the OAuth client. In-process consumers use
`ConnectedAccountService.access_token_for_provider(workspace_id, user, provider)`.

## What comes next

Connections exist so that Otari can act with them. The follow-ups, in order:

- **`credential: "connected"` on `mcp_servers`** and on built-in connectors:
  a completion request that names a `user` gets that user's token injected,
  and the application never sees it.
- **`connection_required`** as a structured outcome (a response field and a
  streaming event) when a request needs an app the user has not connected or
  has connected with too few scopes, carrying the authorization link, so the
  agent loop can pause for consent and resume the way Octonous's does.
- A dashboard page for operators to connect their own accounts, which is the
  same API with the operator as the user.
