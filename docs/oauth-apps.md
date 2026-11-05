# Connect an app with OAuth

Otari can deliver a revocable API key to an external application after the user
signs in and approves a workspace. The key inherits the user's model and tool
restrictions, remains subject to budgets, and appears on the API keys page as
`<app name> (OAuth)`. Revoke it there to stop the application immediately.

## Register an application

Register each public client in your gateway config:

```yaml
oauth_apps:
  thunderbolt:
    name: Thunderbolt
    redirect_uris:
      - https://your-thunderbolt.example/oauth/otari
      - https://app.thunderbolt.io/oauth/callback
      - http://localhost/
```

Restart the gateway after changing registration. Empty `oauth_apps` disables
these endpoints. Removing a registration prevents new approvals and exchanges;
revoke already-issued keys on the API keys page. HTTPS redirects match exactly. HTTP is allowed only for
loopback hosts; a native app may choose a different port on the registered
loopback host and path, as RFC 8252 requires. Wildcards, query strings, fragments
and embedded credentials are not accepted in registrations.

## Authorize

Generate a random PKCE verifier and state in the client. Keep both in short-lived
client storage that survives the authorization redirect; browser clients should
use a tab-scoped store.
Open `/api/v1/oauth/apps/authorize` on the gateway with these query parameters:
`client_id`, `redirect_uri`, `response_type=code`, `state`,
`code_challenge` and `code_challenge_method=S256`.

The authorization endpoint uses the deployment's `ui_base_url` (falling back to
`public_base_url`), so a separately hosted dashboard works too. The dashboard
signs in the user if necessary, displays the registered app name
and destination, and asks for a workspace. Approval posts to
`/api/v1/oauth/apps/authorize`. The user must have contributor access to that
workspace in their active organization. Cancel returns `error=access_denied`
and the original state. Invalid clients and redirects are never followed.

The callback receives `code` and `state`. Validate state before exchanging the
code. POST `application/x-www-form-urlencoded` to `/api/v1/oauth/apps/token`:

```text
grant_type=authorization_code&client_id=thunderbolt&redirect_uri=https%3A%2F%2Fyour-thunderbolt.example%2Foauth%2Fotari&code=the-returned-code&code_verifier=the-original-verifier
```

The JSON response contains `access_token` and `token_type: Bearer`. Refusals use
OAuth `error` and `error_description` fields; token responses must not be cached.
Use the token against Otari's OpenAI-compatible API. The code expires after two minutes, is bound to
the exact redirect and client, and can be exchanged only once. No client secret
is required. This flow issues an API key, with explicit revocation rather than
refresh tokens. The key is minted only when the code is exchanged, after checking
the user still has access. Abandoned approvals create no API keys; expired
approvals are pruned on subsequent approvals.

Browser clients also need their origin in `cors_allow_origins`. Tokens must be
kept in the application's credential store, never URLs or logs.

## Discover direct web tools

Call `/api/v1/tools` with the issued token before offering direct search or fetch.
For `otari_web_search` and `otari_web_fetch`, `direct_available: true` means the
default `/api/v1/search` or `/api/v1/fetch` endpoint is configured and allowed by
the workspace and key's effective tool grants. Organization search keys work
without a deployment-level search instance. The existing `available` field
describes tools used inside model requests, which can have legacy backends that
do not support direct calls. If `direct_available` is false or absent, let users
configure a separate search provider instead of assuming the direct endpoint works.
