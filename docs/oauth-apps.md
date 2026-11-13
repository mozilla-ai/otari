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

Each key an app is issued holds the [scopes](api-keys.md#scopes) its
registration names under `scopes`, and `inference:run` alone when it names none, so
an app calls models and nothing else unless you say so. An app key is never a
service key, so it cannot hold the `end-users` scopes.

Restart the gateway after changing registration. Empty `oauth_apps` disables
these endpoints. Removing a registration prevents new approvals and exchanges;
revoke already-issued keys on the API keys page. HTTPS redirects match exactly. HTTP is allowed only for
loopback hosts; a native app may choose a different port on the registered
loopback host and path, as RFC 8252 requires. Wildcards, query strings, fragments
and embedded credentials are not accepted in registrations.

## Authorize

Generate a random PKCE verifier of 43–128 characters and a separate random state. Compute
the S256 challenge as `BASE64URL(SHA256(ASCII(verifier)))`, omitting base64 padding.
Keep the verifier and state in short-lived client storage that survives the
authorization redirect; browser clients should
use a tab-scoped store.
Use the deployment's configured API base URL rather than hardcoding the versioned `/api/v1` prefix;
OAuth discovery metadata is not currently published.
Open `/api/v1/oauth/apps/authorize` on the gateway with these query parameters:
`client_id`, `redirect_uri`, `response_type=code`, `state`,
`code_challenge` and `code_challenge_method=S256`. Other parameters, such as
`scope`, are ignored: the key carries the user's own access in the approved
workspace, narrowed to the scopes the registration names.

The authorization endpoint uses the deployment's `ui_base_url` (falling back to
`public_base_url`), so a separately hosted dashboard works too. The dashboard
signs in the user if necessary, displays the registered app name
and destination, and asks for a workspace. Approval posts to
`/api/v1/oauth/apps/authorize`. The user must have contributor access to that
workspace in their active organization. Cancel returns `error=access_denied`
and the original state and issuer. Invalid clients and redirects are never followed.

The callback receives `code`, `state`, and `iss`. Validate state and require `iss`
to equal the authorization API origin (scheme, host and port, without a trailing slash)
before exchanging the code. This is the authorization host, which may differ from
the inference or dashboard host. Reverse proxies must preserve the external request
origin when forwarding requests. POST `application/x-www-form-urlencoded` to `/api/v1/oauth/apps/token`:

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

Browser clients also need their origin in `cors_allow_origins` for token exchange.
That does not let them approve grants: consent requires a request from the
dashboard origin configured by `ui_base_url` or `public_base_url`, or a same-origin
request to the gateway dashboard. Tokens must be
kept in the application's credential store, never URLs or logs.

## Disconnect

POST `application/x-www-form-urlencoded` to `/api/v1/oauth/apps/revoke` with
`token=<access_token>`. `token_type_hint=access_token` is optional. No client secret,
master key, or session cookie is required. Only the presented API key is revoked;
its usage history remains. Any ordinary API key can also revoke itself by
presenting its secret; this endpoint never grants authority over another key or
the deployment master key. Unknown or already revoked tokens return an empty
HTTP 200 response, as RFC 7009 requires. Discard the local credential after success.
The API keys page can also revoke a credential if the app is no longer available.
