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

Each key an app is issued holds the [scopes](https://docs.otari.ai/guides/control/api-keys#scopes) its
registration named when the person approved it, and the [default scopes](https://docs.otari.ai/guides/control/api-keys#scopes) when it
names none, so an app calls models and tools and nothing else unless you say so. An app key is never a
service key, so it cannot hold the `end-users` scopes.

Restart the gateway after changing registration. Empty `oauth_apps` disables
these endpoints unless `oauth_open_apps` is on. Removing a registration prevents new approvals and exchanges;
revoke already-issued keys on the API keys page. With `oauth_open_apps` on, a removed
client ID that is its callback's origin, such as `https://app.example`, still
connects as an [unregistered app](#let-unregistered-apps-connect), from any path on
that origin and marked as unverified. HTTPS redirects match exactly. HTTP is allowed only for
loopback hosts; a native app may choose a different port on the registered
loopback host and path, as RFC 8252 requires. Wildcards, query strings, fragments
and embedded credentials are not accepted in registrations.

## Let unregistered apps connect

Set `oauth_open_apps: true` to let any site connect without a registration, so
a static site or a local prototype can offer a sign-in button without asking
you first. An unregistered app sends its callback's origin as `client_id`,
written as a browser writes `location.origin`: `https://someone.github.io`, or
`http://localhost:8080` for loopback. Any path on that origin is accepted as
`redirect_uri`, and nothing off it. HTTP is again allowed only for loopback
hosts, and an origin longer than 128 characters cannot be a client ID. The origin
must be exactly what a browser writes: lowercase, with no default port, and with an
international host in its ASCII `xn--` form, which is how the consent screen shows it,
so a lookalike of another site's name is shown as the different name it is.

The consent screen names the site and says Otari has not verified it. The name
is only the site's host, so on a host that serves many people's pages by path, it
says nothing about whose page asked. The key
holds the [default scopes](https://docs.otari.ai/guides/control/api-keys#scopes) and appears on the API keys page
as `<site> (OAuth)`, for example `someone.github.io (OAuth)`. A registered
client ID keeps its own registration, so turning this on changes nothing for
registered apps. The rest of the flow is the same as for a registered app.

Any site can then ask your users for a key, and a careless approval hands one to
a site they should not trust. The key is still bound by the person's budgets and
workspace access, and they can revoke it on the API keys page. Leave this off
unless your users are expected to connect apps you have not reviewed.

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
workspace, narrowed to the scopes the consent screen showed that the
registration still named when the person approved. The token response names
them in its own `scope` member. An operator who narrows a registration while
somebody is reading the screen narrows their key; one who widens it does not
widen theirs.

Two optional parameters suggest limits for the key, with the names OpenRouter's
sign-in uses. `limit` is a spending limit in USD, a positive amount under
1,000,000 with at most two decimals, such as `25` or `12.50`. `expiration` is one
of `1-hour`, `1-day`, `7-days`, `30-days`, `90-days`, `180-days`, `1-year` or
`never`. The consent screen starts from them and the person can change either,
so treat them as suggestions. A value the screen cannot use, including one longer
than 32 characters for `limit` or 16 for `expiration`, starts it at no limit or
`never`, which are also the defaults. An approved limit caps the key over its
whole life through a budget of its own, named like the key and listed with the
organization's budgets, and an approved expiry counts from the moment the key is
issued. Where `budget_strategy` is `disabled`, no budget binds a key, so the
consent screen offers no limit and a suggested one is dropped.

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
the inference or dashboard host. Otari takes it from `public_base_url` when that is
set, and from the request otherwise, so set `public_base_url` behind a proxy that
rewrites the host or terminates TLS. POST `application/x-www-form-urlencoded` to `/api/v1/oauth/apps/token`:

```text
grant_type=authorization_code&client_id=thunderbolt&redirect_uri=https%3A%2F%2Fyour-thunderbolt.example%2Foauth%2Fotari&code=the-returned-code&code_verifier=the-original-verifier
```

The JSON response contains `access_token`, `token_type: Bearer`, `scope`, and
`manage_url`, the absolute URL of the dashboard page where the person sees the key
and revokes it, so an app can link there. Refusals use
OAuth `error` and `error_description` fields; token responses must not be cached.
Use the token against Otari's OpenAI-compatible API. The code expires after two minutes, is bound to
the exact redirect and client, and can be exchanged only once. No client secret
is required. This flow issues an API key, with explicit revocation rather than
refresh tokens. The key is minted only when the code is exchanged, after checking
the user still has access. Abandoned approvals create no API keys; expired
approvals are pruned on subsequent approvals.

Browser clients also need their origin in `cors_inference_origins` for token
exchange and revocation, which also opens the model and web tool endpoints the
key is for. Never put an app's origin in `cors_allow_origins`: that list sends
credentials. Neither lets an app approve grants: consent requires a request from the
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
