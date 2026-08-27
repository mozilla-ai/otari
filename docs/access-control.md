# Access control: users, keys, and budgets

Standalone Otari decides three things about every request: who is calling (the **user**), what credential they presented (the **API key**), and whether they have room to spend (the **budget**). This guide is a task-oriented tour of those three, with the management endpoints that drive them. Everything here is standalone-only; hybrid mode delegates identity and spend to otari.ai.

The user, key, and budget endpoints in this guide all require the master key (some other management endpoints, such as read-only pricing lookups, also accept a regular API key; see the [API reference](api-reference.md)). Send the master key as `Otari-Key: <master-key>` or `Authorization: Bearer <master-key>`. Most of these actions have a dashboard equivalent; see the [Admin dashboard guide](dashboard.md). Creating and deleting a user is the exception: the dashboard mints one when a member is added or a key is issued, and has no page for it, so the endpoints below are the way to do it directly.

**The master key authenticates this API. It is not, for long, how you sign in to the dashboard.** It bootstraps a new deployment and stays the deployment-wide API credential, which is what every script, CI job, and `curl` example here uses. The browser sign-in moves to an email address and a password once an operator claims the deployment, and [Dashboard sessions and identity](#dashboard-sessions-and-identity) below is where that happens. Retiring a login is not retiring a credential: nothing in this guide stops working when it does.

## How the pieces fit

- A **user** is the identity that spend and usage attach to. A user carries an optional default model allow-list and an optional budget.
- An **API key** is a credential a client sends to Otari. Each key belongs to a user. A key can narrow which models it may call.
- A **budget** is a spending limit with an optional reset period. It is a per-user cap; assign it to one user or share it across many.

A request is authenticated to a key, the key resolves to its user, the user's budget is checked and reserved before the provider call, and the usage is billed to that user afterward.

## Users

Create a user, optionally with a default model allow-list and a budget:

```bash
curl -X POST http://localhost:8000/v1/users \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "alice",
    "alias": "Alice (research)",
    "budget_id": "<budget-id>",
    "allowed_models": ["openai:gpt-4o-mini", "anthropic:*"]
  }'
```

- `user_id` is the stable identifier you choose; it is what spend and usage key on.
- `allowed_models` is the default access-list the user's keys inherit. `null` (or omitted) means any model, `[]` denies everything, and a list restricts to canonical `instance:model` entries, with `instance:*` and `instance:prefix*` wildcards. The list also gates the non-model spend surfaces that key on the same `instance:model` form, so a restricted key that should be able to call [`POST /v1/search`](api-reference.md#search) needs its search tool named too, as `<provider>:<tool>` (for example `exa:exa-search`).
- `blocked: true` stops the user from making requests without deleting anything; their calls are rejected until you unblock them.

Manage users with `GET /v1/users`, `GET /v1/users/{user_id}`, `PATCH /v1/users/{user_id}` (update alias, budget, `blocked`, or `allowed_models`), and `DELETE /v1/users/{user_id}`. A user's response includes `spend` and `reserved` (in-flight spend held by accepted but not-yet-settled requests); the committed total is `spend + reserved`. `GET /v1/users/{user_id}/usage` returns that user's request log.

### The default user

A key created with no `user_id` is bound to a shared user called `default`, created on first use. All such keys share one identity, so they share budget, usage, and files. Give a key an explicit `user_id` whenever you want to track or cap it separately.

## API keys

Create a key for a user. The plaintext key (a `gw-...` value) is returned once and never again; store it immediately.

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "alice-laptop",
    "user_id": "alice",
    "expires_at": "2026-12-31T23:59:59Z",
    "allowed_models": ["openai:gpt-4o-mini"]
  }'
```

- `expires_at` is an optional expiry; omit it for a key that never expires. Expired keys are rejected.
- `allowed_models` restricts this one key. The resolution is an override, not an intersection: a key's own list wins when set, a key with no list of its own inherits its user's default, and no list anywhere means unrestricted. A key can only narrow, never widen: creating or updating a key with a list broader than its user's default is rejected with `400`.
- The listing endpoints never return the plaintext again; they show only a `key_prefix` fingerprint (the key's leading characters).

Manage keys with `GET /v1/keys`, `GET /v1/keys/{key_id}`, `PATCH /v1/keys/{key_id}` (rename, toggle `is_active`, change expiry or `allowed_models`), and `DELETE /v1/keys/{key_id}`. To replace a key's secret without changing its identity or settings, use `POST /v1/keys/{key_id}/rotate`; it returns a new plaintext once and invalidates the old secret.

### Requests that name another user

By default a non-master key that names a `user` other than its own in the request body is rejected with `403`. This is the `reject_user_mismatch` setting (default `true`), and it applies to every non-master key on the deployment.

A key can override that default in either direction with its own `reject_user_mismatch`, set at create time or via `PATCH /v1/keys/{key_id}`: `null` (the default) inherits the deployment setting, `false` always accepts a mismatched `user`, `true` always rejects one. Use `false` on the one key that needs it (for example Claude Code, whose `metadata.user_id` is a telemetry blob rather than a user id) and leave the deployment strict; on a deployment that has already relaxed the setting globally, use `true` to keep an individual key strict.

Either way spend stays bound to the key's own user and the client value is forwarded to the provider as an end-user tag only, so leniency never lets a key charge another user. The master key may always bill an arbitrary user.

## Budgets

A budget is a spending cap with an optional reset period, and the only place in Otari that maps a cap to an amount. Everything that enforces a limit names a budget rather than restating the figure, so editing one moves every place it applies.

How it is enforced depends on what names it. Assigned to a user (`users.budget_id`), `max_budget` is the limit **per user**, so a budget shared by several users caps each of them at that amount rather than in aggregate. Named by a scoped ceiling ([Organizations and workspaces](#organizations-and-workspaces)), it is a single allowance everyone under that scope draws on together.

```bash
curl -X POST http://localhost:8000/v1/budgets \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "daily-10",
    "max_budget": 10.0,
    "budget_duration_sec": 86400
  }'
```

- `max_budget` is the per-user ceiling in your pricing currency. Otari reserves an estimated cost before each call and reconciles it after, so a request that would exceed the cap is rejected before it runs.
- `budget_duration_sec` is a rolling reset period in seconds (for example `86400` for daily, `604800` for weekly), counted from the last reset. On each period boundary the user's spend rolls back to zero and a reset is recorded.
- `reset_alignment` is the other way to say a period, snapping the window to a UTC calendar boundary: `calendar_day`, `calendar_week`, or `calendar_month`. It is the only way to express a calendar month, since 2592000 seconds is a different and slightly more generous product. Mutually exclusive with `budget_duration_sec`; sending both is refused with a 400.
- Omit both for a cap that never resets.

Assign a budget to a user by setting `budget_id` on the user (at create time or via `PATCH /v1/users/{user_id}`); sending an explicit `null` detaches it. Manage budgets with `GET /v1/budgets`, `GET /v1/budgets/{budget_id}`, `PATCH /v1/budgets/{budget_id}`, and `DELETE /v1/budgets/{budget_id}`. A delete is refused with a 409 while a workspace hands the budget to its members or a scoped ceiling enforces it; the message names which, and where to change it. A budget's response rolls up the users assigned to it: `user_count`, `total_spend`, and `total_reserved`. `GET /v1/budgets/{budget_id}/reset-logs` returns the per-user reset history.

The enforcement strategy is configurable with `OTARI_BUDGET_STRATEGY` (`for_update` row-lock, `cas` compare-and-swap, or `disabled`); see [Configuration](configuration.md).

## Organizations and workspaces

Otari is growing a tenancy layer above the users, keys, and budgets described here: an **organization** owns **workspaces**, and identities join both as members with a fixed role (`owner`, `admin`, `member`, or `viewer`). It is available over the API (`/v1/organizations/*` and `/v1/workspaces/*`, master-key authenticated like the rest of this guide) and in the dashboard, under Organization in the sidebar; see [Admin dashboard](dashboard.md#organization).

A self-hosted deployment is **one organization with several people in it**: the organization is provisioned for you, and workspaces are the unit you separate teams and projects by. That is the shape almost every deployment keeps, and hosting mutually isolated tenants on one deployment is still what a hosted control plane is for.

A second organization is possible, though, because it is already reachable: invite an address that belongs to an organization elsewhere on this deployment and they end up in two. So three endpoints exist for it, all of them scoped to the caller:

- `POST /v1/organizations` creates one, with the caller as its owner and a default workspace to work in. Only a name is sent; the slug is derived from it with a random suffix, so two organizations may share a name and a later rename does not move the slug. It does **not** move the caller into the new organization.
- `GET /v1/organizations/me/memberships` lists the organizations the caller is an active member of, with their role in each and which one is current. It is the caller's own memberships, not a directory of the deployment's organizations.
- `POST /v1/organizations/me/switch` points the caller's identity at another organization they belong to. Everything scoped follows it: workspaces, keys, budgets and usage all resolve through that pointer. An organization the caller holds no active membership in answers `404`, whether or not it exists.

Switching is not renaming: `PATCH /v1/organizations/me` renames the organization already active. And there is no delete: every historical attribution resolves through rows that hang off an organization.

Nothing is required to set it up. The first request to one of those endpoints provisions a default organization, a default workspace, and one owner identity representing the operator, and every later request resolves that same identity. Organization owners and admins can create further workspaces, add members, and manage roles; a workspace's own owners and admins can manage the workspace they belong to.

Adding a member takes an email address (`POST /v1/organizations/me/members`), optionally with the workspaces to grant at the same time. If no identity holds that address yet, one is created carrying it, and the member is active immediately with nothing emailed. Such an identity is a roster and attribution entry until it signs up: it carries no password, and the address is the handle [signup](#signup-claiming-a-roster-identity) matches it on to give it one.

### Invitations

`POST /v1/organizations/me/member-invitations` is the other way to add someone: the membership lands `invited` rather than `active`, and an email with an accept link goes out if mail is configured (see [Configuration](configuration.md)). If it isn't, the response still carries the link (`accept_link`) so the operator can share it another way; `mail_sent` says whether it was actually emailed.

The recipient follows the link to a public accept page (`POST /v1/invitations/validate` to preview it, `POST /v1/invitations/accept` to commit, both with the token in the body rather than the URL), which resolves the membership to `active` and grants any workspaces parked on the invitation. No session is minted, and accepting sets no password: the identity it resolves to is in the same state as an address added directly, so it signs in once it [signs up](#signup-claiming-a-roster-identity) the same way.

An invitation expires after `invitation_expiry_hours` (default 7 days) and can be revoked before it is accepted (`DELETE /v1/organizations/me/member-invitations/{invitation_id}`), which cancels it and suspends the membership, the same as removing a member. Re-inviting the same address revives it.

Three rules exist to stop a tenancy from becoming unmanageable or from losing data:

- an organization always keeps at least one active owner, so the last owner cannot be demoted or removed (`400`);
- an organization always keeps at least one workspace, so the last one cannot be deleted (`400`);
- a workspace that still holds API keys, usage, aliases, or routing policies cannot be deleted (`409`). Move or delete what it holds first.

Removing a member suspends their membership rather than deleting it, which keeps their past usage attributable.

Granting the `owner` role is an owner's to give. An admin manages members, workspaces and roles, and cannot promote anyone (themselves included) to owner, nor add one.

### What a role reads

An organization role decides how much of that organization a member sees, and nothing beyond it. `owner` and `admin` manage the organization and read every workspace in it; `member` and `viewer` read the workspaces they belong to. That one rule covers the workspace list, every workspace-scoped read, and the organization's usage (`/v1/organizations/me/usage{,/count,/summary,/series}`, which resolves its scope from the caller's own membership and answers `404` for a workspace outside it).

No organization role reaches the deployment's own surfaces. `/v1/usage` reads every tenant, and it, `/v1/keys`, `/v1/users`, `/v1/settings`, `/v1/provider-credentials` and the rest of the deployment-wide plane need operator authority, which is `is_superuser` or the bootstrap identity and is granted from Platform Admin rather than from a roster. Promoting somebody to organization admin does not confer it, and is not meant to.

### Deployment-wide account administration

Everything above stops at an organization's boundary: the roster lists that organization's members, and a membership suspended everywhere leaves the roster with nowhere else to be found. `/v1/admin` is the surface that does not, and it is for whoever operates the deployment rather than for a tenant.

```bash
# Every account on the deployment, with the organizations each belongs to and
# when it last signed in to the dashboard.
curl -H "Otari-Key: $OTARI_MASTER_KEY" http://localhost:8000/v1/admin/users

# Deactivate one. Its dashboard sessions end immediately. Memberships, keys and
# usage history are untouched, so any API key this account minted keeps
# authenticating until it is revoked on its own (`PATCH /v1/keys/{key_id}`):
# deactivating closes the dashboard, not the API.
curl -X PATCH -H "Otari-Key: $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"is_active": false}' \
  http://localhost:8000/v1/admin/users/$USER_ID

# Grant or remove operator access, which is what reaches this surface.
curl -X PATCH -H "Otari-Key: $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"is_superuser": true}' \
  http://localhost:8000/v1/admin/users/$USER_ID
```

`last_sign_in_at` is null for an account that has never signed in to the dashboard, and it goes on being null: it is a stored stamp rather than a reading of the live session table, so it does not turn back into "never" when a session expires.

**Who may use it.** A superuser, or the identity the `tenancy_bootstrap_user_id` marker names, which the master key resolves to. Anyone else gets `404` rather than `403`, so the surface does not confirm it exists to a caller who may not use it, and the dashboard does not read the refusal as an expired session. The marker arm is what keeps the deployment reachable if a superuser flag is cleared by hand.

**Two changes are refused, in one direction each.** An operator cannot deactivate their own account or drop their own operator access: deactivating ends the session they are holding, and dropping the flag takes away the page they would undo it from; and neither can be taken from the bootstrap operator, which is the identity master-key sign-in resolves to, so deactivating it would turn the fallback credential into a session that dies on arrival. Granting either back is not refused, which is what makes a cleared flag repairable here.

Creating an account is not part of this surface: an account with no membership can do nothing, and memberships are the organization surface above. Neither is deleting one, for the reason removing a member suspends rather than deletes: past usage stays attributable.

The dashboard renders it at **Accounts**, on the organization rail beside Settings, and the row is absent for a caller the surface would refuse.

### Dashboard sessions and identity

Signing in to the dashboard exchanges a credential for a session: an opaque token in an HttpOnly cookie, stored only as a SHA-256 hash, revocable server-side, expiring on `dashboard_session_ttl_hours`. Each session names the identity it was minted for, so a cookie-authenticated request resolves a user and, through that user's `active_organization_id`, the organization it is acting in. `POST /v1/auth/session` returns both ids alongside the expiry.

**Which credential signs in depends on whether the deployment has been claimed.** A deployment whose operator identity has no password yet accepts the master key, which is what makes first boot work with nothing configured. The moment that identity has one, the master key stops being accepted at the sign-in endpoint and email and password is the login. It is the operator's own password that decides this and nobody else's: a member who [signs up](#signup-claiming-a-roster-identity) or resets theirs claims their account, not the deployment, and an identity that arrived from a migration carrying a password has claimed nothing at all. `GET /v1/bootstrap` publishes which of the two applies right now, in `sign_in_methods`, so the sign-in page asks for the credential that will work:

```bash
curl http://localhost:8000/v1/bootstrap        # no credential needed
# {"deployment_type":"standalone", ..., "sign_in_methods":["master_key"]}
```

Nothing schedules the switch and nothing expires. A deployment that never claims goes on signing in with the master key indefinitely, which is a reasonable end state for a single-operator gateway that only ever talks to itself. One consequence is worth knowing, and it has two halves. While a deployment is unclaimed, the sign-in screen offers the master key and not the password form, so a member who has signed up there signs in by calling `POST /v1/auth/session` directly until the operator claims it; and if they do reach the dashboard, **Account settings** offers them the claim form rather than the change-password form, which the endpoint then refuses for the missing `current_password`. Both follow from the same thing: `sign_in_methods` describes the deployment, and the dashboard has no route yet for asking what the signed-in identity itself holds. Claiming before you add anyone avoids both.

#### Claiming the deployment

First boot provisions the operator identity as a label with no address and no password, so claiming supplies both. The dashboard offers it at **Account settings**, reached from the account control at the foot of the sidebar, which is one call authenticated by the session the master key already minted. The same call over HTTP, authenticated by the master key itself:

```bash
curl -X PUT http://localhost:8000/v1/auth/password \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "new_password": "<a password>"}'
```

- A password is at least 8 characters and at most 72 bytes, which is bcrypt's own ceiling; accented and non-Latin characters count for more than one byte each. There are no composition rules.
- The address is stored lower-cased and matched case-insensitively at sign-in. Claiming through the master key does not deliver to it or verify it: the master key is what proves the claim, and `email_verified_at` is stamped as a consequence rather than checked. A roster member proves their own address by [signing up](#signup-claiming-a-roster-identity) instead, which does send and check a verification link.
- The same endpoint changes a password later, and the same **Account settings** page is where the dashboard does it. From a session it needs `current_password`; sent with the master key in a header it does not, which is the recovery path (see below) and the one form of this call the dashboard cannot make, since reaching the page needs a session. Changing the address afterwards is refused rather than half-supported.
- Every other session that identity holds is revoked, so a cookie minted under the old password does not outlive it. The caller's own session is spared when the change came from the browser, and is not when it came from the master key, since a header caller has no session to keep.
- **Claiming is one-way.** No endpoint clears a password, so a deployment whose operator has set one cannot be returned to master-key sign-in. The sign-in screen follows `sign_in_methods` and asks for the address and password from then on, and the master key still authenticates the whole management API, so the step costs you nothing you cannot reach; it is simply not reversible. See the [Admin dashboard guide](dashboard.md).

After that, sign in with the address:

```bash
curl -X POST http://localhost:8000/v1/auth/session \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "<the password>"}'
```

A failed sign-in answers `401 Incorrect email or password` whichever part was wrong, and takes the same time to do it, so the endpoint cannot be used to find out which addresses hold an account. A master key presented after the deployment is claimed answers `403` naming the password login, which is a different answer from a wrong master key's `401` on purpose: the credential is fine, that use of it is over.

#### Freezing sign-ins for a redeploy

An operator can stop the gateway issuing new dashboard sessions while it is being updated, so nobody signs in mid-migration:

```bash
curl -X PATCH http://localhost:8000/v1/settings/maintenance-mode \
  -H "Otari-Key: $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

`POST /v1/auth/session` then answers `503` for every credential, including the master key on a deployment still using it to sign in, and `GET /v1/bootstrap` reports `maintenance_mode: true` so the sign-in screen says what is happening rather than presenting a form that can only be refused. Three things it deliberately does not do: it does not revoke sessions already issued, so the operator who set it stays signed in; it does not touch the data plane or the rest of the management API, so API keys and completions carry on serving; and it does not exempt any identity, because it does not need to. The switch is master-key gated and reachable through the header, which never passes through the door the freeze closes, so the way back out does not depend on signing in. `PATCH` it back to `false`, or read the current state with `GET /v1/settings/maintenance-mode`.

**Keep your master key to hand before you set this.** The header is what lifts the freeze from a browser holding no session, so an operator on a claimed deployment who no longer has the generated key, has signed out, and has frozen sign-ins has no route back in through the app. It is recoverable, by setting `OTARI_MASTER_KEY` and restarting, which is within reach of anyone already redeploying, but it is a restart rather than a click.

It is stored rather than held in memory, so a deployment running several replicas freezes all of them from one call, and the freeze survives a restart.

#### What the master key still does

Everything else. It authenticates `/v1/keys`, `/v1/users`, `/v1/budgets`, and the rest of this guide exactly as before, in the `Otari-Key` or `Authorization` header. Claiming a deployment changes one endpoint's answer and no others.

It also stays the way back in. An operator who forgets the password sets a new one through the same `PUT /v1/auth/password`, with the master key in a header and no `current_password`. That is deliberate rather than a gap: a caller holding the master key can already do anything the management API can do, so asking them for a password they have lost would lock the dashboard while leaving the API wide open. Password recovery for someone who holds no master key needs mail; see [Password reset](#password-reset) below.

#### Signup: claiming a roster identity

An identity an admin added or invited by address carries no password until it signs up. `POST /v1/auth/signup` claims it: it sets a password on the identity that address already names and sends a verification link. It never creates an identity from nothing (self-service registration for a wholly new address is not part of this edition), and it is enumeration-safe about that: an address nobody has touched, one that already completed signup, and one whose identity has been deactivated all answer with the same generic message and nothing written or mailed, the same shape resending and requesting a reset already take below. Only the password itself is judged and reported on its own terms (too short, too long), since that says nothing about whether the address exists.

```bash
curl -X POST http://localhost:8000/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email": "erin@example.com", "password": "<a password>"}'
```

A genuinely pending identity exists and has a password from this call on, but it is **hard-blocked from signing in until it verifies**, with no time limit on how long it may wait to: `POST /v1/auth/session` answers `403` for the right password on an unverified address, naming the reason rather than folding it into the generic `401`, because by then the password has already proven the caller holds the account. The verification link itself does expire (`email_verification_expiry_hours`, default 48), and can be requested again at any time:

```bash
curl -X POST http://localhost:8000/v1/auth/verify-email \
  -H "Content-Type: application/json" \
  -d '{"token": "<the token from the link>"}'

curl -X POST http://localhost:8000/v1/auth/resend-verification \
  -H "Content-Type: application/json" \
  -d '{"email": "erin@example.com"}'
```

The dashboard offers the whole of this without `curl`: **Added to this gateway? Claim your account** on the sign-in screen is the signup form, the verification link lands on a page that confirms the address on arrival, and **Need a new verification link?** resends. Those links are absent on a deployment that cannot send mail, since none of the three routes can work there.

A verification token is single-use: presenting it again, or presenting an unknown, expired, or deactivated identity's token, answers `400` without saying which is true. Resending answers the same message whether the address is unregistered, already verified, or genuinely waiting, and only the last case actually sends anything, so the endpoint cannot be used to learn which addresses exist. All three routes share the sign-in endpoint's rate limiter (`dashboard_login_rate_limit_per_minute`) and answer `503` naming what is missing when this deployment cannot send mail (`GET /v1/bootstrap`'s `mail_ready` says so in advance).

#### Password reset

`POST /v1/auth/password/reset` mails a reset link to an address that has a password, whether or not that address has verified yet (forgetting a password predates ever confirming it). `POST /v1/auth/password/reset/confirm` completes it with the token and a new password:

```bash
curl -X POST http://localhost:8000/v1/auth/password/reset \
  -H "Content-Type: application/json" \
  -d '{"email": "erin@example.com"}'

curl -X POST http://localhost:8000/v1/auth/password/reset/confirm \
  -H "Content-Type: application/json" \
  -d '{"token": "<the token from the link>", "new_password": "<a new password>"}'
```

**Forgot your password?** on the dashboard's sign-in screen is the same pair of calls, and the reset link lands on a page that asks for the new password. The request answers the same message whether or not the address holds a password, for the same enumeration-safety reason resending a verification link does. The reset token expires (`password_reset_expiry_hours`, default 2) and is single-use: unlike a stateless token, it is cleared the moment it is spent, so it cannot be replayed even inside its own expiry window, including by a second request racing the first: the identity row is locked before the clear, so only one concurrent redemption spends it. That last part needs PostgreSQL. SQLite, the default for a single-node deployment, has no row locks, so two redemptions of one token arriving together can both land there. It is also cleared the moment the identity's password changes through any other channel (an ordinary self-service change, an operator recovery through the master key) while it is still live, so a reset link generated and then overtaken elsewhere cannot undo that change later. Completing a reset revokes every other session the identity holds, the same as an ordinary password change. Both routes share the sign-in rate limiter and the same `503`-when-unconfigured behavior signup does.

#### Passkeys

A passkey is a key pair whose private half never leaves the authenticator (a laptop's secure enclave, a phone, a hardware key). Registering one stores its public half in `webauthn_credential`; signing in has the authenticator sign a server-chosen challenge. Nothing this deployment stores can be used to sign in as anybody, which is the difference from a password hash.

Passkeys are **additive**. They do not replace the master key or the password: `GET /v1/bootstrap` reports `passkey` in `sign_in_methods` alongside whichever of those two the deployment currently takes, and only once the deployment is configured for WebAuthn and holds at least one passkey its current relying-party ID can assert. A passkey also never creates an identity: registering one is done from inside a session, so a deployment is still claimed with the master key and a password, and a passkey joins an identity afterwards.

```bash
# Registering, from inside a session (or with the master key, as here).
curl -X POST http://localhost:8000/v1/auth/webauthn/register/options -H "Otari-Key: Bearer $OTARI_MASTER_KEY"
curl -X POST http://localhost:8000/v1/auth/webauthn/register \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"credential": {...the browser's response...}, "name": "Work laptop"}'

# Signing in, unauthenticated.
curl -X POST http://localhost:8000/v1/auth/webauthn/authenticate/options
curl -X POST http://localhost:8000/v1/auth/webauthn/authenticate \
  -H "Content-Type: application/json" -d '{"credential": {...the assertion...}}'
```

Sign-in is **usernameless**: the options carry no `allowCredentials`, so the browser offers whichever passkey it holds for this relying party and the assertion names the credential that answered. That is what lets somebody sign in without typing anything, and it also keeps the endpoint from becoming an oracle for which addresses hold a passkey here. Registration therefore asks for a discoverable (resident) credential; an authenticator too old to store one is the case this does not serve.

Each ceremony is two calls joined by a single-use challenge held in `webauthn_challenge`. The row is deleted as it is spent, so a captured assertion cannot be replayed; a ceremony that fails verification rolls back instead, leaving the challenge for the retry. It is a table rather than a signed cookie because a challenge has to be retired server-side, and rather than process memory because a deployment runs more than one worker.

`GET /v1/auth/webauthn/credentials` lists the caller's own passkeys, `PATCH .../{id}` renames one, and `DELETE .../{id}` removes it. The list carries no key material. Deleting the last passkey is allowed: an email and password is still the deployment's login, so it is not a lockout, and refusing would strand whoever just lost the authenticator.

##### The relying-party ID

A passkey is scoped by the authenticator to a **relying-party ID**, a bare domain. A credential created under one ID cannot be asserted under another, and the key material never leaves the authenticator, so **changing the ID orphans every passkey already registered** and no data migration recovers them. The ID is therefore configured or derived once, never read off the request:

| Setting | Meaning |
| --- | --- |
| `webauthn_rp_id` | The ID itself, a bare domain with no scheme, port or path. Optional. |
| `public_base_url` | This deployment's own address. When `webauthn_rp_id` is unset, the ID is this URL's **host**, with the scheme and port dropped (`https://otari.example.com:8443` → `otari.example.com`). |
| `webauthn_allowed_origins` | Origins a ceremony may be performed from, each with a scheme. Defaults to `public_base_url` alone. |
| `webauthn_rp_name` | What an authenticator shows while creating a passkey. Cosmetic. |

A deployment with no address of its own has no relying party, offers no passkeys, and is not misconfigured; the ceremony endpoints answer `503` naming the setting to fill in, and `GET /v1/bootstrap` reports `passkeys_ready: false` so the dashboard does not offer a form that could only fail. **`public_base_url` is what turns passkeys on, and it is enough on its own.** `webauthn_rp_id` is an override rather than an alternative: it changes which domain a passkey is bound to and still needs an origin to serve the ceremony from, so setting it alone leaves the deployment with no relying party exactly as if neither were set. Startup says so rather than leaving it to be discovered from a `503`. Deriving the ID from the request's `Host` header instead is the shape this deliberately does not take: `Host` is attacker-controlled on any deployment that does not pin it, and the ID is the only thing scoping a passkey to this site.

Set `webauthn_rp_id` explicitly to bind passkeys to a **parent** domain of the one serving the dashboard: a passkey under `example.com` works on `otari.example.com`, but not the reverse. Every entry in `webauthn_allowed_origins` must be the ID or a subdomain of it, which is checked at startup rather than left to fail as an unexplained `SecurityError` in a browser. Each row records the `rp_id` it was registered under, so a credential the current configuration cannot assert is kept out of the ceremonies instead of being offered and then failing. It is **not** hidden from the credential list: it comes back with `is_usable: false`, because a passkey that silently vanished would leave nothing to explain the empty page and nothing to delete. Listing, renaming and deleting stay available on a deployment with no relying party at all, for the same reason.

Because a relying-party ID cannot move, one constraint outlives this document. [mozilla-ai/otari-ai#1716](https://github.com/mozilla-ai/otari-ai/issues/1716) settled that migrating otari.ai users **import their credentials rather than claiming new accounts**; an imported row's `rp_id` is `otari.ai`, so that import holds exactly while the hosted origin stays `otari.ai`. Moving it re-scopes every imported passkey and the people holding them have to register again.

#### OAuth sign-in (Google and GitHub)

Sign in with a Google or GitHub account instead of typing a credential. Off by default: a deployment that registers no OAuth client offers no OAuth affordance at all, and the sign-in screen has no dead buttons on it.

**It widens how a member signs in, never who may.** An OAuth identity signs in as an account an operator already put on the roster, matched on the address the provider vouches for. An address nobody added is refused rather than provisioned, which is the same rule signup already follows: enabling Google sign-in must not mean that every holder of a Google account can get into your gateway. The decision sits behind `IdentityProviderPort`, so an edition that wants to provision on first sight binds its own adapter and Otari's own is left alone.

Turning it on takes a registered OAuth client and three settings:

| Setting | Meaning |
| --- | --- |
| `oauth_google_client_id` / `oauth_google_client_secret` | The Google OAuth client. Both, or Google is not offered. |
| `oauth_github_client_id` / `oauth_github_client_secret` | The GitHub OAuth client. Both, or GitHub is not offered. |
| `public_base_url` | This deployment's own address. The redirect URI is derived from it, so without one neither provider is offered. |

Register the redirect URI with the provider as exactly `{public_base_url}/auth/{provider}/callback`, for example `https://otari.example.com/auth/google/callback`. It is **not** a dashboard hash path, and cannot be: a redirection URI may not carry a fragment ([RFC 6749 §3.1.2](https://www.rfc-editor.org/rfc/rfc6749#section-3.1.2), and Google rejects one outright). The gateway serves that plain path and redirects it into the dashboard page that finishes the sign-in. Otari derives both the URI it sends with the authorization request and the one it sends with the exchange from this single setting, so the two cannot disagree with each other; they can still disagree with what you registered, which is the usual cause of an `invalid_grant` from the provider.

`GET /v1/bootstrap` reports the configured providers in `oauth_providers`, and the sign-in screen renders one button per entry. Like passkeys this is **additive**: it sits beside whichever typed credential the deployment currently takes, and never replaces one.

```bash
# 1. Ask where to send the browser. The state comes back for the browser to keep.
curl http://localhost:8000/v1/auth/oauth/google/authorize

# 2. The person completes the consent screen and lands back on
#    /auth/google/callback?code=...&state=..., which redirects into the
#    dashboard. Once the dashboard has checked the state it spends the code:
curl -X POST http://localhost:8000/v1/auth/oauth/google/callback \
  -H "Content-Type: application/json" -d '{"code": "<the code>"}'
```

The exchange mints the same HttpOnly session cookie a password does, so nothing downstream of a sign-in behaves differently. Three refusals are worth knowing, and each says what to do: a provider that will not vouch for the address (`401`), an address no active identity here holds (`401`), and a provider this deployment did not configure (`503`, naming the settings). A deactivated identity is refused as unknown rather than told its account is switched off.

**A verified provider address lifts the local verification gate.** A member an operator added has never confirmed their address to this gateway, and the password login hard-blocks that. The provider's assertion is a stronger proof of the same fact, so an OAuth sign-in stamps the verification and lets them in. On a deployment with no outgoing mail, that is the only way a member can get in without an operator setting something up for them.

**PKCE is deliberately off**, and turning it on is a real change rather than a default to restore. Authorizing and calling back are two independent requests with nothing kept server-side between them, so a PKCE verifier minted while building the authorization URL would have nowhere to live until the exchange. The CSRF `state` survives that gap by living in the browser instead: the dashboard stores it in `sessionStorage` when it sends somebody to the provider and compares it when the provider sends them back, and a callback whose state does not match is abandoned without the code ever reaching this gateway. Enabling PKCE needs a shared server-side store first.

**`email_verified` is read strictly.** The provider reports it as three states, not two: vouched for, explicitly not, or never mentioned. Otari treats the third as unverified, so an address a provider merely returned is never laundered into a verified identity. ([mozilla-ai/otari-ai#1551](https://github.com/mozilla-ai/otari-ai/issues/1551) moves identity resolution onto that three-state model and onto keying by provider subject rather than by address.)

The protocol mechanics come from [apron-auth](https://pypi.org/project/apron-auth/), which owns the provider endpoints, the code exchange and the userinfo fetch. What Otari keeps is which providers are configured, which scopes are asked for (`openid email profile` for Google; `read:user user:email` for GitHub), and how a fetched identity maps onto an account here.

#### Who can sign in

An identity with a password can sign in once it has verified its address: the operator gets one by claiming the deployment (verified automatically, since the master key proved it), and a roster member gets one by signing up (verified by the link). There is still no way for an admin to set a password on somebody else's identity; a member added or invited by address holds a role and can be placed in workspaces, but only that address's own signup or reset gives it a way in. Passkeys and OAuth are described above, and neither is a way in of its own: a passkey is added to an identity that can already sign in, and an OAuth account signs in as a roster identity that already exists. So the roster is still the whole answer to who may sign in, whichever credential they use.

A session is revoked on sign-out, on a password change as described above, on master-key rotation (every session, with the rotating tab's own re-minted for the same identity), when the master key changes across a restart, and when the identity it names is deleted or deactivated. A deactivated identity also stops being able to sign in, rather than keeping access until its cookie expires. Deactivation is enforced when the session is read, and the identity's sessions are deleted at that point rather than only refused, so re-activating it later does not hand back the access of any cookie that was presented while it was off. Nothing sweeps the rest: a cookie that is never presented in that window survives to its TTL, because no flow here deactivates an identity and none therefore revokes ahead of the read.

An opaque session token is the settled shape here, not a stopgap: it is revocable, which a bearer JWT is not, and [mozilla-ai/otari-ai#1716](https://github.com/mozilla-ai/otari-ai/issues/1716) settled that sessions are the steady-state dashboard login. **The platform's JWT `Token` therefore does not survive the rehome.** Anything on the platform frontend that reads or stores a bearer token changes when its pages arrive: the credential is an HttpOnly cookie the page's own script cannot read, sent automatically and same-origin only, so there is no token to attach to a header and no expiry to decode out of a payload. Sign-in state comes from the session endpoint's response and from a 401 bounce, not from inspecting a token.

### Adopting an existing tenancy

Provisioning adopts an organization whose slug is `default`, which is the one it would have created itself. It cannot adopt any other, because every route is scoped to the organization the operator identity is currently pointed at. `POST /v1/organizations/me/switch` is no way in either: it refuses an organization the caller holds no active membership in, and a freshly provisioned operator identity holds none in an imported one. So an organization this deployment did not provision is unreachable through the API until the operator identity points at it.

That is the state a database restored or imported from elsewhere arrives in: those slugs are `{name}-{suffix}` and never the literal `default`. Otari refuses rather than shadowing it, and the tenancy endpoints answer `500` with `Internal server error` while the specific organizations are named in the gateway's log.

Two rows decide this, and both have to move. The marker is a `runtime_settings` row keyed `tenancy_bootstrap_user_id`, holding the id of the identity every request resolves to; it is deliberately not settable over the API, since repointing it changes who the operator *is*. The organization served is then that identity's own `active_organization_id`, **not** anything the marker says. An imported identity usually belongs to several organizations, and that pointer holds whichever one they last switched to on the platform, so setting the marker alone adopts whatever organization that happens to be:

```sql
-- 1. Find an active owner of the organization to adopt.
SELECT u.id, u.email, u.full_name, u.active_organization_id
FROM "user" u
JOIN organization_member om ON om.user_id = u.id
JOIN organization o ON o.id = om.organization_id
WHERE o.slug = 'acme-1a2b3c4d' AND om.role = 'owner' AND om.status = 'active';

-- 2. Point that identity at the organization to adopt. Without this the
--    gateway serves whichever organization the identity was last active in,
--    and reports the operator's role there rather than their ownership here.
UPDATE "user"
SET active_organization_id = (SELECT id FROM organization WHERE slug = 'acme-1a2b3c4d')
WHERE id = '<the id from step 1>';

-- 3. Point the marker at that identity. An upsert, not an UPDATE: a gateway
--    that refused rather than provisioning never wrote the marker row, so an
--    UPDATE would match nothing and the refusal would repeat unchanged.
--    updated_at is NOT NULL with no database-side default, so it is supplied.
INSERT INTO runtime_settings (key, value, updated_at)
VALUES ('tenancy_bootstrap_user_id', '<the id from step 1>', CURRENT_TIMESTAMP)
ON CONFLICT (key) DO UPDATE
SET value = excluded.value, updated_at = excluded.updated_at;
```

Confirm with `GET /v1/organizations/me`, which should name the adopted organization and report the role `owner`. No restart is needed: both rows are read from the database on each request.

One ordering is not caught. The refusal only runs while the marker is unresolved, so it covers importing into a deployment that has never served a tenancy request. Import *after* this gateway has provisioned its own default organization and nothing refuses: the marker already resolves, and the imported rows are silently unreachable. Repointing the marker is still the fix, and importing before the first tenancy request is what turns a silent case into a loud one.

### Workspace-scoped spend

API keys, usage rows, model aliases, and routing policies each belong to a workspace. A key's workspace is fixed when it is issued and read off the key on every request, never off a header, so a caller cannot bill another workspace. A master-key request has no key row and lands in the deployment's default workspace.

Budgets come in two kinds, both enforced, and a request has to pass both:

- **Per-user budgets** are the `budgets` table described above, attached with `users.budget_id`, and unchanged. One budget shared by several users is a limit each of them gets in full, not a pot they share.
- **Scoped budgets** (`/v1/scoped-budgets`) cap a tenancy scope instead: an organization, a workspace, a workspace membership, an organization membership, or a single API key. A ceiling optionally narrows to one provider, so "this workspace may spend $50 a month, of which no more than $10 at Anthropic" is two rows. Every ceiling that applies to a request must admit it; there is deliberately no rule that a workspace's ceilings must sum to less than its organization's, since the organization's already bounds the total.

A scoped ceiling's period comes from one of two fields, never both. `budget_duration_sec` is a rolling window of N seconds measured from the last reset, so a ceiling whose window runs out while nothing is being served restarts on the next request and its reset time walks forward from there. `reset_alignment` snaps the window to a UTC calendar boundary instead: `calendar_day` to midnight, `calendar_week` to Monday 00:00, `calendar_month` to the 1st. Because an aligned window is derived from the boundary rather than from the request that noticed the expiry, a ceiling nothing touched for two months rolls straight into the current month, dated the 1st, with fresh counters and no backfill of the months it slept through. A calendar month is the case seconds cannot express at all: 2592000 makes a year of 12.17 periods against 12 months, so a monthly cap written that way is about 1.5 percent more generous than it reads. A ceiling with neither field never resets, and one carrying both is refused.

Those boundaries are UTC, deliberately, so a deployment in Tokyo sees a daily ceiling reset at 09:00 local. A local calendar day is 23 or 25 hours across a DST transition, which is real complexity for a benefit nobody has asked for yet; a reset timezone stays additive if that changes.

A key marked `exclude_from_budget`, and `OTARI_BUDGET_STRATEGY=disabled`, bypass both kinds.

Three things scoped ceilings do not yet count. Externally recorded spend (`POST /v1/usage/external`) and imported usage have no request scope to resolve a workspace or provider from, so they settle against the per-user counters alone. A batch is the third: it is gated when it is created, but its real cost arrives later through the same external path, so it never reaches a ceiling either. A workspace's `current_spend` therefore will not match a `SUM(cost)` over its usage rows once imports or batches are in play.

One more limit is worth knowing before you narrow a ceiling to a provider. A request is admitted against the ceilings for the provider its routing policy names *first*. If that attempt fails and the policy falls over to another provider, the second provider's narrowed ceiling is neither checked nor charged, so a narrowed cap is only reliable on a provider that is the head of every policy that can reach it. Aggregate ceilings, the ones with no provider, are unaffected: they apply whichever provider serves.

Identities and the request plane are still two tables (`user` for tenancy, `users` for per-request spend), bridged by minting a `users` row named after a member's identity id. Converging them is tracked in [mozilla-ai/otari-ai#1727](https://github.com/mozilla-ai/otari-ai/issues/1727), under the wider reconciliation in [mozilla-ai/otari-ai#1452](https://github.com/mozilla-ai/otari-ai/issues/1452).

## See also

- [Admin dashboard](dashboard.md): the same keys and budgets in the browser UI, with a user shown as a member of the organization.
- [Configuration](configuration.md): `reject_user_mismatch`, `budget_strategy`, and related settings.
- [API reference](api-reference.md): the full endpoint and schema listing.
