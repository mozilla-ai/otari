# Access control: users, keys, and budgets

Standalone Otari decides three things about every request: who is calling (the **user**), what credential they presented (the **API key**), and whether they have room to spend (the **budget**). This guide is a task-oriented tour of those three, with the management endpoints that drive them. Everything here is standalone-only; hybrid mode delegates identity and spend to otari.ai.

The user, key, and budget endpoints in this guide all require the master key (some other management endpoints, such as read-only pricing lookups, also accept a regular API key; see the [API reference](api-reference.md)). Send the master key as `Otari-Key: <master-key>` or `Authorization: Bearer <master-key>`. The same actions are available in the dashboard's **Access** section; see the [Admin dashboard guide](dashboard.md).

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

A budget is a spending cap with an optional reset period. `max_budget` is the limit **per user**, so a budget shared by several users caps each of them at that amount rather than in aggregate.

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
- `budget_duration_sec` is the reset period in seconds (for example `86400` for daily, `604800` for weekly). Omit it for a cap that never resets. On each period boundary the user's spend rolls back to zero and a reset is recorded.

Assign a budget to a user by setting `budget_id` on the user (at create time or via `PATCH /v1/users/{user_id}`). Manage budgets with `GET /v1/budgets`, `GET /v1/budgets/{budget_id}`, `PATCH /v1/budgets/{budget_id}`, and `DELETE /v1/budgets/{budget_id}`. A budget's response rolls up the users assigned to it: `user_count`, `total_spend`, and `total_reserved`. `GET /v1/budgets/{budget_id}/reset-logs` returns the per-user reset history.

The enforcement strategy is configurable with `OTARI_BUDGET_STRATEGY` (`for_update` row-lock, `cas` compare-and-swap, or `disabled`); see [Configuration](configuration.md).

## Organizations and workspaces

Otari is growing a tenancy layer above the users, keys, and budgets described here: an **organization** owns **workspaces**, and identities join both as members with a fixed role (`owner`, `admin`, `member`, or `viewer`). It is available over the API (`/v1/organizations/*` and `/v1/workspaces/*`, master-key authenticated like the rest of this guide) and in the dashboard, under Organization in the sidebar; see [Admin dashboard](dashboard.md#organization).

A self-hosted deployment is **one organization with several people in it**, not several tenants: the organization is provisioned for you and cannot be created, switched or deleted, and workspaces are the unit you separate teams and projects by. Hosting mutually isolated tenants on one deployment is what a hosted control plane is for.

Nothing is required to set it up. The first request to one of those endpoints provisions a default organization, a default workspace, and one owner identity representing the operator, and every later request resolves that same identity. Organization owners and admins can create further workspaces, add members, and manage roles; a workspace's own owners and admins can manage the workspace they belong to.

Adding a member takes an email address (`POST /v1/organizations/me/members`), optionally with the workspaces to grant at the same time. If no identity holds that address yet, one is created carrying it, and the member is active immediately with nothing emailed. Such an identity is a roster and attribution entry today: it carries no password, and only the operator can set one for their own identity, so it cannot sign in until the signup and reset flows land. The address is the handle those flows will match it on.

### Invitations

`POST /v1/organizations/me/member-invitations` is the other way to add someone: the membership lands `invited` rather than `active`, and an email with an accept link goes out if mail is configured (see [Configuration](configuration.md)). If it isn't, the response still carries the link (`accept_link`) so the operator can share it another way; `mail_sent` says whether it was actually emailed.

The recipient follows the link to a public accept page (`POST /v1/invitations/validate` to preview it, `POST /v1/invitations/accept` to commit, both with the token in the body rather than the URL), which resolves the membership to `active` and grants any workspaces parked on the invitation. No session is minted, and accepting sets no password: the identity it resolves to is in the same state as an address added directly, so it signs in once the signup and reset flows give it a way to set one.

An invitation expires after `invitation_expiry_hours` (default 7 days) and can be revoked before it is accepted (`DELETE /v1/organizations/me/member-invitations/{invitation_id}`), which cancels it and suspends the membership, the same as removing a member. Re-inviting the same address revives it.

Three rules exist to stop a tenancy from becoming unmanageable or from losing data:

- an organization always keeps at least one active owner, so the last owner cannot be demoted or removed (`400`);
- an organization always keeps at least one workspace, so the last one cannot be deleted (`400`);
- a workspace that still holds API keys, usage, aliases, or routing policies cannot be deleted (`409`). Move or delete what it holds first.

Removing a member suspends their membership rather than deleting it, which keeps their past usage attributable.

Granting the `owner` role is an owner's to give. An admin manages members, workspaces and roles, and cannot promote anyone (themselves included) to owner, nor add one.

### Dashboard sessions and identity

Signing in to the dashboard exchanges a credential for a session: an opaque token in an HttpOnly cookie, stored only as a SHA-256 hash, revocable server-side, expiring on `dashboard_session_ttl_hours`. Each session names the identity it was minted for, so a cookie-authenticated request resolves a user and, through that user's `active_organization_id`, the organization it is acting in. `POST /v1/auth/session` returns both ids alongside the expiry.

**Which credential signs in depends on whether the deployment has been claimed.** A deployment where no identity has a password yet accepts the master key, which is what makes first boot work with nothing configured. The moment an identity has one, the master key stops being accepted at the sign-in endpoint and email and password is the login. `GET /v1/bootstrap` publishes which of the two applies right now, in `sign_in_methods`, so the sign-in page asks for the credential that will work:

```bash
curl http://localhost:8000/v1/bootstrap        # no credential needed
# {"deployment_type":"standalone", ..., "sign_in_methods":["master_key"]}
```

Nothing schedules the switch and nothing expires. A deployment that never claims goes on signing in with the master key indefinitely, which is a reasonable end state for a single-operator gateway that only ever talks to itself.

#### Claiming the deployment

First boot provisions the operator identity as a label with no address and no password, so claiming supplies both. The dashboard offers it at **Account settings**, reached from the account control at the foot of the sidebar, which is one call authenticated by the session the master key already minted. The same call over HTTP, authenticated by the master key itself:

```bash
curl -X PUT http://localhost:8000/v1/auth/password \
  -H "Otari-Key: <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "new_password": "<a password>"}'
```

- A password is at least 8 characters and at most 72 bytes, which is bcrypt's own ceiling; accented and non-Latin characters count for more than one byte each. There are no composition rules.
- The address is stored lower-cased and matched case-insensitively at sign-in. It is not delivered to and not verified by anyone: this edition has no verification flow yet, and the master key is what proved the claim.
- The same endpoint changes a password later, and the same **Account settings** page is where the dashboard does it. From a session it needs `current_password`; sent with the master key in a header it does not, which is the recovery path (see below) and the one form of this call the dashboard cannot make, since reaching the page needs a session. Changing the address afterwards is refused rather than half-supported.
- Every other session that identity holds is revoked, so a cookie minted under the old password does not outlive it. The caller's own session is spared when the change came from the browser, and is not when it came from the master key, since a header caller has no session to keep.
- **Claiming is one-way.** No endpoint clears a password, so a deployment that has been claimed cannot be returned to master-key sign-in. The sign-in screen follows `sign_in_methods` and asks for the address and password from then on, and the master key still authenticates the whole management API, so the step costs you nothing you cannot reach; it is simply not reversible. See the [Admin dashboard guide](dashboard.md).

After that, sign in with the address:

```bash
curl -X POST http://localhost:8000/v1/auth/session \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "<the password>"}'
```

A failed sign-in answers `401 Incorrect email or password` whichever part was wrong, and takes the same time to do it, so the endpoint cannot be used to find out which addresses hold an account. A master key presented after the deployment is claimed answers `403` naming the password login, which is a different answer from a wrong master key's `401` on purpose: the credential is fine, that use of it is over.

#### What the master key still does

Everything else. It authenticates `/v1/keys`, `/v1/users`, `/v1/budgets`, and the rest of this guide exactly as before, in the `Otari-Key` or `Authorization` header. Claiming a deployment changes one endpoint's answer and no others.

It also stays the way back in. An operator who forgets the password sets a new one through the same `PUT /v1/auth/password`, with the master key in a header and no `current_password`. That is deliberate rather than a gap: a caller holding the master key can already do anything the management API can do, so asking them for a password they have lost would lock the dashboard while leaving the API wide open. Password recovery for someone who holds no master key needs mail, and arrives with the signup and reset flows.

#### Who can sign in, and who cannot yet

Only an identity with a password, and only the operator can get one today: the endpoint above always acts on the caller's own identity, so there is no way for an admin to set someone else's. A member added by address holds a role and can be placed in workspaces, and their address is the handle the signup and reset flows will match them on. Those flows, along with email verification and OAuth and passkey sign-in, are the rest of the identity track.

A session is revoked on sign-out, on a password change as described above, on master-key rotation (every session, with the rotating tab's own re-minted for the same identity), when the master key changes across a restart, and when the identity it names is deleted or deactivated. A deactivated identity also stops being able to sign in, rather than keeping access until its cookie expires. Deactivation is enforced when the session is read, and the identity's sessions are deleted at that point rather than only refused, so re-activating it later does not hand back the access of any cookie that was presented while it was off. Nothing sweeps the rest: a cookie that is never presented in that window survives to its TTL, because no flow here deactivates an identity and none therefore revokes ahead of the read.

An opaque session token is the settled shape here, not a stopgap: it is revocable, which a bearer JWT is not, and [mozilla-ai/otari-ai#1716](https://github.com/mozilla-ai/otari-ai/issues/1716) settled that sessions are the steady-state dashboard login. **The platform's JWT `Token` therefore does not survive the rehome.** Anything on the platform frontend that reads or stores a bearer token changes when its pages arrive: the credential is an HttpOnly cookie the page's own script cannot read, sent automatically and same-origin only, so there is no token to attach to a header and no expiry to decode out of a payload. Sign-in state comes from the session endpoint's response and from a 401 bounce, not from inspecting a token.

### Adopting an existing tenancy

Provisioning adopts an organization whose slug is `default`, which is the one it would have created itself. It cannot adopt any other, because every route is scoped to the organization the operator identity is currently pointed at, and there is no route to list, switch, or fetch an organization by id. So an organization this deployment did not provision is unreachable through the API until the operator identity points at it.

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

- [Admin dashboard](dashboard.md): the same users, keys, and budgets in the browser UI.
- [Configuration](configuration.md): `reject_user_mismatch`, `budget_strategy`, and related settings.
- [API reference](api-reference.md): the full endpoint and schema listing.
