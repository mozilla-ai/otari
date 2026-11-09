# Sign-in and accounts

This page sets up the ways people sign in to a gateway's dashboard, and explains
accounts, sessions, signup and the deployment-wide account controls. Password
sign-in needs no setup; passkeys and Google or GitHub sign-in do.

## Set up passkeys

1. Set `public_base_url`, which sets the origin and relying-party ID.
2. Only when passkeys must be bound to a parent domain, set `webauthn_rp_id`.
3. Where an edge serves the dashboard on a different host from the gateway, set
   `webauthn_rp_id` to a parent domain of both, and list the dashboard origin in
   `webauthn_allowed_origins`. The ID is not derived from `ui_base_url`, so
   without this the ceremony fails in the browser and nothing is logged
   ([#1134](https://github.com/mozilla-ai/otari/issues/1134)).

## Set up OAuth sign-in

1. Set `public_base_url`.
2. Register this redirect URI with the provider:

   ```text
   {public_base_url}/auth/{provider}/callback
   ```

   The gateway answers that path and redirects the browser into the dashboard.
3. Set the provider's client ID and secret (`google_client_id` and
   `google_client_secret`, or `github_client_id` and `github_client_secret`).
4. Where an edge serves the dashboard elsewhere, also set `ui_base_url`; see
   [The interface address](configuration.md#the-interface-address).

## How it works

### Users and identities

Otari keeps identities for dashboard sign-in and user records for request
attribution and per-user budgets. Deleting or deactivating a user blocks future
access but keeps historical usage.

### Dashboard sessions and identity

The first operator signs in with the master key, which Otari exchanges for an
opaque, HttpOnly session cookie. Once the operator sets an email and password,
the dashboard signs in with those; the master key stays an API credential and
recovery path.

The email and password form appears whenever any active identity has a
password, even before the operator claims the deployment. Until then, the
master-key box is shown beside it.

Sessions expire after `dashboard_session_ttl_hours` and can be revoked.
Password changes, master-key rotation, sign-out and identity deactivation
revoke the relevant sessions. A request with a session that no longer
authenticates gets 401 "Your dashboard session has ended. Sign in again."

A session authorizes the management API only. Inference paths such as
`/api/v1/chat/completions` take an API key or the master key, never a cookie: a
keyless request falls to the deployment's default workspace, so a cookie there
would let any member of any organization spend that workspace's provider
credential.

The Playground is the exception, through its own endpoint,
`POST /api/v1/playground/chat/completions`:

- It bills the signed-in person, in a workspace they are proven to belong to.
- A viewer of the organization or workspace is refused (`403`), for completions
  and Playground file uploads alike.
- No credential is minted for or held in the browser.
- The usage row has no `api_key_id` and its own endpoint label, so in-product
  traffic stays separate from integrations.

### Passkeys

Passkeys are optional and add to password sign-in; a passkey never creates an
account. Each passkey is bound to a relying-party ID (a domain), taken from
`public_base_url` unless `webauthn_rp_id` sets one. Changing the relying-party
ID makes existing passkeys unusable; the dashboard keeps listing them so their
owner can remove them.

`passkeys_enabled` (`OTARI_PASSKEYS_ENABLED`, default `true`) is the only way
to turn passkeys off while keeping `public_base_url`, which OAuth redirects and
mail links need. With it `false`:

- the passkey routes are not mounted (they answer 404, not 503)
- no `passkey` sign-in method is published, and the dashboard hides its
  passkeys page
- the `webauthn_*` settings are not checked at startup
- registered passkeys stay in the database and work again if it is turned back
  on

### OAuth sign-in (Google and GitHub)

#### How provider accounts are linked

Otari links each provider account to one Otari account by the provider's ID for
the person, not by address:

- An account has at most one link per provider, so it can link one Google and
  one GitHub account.
- A linked sign-in keeps working after the address changes at the provider.
- A second account at the same provider is refused, even with the same address.
- There is no way to remove a link yet. A person whose provider account was
  replaced signs in with a password or passkey instead.

**First sign-in.** The provider account is linked to the Otari account holding
the address the provider verified. If no account holds it, `open_signup`
decides, as for [signup](#signup):

- Disabled: the sign-in is refused. So is one on an account an owner or admin
  added or invited whose address nobody has verified, unless it is still an
  active member or has an invitation it can accept.
- Enabled: Otari creates an account with its own organization and workspace,
  with no verification email, since the provider verified the address.

**Refusals and errors.**

- A first sign-in is refused if the provider did not verify the address.
- Any sign-in is refused if the provider says the address is unverified, or if
  the account is deactivated.
- A provider that shares no address is refused with a message saying so.
- A provider that fails, cannot be reached, or refuses this deployment's client
  credentials gets 502. Only a code the provider calls invalid, expired or
  spent gets 400.

**What a first sign-in replaces.** On an address that is unverified, or only
vouched for by an organization (see [Invitations](members.md#invitations)), the
first sign-in marks it verified and removes what was set up before: the
password, any verification or reset link, sessions signed in with that
password, and passkeys registered from one. On an unverified address it also
replaces the signup display name with the provider's, and renames the
organization signup created after it, unless someone else belongs to it.

The callback response flags this (`password_removed`), and the dashboard warns
the person before they continue. They can set a new password from their
account page. A password on an already verified address is kept, with its
sessions and passkeys.

Later linked sign-ins leave the password and verification alone, unless the
address has since become only vouched for (an operator's
[password link](#deployment-wide-account-administration) passed on rather than
mailed does that). Then the sign-in proves the address again and removes what
was set up through the vouch.

**Terms.** With `terms_url` set, a sign-in that creates an account, or is the
first way in to one an owner or admin added or invited, needs the person to
accept the terms. The dashboard asks before sending them to the provider and
returns the acceptance with the callback (`terms_accepted`). Without it, the
sign-in is refused with `terms_not_accepted` and nothing is kept (no account,
organization, workspace, link or session). Sign-ins to an already claimed
account, or through an already linked provider account, do not ask.

A sign-in that creates an account, or is the first way in to one an owner or
admin added, is reported as a signup to the deployment's growth integration,
like the other signup paths.

### Signup

Signup sets a password for an address and sends a verification link. Signup
needs [mail](configuration.md#mail), because an account that cannot verify its
address cannot sign in.

The link verifies the address only together with that password, because anyone
can sign up with an address they do not hold:

- The verification page asks for the password. A wrong one is refused without
  spending the link; an unknown, expired or spent link gets a different
  message that says nothing about the password.
- Wrong passwords are throttled per client address; a right one is never held
  back.
- Signing up again before verifying changes nothing: the first password is the
  one the link takes. An owner who never signed up, or forgot that password,
  uses "Forgot your password?" instead.
- The verification and reset pages check their link on open, without spending
  it, and say at once if it expired, was used, or was replaced by a newer one.

With `terms_url` set, signup is refused unless the terms are accepted, with the
form or with a provider (see [OAuth sign-in](#oauth-sign-in-google-and-github)).

What an unknown address does depends on `open_signup`:

- `false` (default), for a single-tenant deployment: signup only completes an
  identity an owner or admin added or invited, and only while it is still
  wanted (an active member or a pending invitee). Revoking the invitation or
  removing the member also stops any verification or reset link minted before.
  Creating an organization requires an active membership in one, so a removed
  member cannot make themselves a tenant.
- `true`, for many tenants: an unknown address is registered with its own
  organization and workspace. So is an address an organization no longer wants
  (invitation revoked, declined or expired, or member removed) when its owner
  first proves it: by signing up (or verifying that signup), by a first
  provider sign-in, or by choosing a password through "Forgot your password?"
  while it is unverified or only vouched for (see
  [Invitations](members.md#invitations)). An address is claimed once: someone
  who already proved it and later loses their last organization keeps the
  account, and signs in to the empty state where they can create one. An
  address with a pending invitation is claimed as is and finds it under
  Invitations.

Signup never sets a password on an already verified address (the state a
Google or GitHub sign-in leaves). Instead it mails the address a notice saying
it already has an account and how it signs in. A provider-only person adds a
password from **Account settings** while signed in.

#### Forgot your password?

"Forgot your password?" mails a link to set or reset the password of an
address that has one, or of one an owner or admin added or invited that nobody
has signed in to yet.

- Completing it verifies the address. Where the address was unverified or only
  vouched for, it also ends every session and removes every passkey set up
  before. Where it was unverified, the signup display name is dropped and the
  organization signup created is renamed after the address, unless someone
  else belongs to it.
- With `open_signup` disabled, an unverified address gets a link, and the link
  completes, only while it is an active member or has an invitation it can
  accept.
- Where terms are published and the account accepted none, the reset page asks
  for them.
- An address that signs in only through a provider gets no link, but the same
  notice signup sends.

#### What the responses reveal

Signup and reset give the same response whether the address was unknown,
claimed, verified or just claimed, and notices go only to the address's own
mailbox. Response timing still differs, because the eligible path sends mail
before answering ([otari#720](https://github.com/mozilla-ai/otari/issues/720)).

Open signup puts tenant creation on an unauthenticated route. The per-IP
throttle on the public auth routes is its only bound, and nothing expires the
organization an unverified signup leaves behind, so run it behind your edge
controls.

### Deployment-wide account administration

The master key controls the deployment. A dashboard session can do
deployment-wide operations only when its identity has operator authority;
organization roles do not grant access to process-wide settings or credentials.

Send the master key as `Authorization: Bearer <master-key>` or
`Otari-Key: <master-key>`. Applications should use scoped API keys. An
inference request on the master key uses the deployment's default workspace.

Operators can read the operator usage API (`/api/v1/usage`), covering every
workspace of the operator's active organization (the default organization for
the master key).

The **Accounts** page (`/api/v1/admin/users`) lists every account, a page at a
time, searchable by name or email (`search`, case-insensitive). From it an
operator can:

- deactivate or reactivate an account. Deactivating ends its sessions and voids
  its pending verification and reset links, so reactivating does not revive
  them.
- grant or remove operator access.
- send a password link.

#### Password links

A password link helps someone who cannot sign in, and is the only way back
from a forgotten password on a deployment without mail.

- With mail configured, the link is emailed to the account's address.
  Otherwise, or if sending fails, the operator gets the link to pass on.
- It works once, expires after `password_reset_expiry_hours`, and replaces any
  earlier reset link.
- A password chosen through a passed-on link records the address as vouched
  for, not proven, like an [invitation](members.md#invitations) link.
- It is refused for a deactivated account, one with no address, one with no
  password that signs in with a provider or passkey, and, with `open_signup`
  disabled, an unverified address no organization still has as a member or
  invitee. With `open_signup` enabled, such an address gets a link, and the
  password chosen through it gives the account its own organization and
  workspace.

`PUT /api/v1/auth/password` with the master key acts only on the operator's own
account: an address sent with it is the operator's sign-in address, and on an
unclaimed deployment it claims the deployment. It refuses an address another
account holds, pointing to the password link.

#### Blocking email addresses

The **Accounts** page also keeps the email blocklist
(`/api/v1/admin/blocked-emails`). An entry is either:

- a whole address (`person@example.com`), or
- a domain (`example.com`, also accepted as `@example.com` or
  `*.example.com`), which also blocks every subdomain.

Entries match case-insensitively, and an internationalized domain matches in
either spelling. `*.` and `@` mark a domain, so they are refused in front of an
address.

A blocked address cannot sign up, be added to or invited into an organization,
accept an earlier invitation, or sign in by password, passkey or provider.
Links mailed before the block are refused as invalid, and an operator cannot
send it a password link.

- Signup, resending verification and password reset answer as for an unknown
  address and send nothing, so the block is not revealed.
- A sign-in is refused with 403 once the credential checks out, and a first
  provider sign-in keeps nothing it would have created.
- A sign-in in flight when an entry is added cannot leave a live session.

Adding an entry ends the dashboard sessions of every matching account; the
response says how many. The accounts are kept, so removing the entry lets them
sign in again. API keys are not affected: revoke those on **Keys**.

An operator cannot block their own address or the bootstrap operator's, just as
neither can be deactivated.
