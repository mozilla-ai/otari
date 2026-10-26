# Sign-in and accounts

This page sets up the ways people sign in to a gateway's dashboard, and explains
how accounts, sessions, signup and the deployment-wide account controls work.
Password sign-in needs no setup; passkeys and Google or GitHub sign-in do.

## Set up passkeys

1. Set `public_base_url`, which establishes the origin and relying-party ID.
2. Only when passkeys must be bound to a parent domain, set `webauthn_rp_id`.
3. Where an edge serves the dashboard on a different host to the gateway, set
   `webauthn_rp_id` to a domain that is a parent of both, and list the dashboard
   origin in `webauthn_allowed_origins`. The ID is not derived from
   `ui_base_url`, so without this the ceremony fails in the browser and nothing
   is logged here ([#1134](https://github.com/mozilla-ai/otari/issues/1134)).

## Set up OAuth sign-in

1. Set `public_base_url`.
2. Register this redirect URI with the provider:

   ```text
   {public_base_url}/auth/{provider}/callback
   ```

   The gateway answers that path itself and redirects the browser into the
   dashboard to finish.
3. Set the provider's client ID and secret (`google_client_id` and
   `google_client_secret`, or `github_client_id` and `github_client_secret`).
4. Where an edge serves the dashboard elsewhere, set `ui_base_url` too; see
   [The interface address](configuration.md#the-interface-address).

## How it works

### Users and identities

Otari maintains identities for dashboard sign-in and user records for request
attribution and per-user budgets. Management flows connect them where needed.
Deleting or deactivating a user prevents future access but preserves historical
usage.

### Dashboard sessions and identity

The first operator signs in with the master key. Otari exchanges it for an
opaque, HttpOnly session cookie. After the operator sets an email and password,
the dashboard uses that identity for sign-in; the master key remains an API
credential and recovery path.

Email and password sign-in is offered whenever any active identity holds a
password, not only once the operator has claimed the deployment. A member added
to the roster and signed up before that point signs in on the same screen, which
offers the master-key box beside the form while both credentials still work.

Sessions are revocable and expire after `dashboard_session_ttl_hours`. Password
changes, master-key rotation, sign-out, and identity deactivation revoke relevant
sessions. A request whose session cookie no longer authenticates is refused
with 401 "Your dashboard session has ended. Sign in again."

A session authorizes the management API. It does not authorize
`/api/v1/chat/completions` or any other inference path, which take an API key
or the master key and nothing else: a keyless request resolves to the
deployment's default workspace, so honoring a cookie there would let any member
of any organization spend that workspace's provider credential.

The Playground is the one surface that runs a completion from a session, and it
is a separate endpoint rather than a relaxation of that rule.
`POST /api/v1/playground/chat/completions` resolves the caller's own attribution
user and proves their membership of the workspace it will bill before the
request reaches the pipeline, so the request is billed to the person who sent it
in a workspace that is theirs. No credential is minted for the browser and none
is held there. The usage row it writes carries no `api_key_id` and its own
endpoint label, which is what keeps in-product traffic separable from a
customer's integration.

### Passkeys

Passkeys are optional and additive to password sign-in; a passkey never creates
an account. A passkey is bound to a relying-party ID, a domain, which Otari takes
from `public_base_url` unless `webauthn_rp_id` names one. Changing the
relying-party ID makes existing passkeys unusable. The dashboard continues
listing unusable credentials so the owner can remove them.

`passkeys_enabled` (`OTARI_PASSKEYS_ENABLED`) defaults to `true`. With it
`false`, none of the passkey routes are mounted (they answer 404, not 503), no
`passkey` sign-in method is published, the dashboard hides its passkeys page,
and the `webauthn_*` settings are not checked at startup, so values left over
from before are inert. Setting it is the only way to switch passkeys off while
keeping `public_base_url`, which OAuth redirects and mail links need. Passkeys
already registered stay in the database, and work again if the setting is
turned back on.

### OAuth sign-in (Google and GitHub)

#### How provider accounts are linked

Otari links each provider account to one Otari account by the provider's own ID for that person, not by the address. An Otari account holds at most one link for each provider, so it can link both a Google and a GitHub account. A sign-in through a link still works after the address changes at the provider. A second account at the same provider is refused, even with the same address. Otari has no way to remove a link yet, so a person whose provider account was replaced signs in with a password or a passkey instead.

The first sign-in with a provider account links it to the account that holds the address the provider verified. Otari links by address only for Google and GitHub, because no tenant can set what they say about an address. If no account holds that address, the result depends on `open_signup`, as it does for [signup](#signup). When `open_signup` is disabled, the sign-in is refused, and so is one on an account an owner or admin added or invited whose address nobody has verified yet, unless it is still an active member or has an invitation it can accept, the same rule [signup](#signup) follows. When it is enabled, Otari creates an account with its own organization and workspace, and sends no verification email, because the provider has already verified the address. The first sign-in is refused if the provider did not verify the address. Any sign-in is refused if the provider says the address is not verified, or if the account is deactivated. A provider that shares no address at all is refused with a message saying so. A provider that fails, cannot be reached, or refuses this deployment's client credentials while finishing the sign-in answers 502, so a provider outage or a misconfigured client is not counted as a client error; only a code the provider calls invalid, expired or spent answers 400.

The first sign-in on an address that is not yet verified, or that an organization only vouched for (see [Invitations](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/organizations-and-access/members#invitations)), marks it verified. It also removes everything set up on that address before then: its password, any verification or reset link, every session signed in with that password, and every passkey registered from one. On an unverified address it replaces the display name chosen at signup with the provider's, and renames the organization that signup created after it, unless someone else belongs to that organization. The provider confirms who owns the address, not who chose that password or that name. The callback's response says so (`password_removed`), and the dashboard tells the person before they go on, so the password does not just stop working. They can set a new password from their account page once signed in. A password on an address that was already verified is kept, with its sessions and passkeys. A later sign-in through the link does not change the account's password or verification, unless the address has since become only vouched for (an operator's [password link](#deployment-wide-account-administration) that was passed on rather than mailed does that): the sign-in then proves the address the same way, and removes the password, links, sessions and passkeys set up through the vouch.

Where the deployment publishes terms (`terms_url`), a sign-in that creates an account, or that is the first way in to one an owner or admin added or invited, needs the person's acceptance of them, as the signup form and a first password from an invitation link do. The dashboard asks before it sends the person to the provider and returns the acceptance with the callback (`terms_accepted`). Without it the sign-in is refused with the code `terms_not_accepted`, and nothing it would have set up is kept: no account, organization, workspace, provider link or session. With it, the acceptance is recorded on the account. A sign-in to an account someone already claimed, or through a provider account already linked to one, does not ask, and an account that never accepted the terms is not asked to later.

A sign-in that creates an account, or that is the first way in to one an owner or admin added, is reported as a signup to whatever growth integration the deployment runs, as the signup form, a first password set from an invitation link, and a first password set by reset are.

### Signup

Signup sets a password for an address and sends a verification link. The link
verifies the address only together with that password: the verification page
asks for it, and a wrong one is refused without spending the link, with a
message saying the link still works. A link that is unknown, expired or spent
gets a different message, which says nothing about the password. Wrong
passwords are counted per client address, so a held link cannot be used to guess
the password at speed; a right one is never held back. Anyone can
sign up with an address they do not hold, and the link then reaches the
address's real owner, whose click must not activate a password a stranger
chose. Signing up again before verifying changes nothing, so the first password
is the one the link takes. An owner who never signed up, or who forgot that
password, uses "Forgot your password?" instead, which replaces it. The
verification and reset pages check their link as they open, without spending
it, and say at once when it has expired or was already used (or a newer link
replaced it). Where the deployment publishes terms (`terms_url`),
signup is refused unless they are accepted, with the form or with a provider (see
[OAuth sign-in](#oauth-sign-in-google-and-github)). What an unknown address does depends
on `open_signup`:

- `false` (default): signup only completes an identity an owner or admin already
  added or invited by address, and only while it is still wanted: an active
  member, or an invitee whose invitation is still pending. An address nobody has
  added gets no account, and neither does one whose invitation was revoked or
  expired or whose membership was removed. This is the posture a single-tenant
  deployment wants, since anyone who can reach the dashboard can reach the form.
  The rule holds when a link is used as well as when it is mailed, so revoking
  the invitation or removing the member also stops a verification link, a resent
  one, and a reset link minted before then. For the same reason, creating an
  organization requires an active membership in one already, so a member who was
  removed cannot make themselves a tenant.
- `true`: an unknown address is registered, with an organization and workspace of
  its own. So is an address an organization once added or invited but no longer
  wants (its invitation was revoked, declined or expired, or the member was
  removed) when it is claimed, that is, when its owner first proves it: by
  signing up with the form (or verifying that signup), by a first sign-in with a
  provider, or by choosing a password through "Forgot your password?" while the
  address is unverified or only vouched for (see [Invitations](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/organizations-and-access/members#invitations)).
  It gets its own organization and workspace, and signs in to them. An address
  is claimed once: someone who already proved their address and later declines
  their last invitation or is removed keeps the account but has no
  organization, and signs in to the empty state [Invitations](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/organizations-and-access/members#invitations)
  describes, where they can create one. An address with an invitation still
  waiting is claimed as it is, and finds the invitation under Invitations. Use
  it where the deployment serves many tenants.

Signup never sets a password on an address that is already verified. That is the
state a Google or GitHub sign-in leaves, and the person who signs in that way
adds a password from Account settings while signed in. Such a signup mails the
address a notice that it already has an account and how it signs in: with its
password (and "Forgot your password?" if it is forgotten), or with its provider.

"Forgot your password?" mails a link to set or reset the password of an
address that has a password, and also of one an owner or admin added or invited
that nobody has signed in to yet, so that person can choose a first password.
Completing it verifies the address, since the link reached its mailbox, and where
the address was unverified or only vouched for it also ends every session and
removes every passkey set up before then. Where it was unverified, the display
name chosen at signup is dropped too, and the organization that signup created is
renamed after the address, unless someone else belongs to it. Where `open_signup`
is disabled, every address that is not yet verified, with or without a password,
follows the signup rule above: it gets a link, and the link completes, only while it is an active
member or holds an invitation it can still accept, so a revoked invitee gets
none. Where the deployment publishes terms and the account has accepted none,
the reset page asks for them. An address that signs in only through a provider
gets no link; it is mailed the same notice signup sends, saying which provider
it signs in with and that a password can be added from Account settings.

Either way the response says the same thing whether the address was unknown,
already claimed, already verified, or genuinely just claimed, so its body
discloses nothing about the address. A notice goes only to the address's own
mailbox, so it tells nobody else anything either. Response *timing* still does, because the
eligible path sends mail before it answers; that is [otari#720](https://github.com/mozilla-ai/otari/issues/720)
and it applies to both postures.

Signup needs mail configured, because an account that cannot verify its address
cannot sign in.

Open signup puts tenant creation on an unauthenticated route. The per-IP
throttle on the public auth routes is the only bound on it today, and nothing
expires the organization an unverified signup leaves behind, so run it behind
whatever edge controls the deployment has.

### Deployment-wide account administration

The master key controls the deployment. A dashboard session can perform
deployment-wide operations only when its identity has operator authority.
Organization roles do not implicitly grant access to process-wide settings or
credentials.

Use the master key through `Authorization: Bearer <master-key>` or
`Otari-Key: <master-key>`. Applications should use scoped API keys instead.

An operator manages every account on the deployment from the dashboard's
Accounts page (`/api/v1/admin/users`): deactivate or reactivate one, grant or
remove operator access, and send a password link. The page lists accounts one
page at a time and searches them by name or email (`search` on the route,
matched case-insensitively). Deactivating an account ends
its sessions and voids its pending verification and reset links, so
reactivating it does not bring an old link back.

A password link is how an operator helps somebody who cannot sign in, and the
only way back for a forgotten password on a deployment that sends no mail.
Where mail is configured, the link is emailed to the account's address, as
"Forgot your password?" does. Otherwise, or when the send fails, the operator
is given the link to pass on. It works once, expires after
`password_reset_expiry_hours`, and replaces any earlier reset link. Because the
mailbox never saw a link that was passed on, a password chosen through one
records the address as vouched for rather than proven, the same as a password
chosen through an [invitation](https://mozilla-ai.gitbook.io/otari-preview/wBxOPVaXYufbmAHVFj2k/guides/organizations-and-access/members#invitations) link. A link is refused for a
deactivated account, one with no address, one that has no password and signs in
with a provider or a passkey (its owner adds a password from Account settings
once signed in), and, where `open_signup` is disabled, an unverified address no
organization still has as a member or invitee. Where `open_signup` is enabled
such an address gets a link, and choosing a password through it gives the
account an organization and workspace of its own, as [signup](#signup) does.

`PUT /api/v1/auth/password` with the master key acts on the operator's own
account, never another one: an address sent with it is the operator's sign-in
address, and on an unclaimed deployment it claims the deployment. It refuses an
address another account holds, and that refusal points to the password link.
