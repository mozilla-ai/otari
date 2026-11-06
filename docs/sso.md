# Single sign-on

An organization can configure its own OpenID Connect (OIDC) connection and let
its people sign in through it instead of Otari's own sign-in methods. An owner
or admin configures the connection from the organization's Single sign-on
settings page, which also shows the preflight report and the enforcement
controls this page documents.

## What admits a sign-in

A domain's DNS proof is what admits a person through a connection, never the
identity provider's own `email_verified` claim: every connection sets
`trust_email_verified` to false, so a provider's word about an address is not
trusted on its own. The organization still has to prove the domain through the
same claim process the Email domains page manages, and the proof that admits a
sign-in is the organization's, not the provider's.

A domain's proof expires 90 days after it was verified, and nothing re-proves
it automatically. Once every domain's proof has lapsed, the connection stops
admitting anyone until one is re-verified. Re-verify a domain before its proof
expires to avoid a gap.

Two organizations may configure a connection to the same issuer. A person is
admitted to an organization only on a domain that organization itself has
proven, never on the issuer alone.

## Accounts

A person's identity at a connection is keyed on the connection's issuer and
the provider's own subject for them, never on their email address. Someone
with no existing link gets a new account; Otari never matches them to an
existing account by address, name, or anything else. That account holds no
email address, so API keys, budgets, and usage stay on whichever account
already held them: an identity change never moves them to a new account.

Where the organization already holds another account for the address a
connection's first sign-in claims, the new account's first visit to the
dashboard carries a message saying so and naming no other account's details.
The two accounts are never merged automatically.

**API keys are rotated, not moved.** A key stranded on an older account stays
usable from that account; there is no operation that transfers a key to
another account. Issue fresh keys on the new account and revoke the old ones.

## The preflight report

Before an administrator can require sign-in through a connection, the
settings page reports who it would affect, against the one predicate
`OIDCConnectionService.preflight_sso_enforcement_for_user` states:

> An active member is affected when they hold no sign-in link to the
> connection.

They would get a new account the first time they use the provider, and the
account they sign in with today keeps whatever API keys it holds.

The count this report gives does not apply the owner exception or the
master-key exemption that enforcement itself applies, so it is greater than
or equal to the number of people enforcement would actually refuse. That is
deliberate: the report answers who must do something (rotate their keys),
not who would be locked out, and it errs toward over-warning.

## Requiring sign-in

An organization can require sign-in through its own connection. Once
required, every other way to act in that organization is refused for an
affected membership, including a live session at the moment enforcement is
switched on, the invitation password-set path, and a password-reset link.
Enforcement attaches to the membership, not to the account: a person who
belongs to another, non-enforcing organization still signs in to that one.

Requiring sign-in is refused before anybody has signed in through the
connection, since an organization cannot require a route nobody has proven
works. It is also refused unless the request names one of the organization's
own owners who keeps a non-SSO route, or explicitly waives that exception. An
organization that waives it gives up its own way back in: if every sign-in
through the provider ever fails, nobody in that organization can sign in
until a deployment operator steps in. The deployment's master-key operator is
never subject to any organization's enforcement.

## Removing a member's link

An owner or admin can remove one member's own link to the connection, which
lets that member link afresh at their next sign-in. This removes a link only.
It does not revoke the member's API keys, does not delete their account, and
does not end their membership; "Remove member" is the control for that. Their
next sign-in through the connection creates a new account, under the same
rule as a first sign-in, and the account they signed in with before keeps its
own API keys, budgets, and history.

## Network egress

Reading a connection's provider configuration and its signing keys is an
outbound request to a host the organization names, not the deployment
operator. Otari refuses a plain HTTP document and a private or reserved
address unless `oidc_allow_private_hosts` opts in, but a hostname can still
resolve to a different, internal address after that check runs: DNS rebinding
defeats a host allowlist checked at one point in time. Run Otari behind
whatever network-layer egress policy the deployment already has; the host
checks here are not a substitute for one.
