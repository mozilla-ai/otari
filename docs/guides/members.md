# Members and invitations

Owners and admins invite people to an organization and choose which workspaces
each person can use. Roles are described in
[Organizations and workspaces](organizations-and-workspaces.md#roles-scoping-and-deleting-an-organization).

## Invite a member

1. Open **Organization** and select **Members & roles**.
2. Click **Invite member**.
3. Enter one or more email addresses, separated by commas or new lines.
4. Select the **role**: Owner, Admin, Member or Viewer.
5. Choose the **workspaces** the person should have access to.
6. Click **Invite member**. The person is emailed an invitation, and the
   membership becomes active once they accept it.

Owners and admins can also change a member's role, change which workspaces they
can access, or remove them from the organization.

## Users and identities

Otari maintains identities for dashboard sign-in and user records for request
attribution and per-user budgets. Management flows connect them where needed.
Client-provided `user` values are never trusted to move spend away from the API
key's bound user. The one exception is a [service key](api-keys.md#service-keys-and-end-users),
which bills end users that belong to its own user.

A user's `allowed_models` is inherited by newly created keys unless the key
defines its own list. A missing list allows any model, an empty list allows none,
and entries may use provider wildcards such as `openai:*`.

Deleting or deactivating a user prevents future access but preserves historical
usage.

## Invitations

An owner or admin can invite a person to an organization and selected workspaces.
If mail is configured, Otari emails the accept link. When the email does not go
out (no mail configured, the send failed, or a mail limit below was reached), the
dashboard shows the link instead, so it can be shared by hand. A link that was
emailed is never shown to the inviter, so only the invitee's mailbox holds it. The
email names the inviter by their address as well as the name they chose, and an
address the mail server would refuse, such as one ending in a dot, is refused
when it is invited.

Invitation email is limited, because it carries text an organization chose (its
name, the inviter's name) from the deployment's own sender, and on a deployment
where anyone can create an organization that would otherwise make it an open
relay. In any 24 hours an organization sends at most
`invitation_mail_limit_per_organization` invitation emails (250 by default), and
an address receives at most `invitation_mail_limit_per_address` (5 by default)
from every organization together, so revoking and inviting again cannot flood it.
An invitation over a limit is still created, and its link is returned to share by
hand, as when mail is off: the link costs the deployment nothing, and refusing
would stop a legitimate admin partway through a bulk invite. The counts are kept
in `rate_limit_store`, so with Redis they hold across replicas.

A link that was lost can be reissued. Reissuing gives a pending or expired
invitation a new link, emailed or returned as above, and the old link stops
working. Inviting an address whose invitation is still pending is refused and says
to reissue it instead. On Members, an invited row says when its invitation has
expired, and offers Reissue and Revoke either way.

The API can also add a member directly (`POST /api/v1/organizations/me/members`),
active at once with no invitation to accept and no email. A new address added
that way gets in by signing up, by "Forgot your password?" or through a provider,
so on a deployment with no mail and no provider it waits for an operator's
password link. Invite it instead, since the invitation link itself can set its
first password.

To invite several people at once, paste their addresses into the invite dialog,
separated by commas or new lines (up to 100). Everyone gets the same role and
workspaces. Each address is invited or refused on its own, so one that is already
a member does not stop the rest, and the result lists every address with whether
its email went out, or its accept link to share when it did not.

Opening the link lets the invitee accept. If the invited address has never signed
in, the accept page asks them to choose a first password, and once they accept
they can sign in straight away. No verification email is needed, so this works on
a deployment without mail. Where the deployment publishes terms, choosing that
password requires accepting them, as signup does. Accepting without a password
records no acceptance, so the provider sign-in that follows it asks for the terms. An address that can already sign in, by password or
through a provider, just accepts; the link cannot set or replace a password on
it. Neither can it for an address another organization has already added: the
inviter holds the link too, so that person accepts without a password and then
signs in through a provider or resets their password, either of which proves the
address.

An invitee does not need the link. Signing up with the invited address, or
signing in with Google or GitHub, completes the account the invitation created,
and every invitation still waiting on it is listed under Invitations in the
dashboard, where it can be accepted or declined. Someone who is not yet a member
of any organization is taken there straight after signing in.

A password chosen through a link counts as verified so the person can sign in,
but Otari records that an organization vouched for the address rather than that
the person proved it. A vouched address does not trigger
[email-domain auto-join](#email-domain-auto-join), and another organization
cannot add or invite it, since the vouching organization's admins could have
chosen its password. A later provider sign-in on it replaces the password with
the provider, and resetting the password by email also proves the address;
either lifts both restrictions.

A link stops working as soon as its membership stops waiting on it: when the
person is added directly, removed, or accepts or declines from their inbox, any
invitation still pending for them is closed. Revoking an invitation suspends
the membership only while it is still waiting; a person who became a member
another way keeps their access. An expired link says it expired every time it is
opened, and every refusal says to ask the inviter for a new link. An address
whose account a deployment operator deactivated cannot be invited or added until
it is reactivated, and an invitation already pending for it stops working until
then too; neither refusal gives the reason, since the admin asking may be
another tenant's. For the same reason, refusing an address another organization
vouched for says what would lift the refusal and nothing about that
organization. An organization name cannot carry line breaks or other control
characters, since it is quoted in invitation emails.

When a link may not set a password (the address belongs to someone another
organization still has as a member or invitee), the accept page says so before
the invitee accepts, and names the way in this deployment offers once they have:
a provider sign-in where one is configured, or "Forgot your password?" where mail
is, or else asking whoever administers the gateway. A password sent anyway is
refused with the same advice. Memberships that were revoked, declined, expired or
removed do not count.

Signing in answers which organization the session acts in. If the person was
removed from the one they last used, it is the oldest one they still belong to,
and if they belong to none it is empty; the dashboard then shows their
invitations, and offers to create an organization where `open_signup` is
enabled. Being removed, declining or having an invitation revoked moves the
person's active organization to the oldest one they still belong to. For
someone who belongs to none, `GET /api/v1/organizations/me` answers 404 saying
so, without naming an organization.

Invitation tokens are bearer credentials. Whoever holds an unused link can join
as the invited address, and choose its first password where the link allows one,
so send it only to that person.
Do not put tokens in logs or analytics. The browser validates and accepts them
through the public invitation endpoints. A link works once, and expires after
`invitation_expiry_hours`.

A signed-in person also sees the invitations addressed to them, and accepts or
declines one without a token: they are already authenticated as the addressee,
so the membership is addressed by id instead. Declining cancels the invitation
and suspends the paired membership, which is what stops the emailed link from
reviving it; a later invitation to the same address revives the membership.

## Email-domain auto-join

An owner or admin can claim an email domain so that colleagues join the
organization without being invited one at a time. It is the second way somebody
becomes a member, and unlike an invitation nobody decides about the individual,
so it is fenced accordingly.

A claim grants nothing on its own. Otari mints a verification token, the admin
publishes it as a TXT record at the claimed domain's apex, and only a claim whose
record Otari has found admits anyone:

```text
otari-domain-verification=<token>
```

Claiming is not exclusive; proving is. Several organizations may hold an unproven
claim on one domain and the first to publish the record takes it, which is what
stops a claim on a domain somebody else owns from locking out its real owner.

Once a domain is proven, anyone signing in with a **verified** address at that
domain joins as the role the claim names. The rules around that are deliberately
narrow:

- Only `member` and `viewer` can be handed out. Proving control of a domain is
  not a decision about a person, so it never confers organization management.
- An unverified address is skipped, so signing up on an address without reading
  the mail is not enough. So is one only vouched for through an invitation
  link, since any organization can invite any address.
- An existing membership is left alone. A suspended one is not revived by
  signing in, and an established role is not overwritten.
- The person's active organization is never changed by joining.
- Public email providers cannot be claimed.

A proof expires. Domains change hands, so a claim stops admitting anyone once its
proof passes the deployment's TTL, and an admin renews it by verifying again from
the Email domains page. Nothing re-reads DNS between verifications, so that TTL is
the window in which a transferred domain still admits people.
