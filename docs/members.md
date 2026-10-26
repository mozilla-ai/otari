# Members and invitations

Owners and admins invite people to an organization and choose which workspaces
each person can use. The roles are described in
[Roles](organizations-and-workspaces.md#roles), and the rules an invitation
follows (mail limits, link expiry, first passwords, vouched addresses) in
[Invitations](#invitations) below.

## Invite a member

1. Open **Organization** and select **Members & roles**.
2. Click **Invite member**.
3. Enter one or more **Email addresses**, separated by commas or new lines.
4. Select the **Role**: Owner, Admin, Member or Viewer.
5. Choose the **Workspaces** the person should have access to.
6. Click **Invite member**. The person gets an accept link by email, or the
   dialog gives you the link to share when no email goes out. The membership
   becomes active once they accept it.

To invite several people at once, paste up to 100 addresses into the dialog.
Everyone gets the same role and workspaces. Each address is invited or refused
on its own, so one that is already a member does not stop the rest, and the
result lists every address with whether its email went out, or its accept link
to share when it did not.

## Manage existing members

On **Members & roles**, the **Role** column changes a member's organization
role, a member's **Edit** action changes the workspaces they can access and
their role in each, and **Remove** takes them out of the organization.

An invited row says when its invitation has expired, and offers:

- **Reissue**, which sends a new accept link (or shows it to share, when the
  email does not go out). The old link stops working, which is how a lost link
  is recovered.
- **Revoke**, which withdraws the invitation.

Inviting an address whose invitation is still pending is refused; reissue it
instead.

## Let colleagues join by email domain

An owner or admin can claim an email domain, so that colleagues with a verified
address there join without an invitation. What a claim grants, and the limits on
it, are in [Email-domain auto-join](#email-domain-auto-join).

1. On the **Email domains** page, claim the domain and choose the role it grants
   (`member` or `viewer`). Otari mints a verification token.
2. Publish the token as a TXT record at the claimed domain's apex:

   ```text
   otari-domain-verification=<token>
   ```

3. Verify the claim from the **Email domains** page. Once Otari finds the
   record, the domain is proven and starts admitting verified addresses.
4. Verify again before the proof's TTL passes to keep the claim admitting
   people.

## How it works

### Invitations

An owner or admin can invite a person to an organization and selected workspaces.
If mail is configured, Otari emails the accept link. When the email does not go
out (no mail configured, the send failed, or a mail limit below was reached), the
dashboard shows the link instead, so it can be shared by hand; Reissue on a
pending or expired invitation does the same with a new link. A link that was
emailed is never shown to the inviter, so only the invitee's mailbox holds it. The
email names the inviter by their address as well as the name they chose, and an
address the mail server would refuse, such as one ending in a dot, is refused
when it is invited.

Invitation email is limited, because it carries text an organization chose (its
name, the inviter's name) from the deployment's own sender, and on a deployment
where anyone can create an organization that would otherwise make it an open
relay. Over any 24 hours an organization can send only so many invitation
emails, and an address can receive only so many from every organization
together, so revoking and inviting again cannot flood it; the settings and their
defaults are under [Mail](configuration.md#mail). An invitation over a limit is
still created, and its link is returned to share by hand, as when mail is off:
the link costs the deployment nothing, and refusing would stop a legitimate
admin partway through a bulk invite. The counts are kept in `rate_limit_store`,
so with Redis they hold across replicas.

The API can also add a member directly (`POST /api/v1/organizations/me/members`),
active at once with no invitation to accept and no email. A new address added
that way gets in by signing up, by "Forgot your password?" or through a provider,
so on a deployment with no mail and no provider it waits for an operator's
password link. Invite it instead, since the invitation link itself can set its
first password.

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

### Email-domain auto-join

A claimed email domain is the second way somebody becomes a member, and unlike an invitation nobody decides about the individual,
so it is fenced accordingly.

A claim grants nothing on its own: only a claim whose verification record
Otari has found admits anyone.

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
