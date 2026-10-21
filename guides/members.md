# Members and invitations

Owners and admins invite people to an organization and choose which workspaces
each person can use. The roles are described in
[Access control](access-control.md#organizations-and-workspaces), and the rules
an invitation follows (mail limits, link expiry, first passwords, vouched
addresses) in [Invitations](access-control.md#invitations).

## Invite a member

1. Open **Organization** and select **Members & roles**.
2. Click **Invite member**.
3. Enter one or more **Email addresses**, separated by commas or new lines.
4. Select the **Role**: Owner, Admin, Member or Viewer.
5. Choose the **Workspaces** the person should have access to.
6. Click **Invite member**. Where the deployment sends mail, the person is
   emailed an accept link; otherwise the dialog gives you the link to share. The
   membership becomes active once they accept it.

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
it, are in [Email-domain auto-join](access-control.md#email-domain-auto-join).

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
