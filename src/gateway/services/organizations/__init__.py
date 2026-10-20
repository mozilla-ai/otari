"""The organizations domain: organizations, workspaces, members and invitations.

Most of it is still in ``services/tenancy``; code new to the domain lands here.
"""

from gateway.services.organizations._invitation_mail_limit import InvitationMailLimit

__all__ = ["InvitationMailLimit"]
