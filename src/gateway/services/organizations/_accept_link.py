"""The link an invitation email sends its invitee to accept the invitation."""

from gateway.services.mail import Mailer


def accept_link(mailer: Mailer, token: str) -> str:
    """Return the link that accepts the invitation ``token`` names, through ``mailer``.

    The link is absolute where the deployment knows its own address, and relative otherwise.
    """
    return mailer.link(f"/#/accept-invitation?token={token}")
