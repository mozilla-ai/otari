"""The copy of one organization invitation email.

The mechanics (layout, escaping, header sanitization, the one-pass fill) belong
to ``services.mail``; what lives here is this message's own values and the
wording around them, which is tenancy's to own. A new message is a body template
pair plus a function this shape, not another renderer.
"""

from gateway.services.mail import MailMessage, render_email
from gateway.services.tenancy.tokens import format_expiry


def _article_for(role: str) -> str:
    """``a`` or ``an`` for the role named in the invitation copy.

    Half the roles need "an": ``OrganizationMemberRole`` is the closed set
    ``owner``/``admin``/``member``/``viewer``, and the first two begin with a
    vowel, so hard-coding "a" made every owner and admin invitation read "as a
    admin". The vowel rule is correct for all four; it is only a heuristic for a
    role that does not exist yet, which is a better failure than a wrong article
    on half the invitations this deployment actually sends.
    """
    return "an" if role[:1].lower() in "aeiou" else "a"


def render_invitation_email(
    *,
    organization_name: str,
    inviter_name: str,
    role: str,
    accept_link: str,
    expiry_hours: int,
) -> MailMessage:
    """Render the invitation message for one recipient."""
    return render_email(
        "invitation",
        subject="You're invited to join {{ORGANIZATION_NAME}} on Otari",
        values={
            "ORGANIZATION_NAME": organization_name,
            "INVITER_NAME": inviter_name,
            "ROLE": role,
            "ROLE_ARTICLE": _article_for(role),
            "ACCEPT_LINK": accept_link,
            "VALID_DAYS": format_expiry(expiry_hours),
        },
    )


__all__ = ["render_invitation_email"]
