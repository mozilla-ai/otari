"""The copy of one organization invitation email.

The mechanics (layout, escaping, header sanitization, the one-pass fill) belong
to ``services.mail``; what lives here is this message's own values and the
wording around them, which is tenancy's to own. A new message is a body template
pair plus a function this shape, not another renderer.
"""

from gateway.services.mail import MailMessage, render_email


def _format_expiry(hours: int) -> str:
    """A recipient-facing duration that never overstates how long a link lives.

    ``invitation_expiry_hours`` is not required to be a multiple of 24.
    Rounding an inexact remainder up to the next day (25 hours -> "2 days")
    overstates it exactly the way rounding down (12 hours -> "1 day") does,
    just in fewer cases; the fix for both is to only switch to days on an
    exact multiple, and stay in hours otherwise.
    """
    if hours < 24 or hours % 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


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
            "ACCEPT_LINK": accept_link,
            "VALID_DAYS": _format_expiry(expiry_hours),
        },
    )


__all__ = ["render_invitation_email"]
