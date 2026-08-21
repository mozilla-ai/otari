"""The copy of the password-reset message.

The mechanics (layout, escaping, header sanitization, the one-pass fill) belong
to ``services.mail``; what lives here is this message's own values and the
wording around them, the same split ``invitation_email`` makes for its message.
"""

from gateway.services.mail import MailMessage, render_email
from gateway.services.tenancy.tokens import format_expiry


def render_password_reset_email(*, reset_link: str, expiry_hours: int) -> MailMessage:
    """Render the password-reset message for one recipient."""
    return render_email(
        "password_reset",
        subject="Reset your Otari password",
        values={
            "RESET_LINK": reset_link,
            "VALID_PERIOD": format_expiry(expiry_hours),
        },
    )


__all__ = ["render_password_reset_email"]
