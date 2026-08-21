"""The copy of the email-verification message.

The mechanics (layout, escaping, header sanitization, the one-pass fill) belong
to ``services.mail``; what lives here is this message's own values and the
wording around them, the same split ``invitation_email`` makes for its message.
"""

from gateway.services.mail import MailMessage, render_email
from gateway.services.tenancy.tokens import format_expiry


def render_verification_email(*, verify_link: str, expiry_hours: int) -> MailMessage:
    """Render the email-verification message for one recipient."""
    return render_email(
        "verify_email",
        subject="Verify your email for Otari",
        values={
            "VERIFY_LINK": verify_link,
            "VALID_PERIOD": format_expiry(expiry_hours),
        },
    )


__all__ = ["render_verification_email"]
