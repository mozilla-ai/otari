"""The copy of the messages a password reset or signup request sends.

The mechanics (layout, escaping, header sanitization, the one-pass fill) belong
to ``services.mail``; what lives here is each message's own values and the
wording around them, the same split ``invitation_email`` makes for its message.
"""

from gateway.services.mail import MailMessage, render_email
from gateway.services.tenancy.tokens import format_expiry


def render_password_reset_email(*, reset_link: str, expiry_hours: int, asset_root: str | None) -> MailMessage:
    """Render the password-reset message for one recipient."""
    return render_email(
        "password_reset",
        subject="Set or reset your Otari password",
        preheader="The link expires in {{VALID_PERIOD}} and can only be used once.",
        asset_root=asset_root,
        values={
            "RESET_LINK": reset_link,
            "VALID_PERIOD": format_expiry(expiry_hours),
        },
    )


def render_account_exists_email(
    *, sign_in_link: str, has_password: bool, provider: str | None, asset_root: str | None
) -> MailMessage:
    """Render the notice that the address already has an account, saying how it signs in.

    ``provider`` is the label of the provider the account signs in with, or ``None`` when that is not known.
    """
    if has_password:
        hint = (
            'Sign in with your password. If you have forgotten it, choose "Forgot your password?" on the '
            "sign-in page to get a link that sets a new one."
        )
    elif provider is not None:
        hint = (
            f"This account signs in with {provider} and has no password. Sign in with {provider}; once signed "
            "in, you can add a password from Account settings."
        )
    else:
        hint = (
            "This account has no password. Sign in the way you did before, with a provider or a passkey; once "
            "signed in, you can add a password from Account settings."
        )
    return render_email(
        "account_exists",
        subject="You already have an Otari account",
        values={"SIGN_IN_HINT": hint, "SIGN_IN_LINK": sign_in_link},
        preheader="This address already has an Otari account, so nothing was changed.",
        asset_root=asset_root,
    )


__all__ = ["render_account_exists_email", "render_password_reset_email"]
