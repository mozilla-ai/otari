"""Renders the invitation email from the static templates in ``gateway/templates/email/``.

Loaded once at import time and filled with plain ``str.replace``, the same
approach ``gateway.root_page`` uses for the tutorial page: no templating
engine, because the two templates here are the whole of what needs rendering
(see ``mail_service`` for why that dependency isn't worth adding).

The organization name and the inviter's name are operator-set strings (an
organization can be renamed to anything; a full name is free text), so the
HTML variant escapes them before substitution. The plain-text variant does
not, since there is no markup there to break out of.
"""

from html import escape
from importlib import resources


def _load(name: str) -> str:
    return resources.files("gateway").joinpath(f"templates/email/{name}").read_text(encoding="utf-8")


_HTML_TEMPLATE = _load("invitation.html")
_TEXT_TEMPLATE = _load("invitation.txt")


def render_invitation_email(
    *,
    organization_name: str,
    inviter_name: str,
    role: str,
    accept_link: str,
    valid_days: int,
) -> tuple[str, str, str]:
    """Return ``(subject, html, text)`` for one invitation email."""
    subject = f"You're invited to join {organization_name} on Otari"

    html = _HTML_TEMPLATE
    for placeholder, value in {
        "ORGANIZATION_NAME": organization_name,
        "INVITER_NAME": inviter_name,
        "ROLE": role,
        "ACCEPT_LINK": accept_link,
    }.items():
        html = html.replace(placeholder, escape(value))
    html = html.replace("VALID_DAYS", str(valid_days))

    text = _TEXT_TEMPLATE
    for placeholder, value in {
        "ORGANIZATION_NAME": organization_name,
        "INVITER_NAME": inviter_name,
        "ROLE": role,
        "ACCEPT_LINK": accept_link,
    }.items():
        text = text.replace(placeholder, value)
    text = text.replace("VALID_DAYS", str(valid_days))

    return subject, html, text


__all__ = ["render_invitation_email"]
