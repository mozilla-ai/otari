"""Renders the invitation email from the static templates in ``gateway/templates/email/``.

Loaded once at import time and filled with plain ``str.replace``, the same
approach ``gateway.root_page`` uses for the tutorial page: no templating
engine, because the two templates here are the whole of what needs rendering
(see ``mail_service`` for why that dependency isn't worth adding).

Placeholders are ``{{LIKE_THIS}}`` rather than bare words, so a substituted
value cannot collide with a later pass: with bare placeholders, an
organization named "Role Corp" would have its own name partly overwritten
when the ``ROLE`` pass ran after the ``ORGANIZATION_NAME`` one, since that
pass scans the whole document, including text an earlier pass just inserted.

The organization name and the inviter's name are operator-set strings (an
organization can be renamed to anything; a full name is free text), so the
HTML variant escapes them before substitution, and the one that reaches an
email header (the subject line) is also stripped of newlines: a raw one
there is either a malformed header (``email.errors.HeaderParseError`` at
send time) or, on a mail stack that tolerates it, a header-injection
vector. The plain-text body is not escaped, since there is no markup there
to break out of.
"""

from html import escape
from importlib import resources


def _load(name: str) -> str:
    return resources.files("gateway").joinpath(f"templates/email/{name}").read_text(encoding="utf-8")


_HTML_TEMPLATE = _load("invitation.html")
_TEXT_TEMPLATE = _load("invitation.txt")


def _sanitize_header_value(value: str) -> str:
    """Strip CR/LF so a value cannot inject or malform an email header.

    Otari never delivers a literal newline in a name or a role, so this is a
    hardening measure against unexpected input, not a feature: it changes
    nothing for the values this codebase actually produces.
    """
    return value.replace("\r", " ").replace("\n", " ")


def _format_expiry(hours: int) -> str:
    """A recipient-facing duration, rounded up so it never overstates how long a link lives.

    ``invitation_expiry_hours`` is not required to be a multiple of 24, and
    rounding down (``hours // 24``) would tell a recipient a 12-hour link is
    good for a day. Rounding up is the direction that is never a lie, and
    staying in hours below one day avoids "1 days" needing its own plural
    check.
    """
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = -(-hours // 24)  # ceiling division
    return f"{days} day{'s' if days != 1 else ''}"


def _fill(template: str, values: dict[str, str]) -> str:
    for placeholder, value in values.items():
        template = template.replace("{{" + placeholder + "}}", value)
    return template


def render_invitation_email(
    *,
    organization_name: str,
    inviter_name: str,
    role: str,
    accept_link: str,
    expiry_hours: int,
) -> tuple[str, str, str]:
    """Return ``(subject, html, text)`` for one invitation email."""
    subject_organization_name = _sanitize_header_value(organization_name)
    subject = f"You're invited to join {subject_organization_name} on Otari"

    valid_for = _format_expiry(expiry_hours)
    html = _fill(
        _HTML_TEMPLATE,
        {
            "ORGANIZATION_NAME": escape(organization_name),
            "INVITER_NAME": escape(inviter_name),
            "ROLE": escape(role),
            "ACCEPT_LINK": escape(accept_link),
            "VALID_DAYS": valid_for,
        },
    )
    text = _fill(
        _TEXT_TEMPLATE,
        {
            "ORGANIZATION_NAME": organization_name,
            "INVITER_NAME": inviter_name,
            "ROLE": role,
            "ACCEPT_LINK": accept_link,
            "VALID_DAYS": valid_for,
        },
    )

    return subject, html, text


__all__ = ["render_invitation_email"]
