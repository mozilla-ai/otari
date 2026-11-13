"""The email that invites a person to join an organization."""

import unicodedata
from datetime import UTC, datetime

from gateway.services.mail import MailMessage, render_email
from gateway.services.organizations._invitation_facts import InvitationFacts

# The wording of the email that is not in a template, A to Z.
_CHECKED_HEADING = "Checked by Otari"
_DOMAIN_PROVEN = "This organization controls the email domain {domain}."
_EXPIRY = "The invitation expires on {day} {month} {year} at {hour:02}:{minute:02} UTC."
_INVITER_NOT_VERIFIED = "The inviter has not verified an email address."
_INVITER_VERIFIED = "The inviter has verified the email address {email}."
_NO_INVITER_NAME = "An organization admin"
_NO_ORGANIZATION_NAME = "an organization"
_NO_PROVEN_DOMAIN = "This organization has not proven a domain. Check that you know the person who invited you."
_WORKSPACES_HEADING = "Workspaces, as the organization named them:"

# Letters that show nothing.
_BLANK_LETTERS = frozenset("\u115f\u1160\u2800\u3164\uffa0")
# The zero width non-joiner and the zero width joiner, which some scripts and emoji need.
_JOINERS = frozenset("\u200c\u200d")
# English month names, January first.
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_NAME_LIMIT = 80
# The Unicode categories of characters that are not shown in a name: control, format, unassigned,
# private use, line separator and paragraph separator. The joiners are format characters that are kept.
_REMOVED_CATEGORIES = frozenset({"Cc", "Cf", "Cn", "Co", "Zl", "Zp"})
# The Unicode major categories of characters that show something: letter, number, punctuation and symbol.
_VISIBLE_CATEGORIES = frozenset("LNPS")


def _checked(facts: InvitationFacts) -> str:
    """Return the statements that Otari makes about an invitation, one on each line."""
    lines = [_DOMAIN_PROVEN.format(domain=domain) for domain in facts.proven_domains] or [_NO_PROVEN_DOMAIN]
    if facts.inviter_email is None:
        lines.append(_INVITER_NOT_VERIFIED)
    else:
        lines.append(_INVITER_VERIFIED.format(email=facts.inviter_email))
    return "\n".join(lines)


def _expiry(expires_at: datetime) -> str:
    if expires_at.tzinfo is None:
        msg = "The expiry of an invitation needs a time zone."
        raise ValueError(msg)
    at = expires_at.astimezone(UTC)
    return _EXPIRY.format(day=at.day, month=_MONTHS[at.month - 1], year=at.year, hour=at.hour, minute=at.minute)


def _is_removed(character: str) -> bool:
    if character in _JOINERS:
        return False
    return character in _BLANK_LETTERS or unicodedata.category(character) in _REMOVED_CATEGORIES


def _typed_name(name: str | None) -> str:
    """Return a name that a person typed as it is shown, or an empty string when it shows nothing.

    The name is shown on one line, without characters that hide text or change its layout.
    A name longer than ``_NAME_LIMIT`` characters is cut and ends in an ellipsis.
    """
    kept = "".join(character for character in name or "" if character.isspace() or not _is_removed(character))
    shown = " ".join(kept.split())
    if not any(unicodedata.category(character)[0] in _VISIBLE_CATEGORIES for character in shown):
        return ""
    if len(shown) <= _NAME_LIMIT:
        return shown
    return f"{shown[:_NAME_LIMIT].rstrip()}…"


def _workspaces(facts: InvitationFacts) -> str:
    """Return the paragraph that lists the workspaces, or an empty string when there is none to list."""
    names = [name for name in map(_typed_name, facts.workspace_names) if name]
    if not names:
        return ""
    return "\n".join([_WORKSPACES_HEADING, *(f"- {name}" for name in names), "", ""])


def render_invitation_email(
    facts: InvitationFacts, *, accept_link: str, expires_at: datetime, asset_root: str | None
) -> MailMessage:
    """Render the email that tells an invitee the facts of an invitation and how to accept it.

    The statements that Otari makes are in a block of their own, which holds no name that a person typed.
    A typed name is shown on one line, without characters that hide text or change its layout,
    and cut to ``_NAME_LIMIT`` characters.
    The expiry is stated as a time in UTC.
    Raise ``ValueError`` when ``expires_at`` has no time zone.
    """
    return render_email(
        "organization_invitation",
        subject="You're invited to join {{ORGANIZATION_NAME}} on Otari",
        preheader="{{INVITER}} invited you. {{EXPIRY}}",
        asset_root=asset_root,
        values={
            "ACCEPT_LINK": accept_link,
            "CHECKED": _checked(facts),
            "CHECKED_HEADING": _CHECKED_HEADING,
            "EXPIRY": _expiry(expires_at),
            "INVITER": _typed_name(facts.inviter_name) or _NO_INVITER_NAME,
            "ORGANIZATION_NAME": _typed_name(facts.organization_name) or _NO_ORGANIZATION_NAME,
            "ROLE": facts.role,
            "WORKSPACES": _workspaces(facts),
        },
    )
