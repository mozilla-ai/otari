"""Opaque bearer tokens, hashed at rest.

Shared by every otari flow that mails a link carrying a credential: an
invitation, an email verification, a password reset. ``secrets.token_urlsafe``
gives the caller a raw token exactly once, for the link; only its SHA-256 hex
ever reaches a column, the same treatment ``organization_service``'s
``_hash_invitation_token`` already gives an invitation token and
``dashboard_session_service`` gives a session token.

Deliberately not a JWT: a stored hash can be looked up, matched, and cleared
on first use, which is what makes a token single-use. A self-verifying token
cannot be revoked or consumed, only outlast its own expiry, which is why
otari-ai's password-reset token can be replayed until it expires.
"""

import hashlib
import secrets


def generate_token() -> str:
    """A fresh bearer token, handed to a caller once and never stored."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 hex of a bearer token; only the hash is ever stored."""
    return hashlib.sha256(token.encode()).hexdigest()


def format_expiry(hours: int) -> str:
    """A recipient-facing duration that never overstates how long a link lives.

    An expiry setting is not required to be a multiple of 24. Rounding an
    inexact remainder up to the next day (25 hours -> "2 days") overstates it
    exactly the way rounding down (12 hours -> "1 day") does, just in fewer
    cases; the fix for both is to only switch to days on an exact multiple,
    and stay in hours otherwise. Shared by every message that carries a
    link's expiry (an invitation, a verification link, a reset link) so the
    wording agrees across all three rather than being reimplemented per one.
    """
    if hours < 24 or hours % 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


__all__ = ["format_expiry", "generate_token", "hash_token"]
