"""Random tokens for links that carry a credential, and the hash that is stored in place of a token."""

import hashlib
import secrets

# This prefix marks a token for a link that an operator hands on.
# The hash of a token covers the prefix, so a token with the prefix added or removed matches no stored hash.
# ``token_urlsafe`` never produces a ``.``, so a token that was made without the prefix never starts with it.
_SHARED_TOKEN_PREFIX = "shared."


def generate_token() -> str:
    """Return a new random token."""
    return secrets.token_urlsafe(32)


def generate_shared_token() -> str:
    """Return a new random token for a link that another person hands to the recipient."""
    return f"{_SHARED_TOKEN_PREFIX}{generate_token()}"


def is_shared_token(token: str) -> bool:
    """Return whether ``token`` came from ``generate_shared_token``.

    Such a token does not prove that its holder receives email at the recipient's address.
    """
    return token.startswith(_SHARED_TOKEN_PREFIX)


def hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of ``token``."""
    return hashlib.sha256(token.encode()).hexdigest()


def format_expiry(hours: int) -> str:
    """Return ``hours`` as text for a recipient, in days only when the hours are a whole number of days.

    The text never says a longer time than ``hours``.
    """
    if hours < 24 or hours % 24:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    days = hours // 24
    return f"{days} day{'s' if days != 1 else ''}"


__all__ = [
    "format_expiry",
    "generate_shared_token",
    "generate_token",
    "hash_token",
    "is_shared_token",
]
