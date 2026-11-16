"""Turns a refused email address into ``InvalidEmailError``."""

from gateway.core.addresses import stored_address
from gateway.exceptions.identity_exceptions import InvalidEmailError


def validated_email(email: str) -> str:
    """Normalize an address to lower case, refusing one that cannot be a handle.

    Raises ``InvalidEmailError`` whether the refusal is on shape or on stored length:
    from the caller's side "this cannot be your address" is one answer,
    and splitting it would publish which of the two rules a probe tripped.
    """
    candidate = stored_address(email)
    if candidate is None:
        raise InvalidEmailError(email)
    return candidate


__all__ = ["validated_email"]
