"""What established an identity's address, and what that establishes about its holder.

``user.email_verified_at`` and ``user.email_proven_at`` are the two columns this answers for.
Both the identity domain and the tenancy services record addresses, so the vocabulary and the
writer sit below both rather than in either, beside the address handling they are about.
"""

from datetime import datetime
from enum import StrEnum

from gateway.models.tenancy import User


class AddressAuthority(StrEnum):
    """What established an identity's sign-in address.

    ``proves_the_holder`` is what ``email_proven_at`` records. An authority that could assert
    any address answers false, because its word is not evidence that the holder controls it.
    """

    MAILBOX = ("mailbox", True)
    """A link that reached the inbox, so whoever opened it receives mail at the address."""

    OPERATOR = ("operator", True)
    """Deployment authority supplied it, which sits above every organization here."""

    ORGANIZATION = ("organization", False)
    """One organization's word for an address any organization could have claimed."""

    SESSION = ("session", False)
    """A dashboard session, which says who is calling and nothing about who receives mail at the address."""

    proves_the_holder: bool

    def __new__(cls, value: str, proves_the_holder: bool) -> "AddressAuthority":
        member = str.__new__(cls, value)
        member._value_ = value
        member.proves_the_holder = proves_the_holder
        return member


def record_address(identity: User, *, now: datetime, authority: AddressAuthority) -> None:
    """Stamp what established this identity's address, keeping a proof implied by a verification.

    ``email_verified_at`` records the first verification and is never moved, so it stays the
    date the address became usable. ``email_proven_at`` is not monotonic.

    NOTE: An authority that does not prove the holder also retires an earlier proof, because
    whoever supplied the address can sign in from here on.
    """
    if identity.email_verified_at is None:
        identity.email_verified_at = now
    identity.email_proven_at = now if authority.proves_the_holder else None
