"""What established an identity's address, and what that establishes about its holder.

``user.email_verified_at`` and ``user.email_proven_at`` are the two columns this answers for.
Both the identity domain and the tenancy services record addresses, so the vocabulary and the
writer sit below both rather than in either, beside the address handling they are about.
"""

from dataclasses import dataclass
from datetime import datetime

from gateway.models.identity import AddressAuthority
from gateway.models.tenancy import User


@dataclass(frozen=True)
class AddressStamp:
    """What establishing an address records on the two columns."""

    verified_at: datetime
    proven_at: datetime | None


def address_stamp(authority: AddressAuthority, *, now: datetime, verified_at: datetime | None) -> AddressStamp:
    """What this authority records, given the verification the identity already carries.

    The one place the rule lives, so a caller that writes the columns through a model and one
    that writes them through a statement cannot answer differently. ``verified_at`` is what the
    identity carries now, and None where it carries none.
    """
    return AddressStamp(
        verified_at=verified_at if verified_at is not None else now,
        proven_at=now if authority.proves_the_holder else None,
    )


def record_address(identity: User, *, now: datetime, authority: AddressAuthority) -> None:
    """Stamp that this authority established the identity's address.

    ``email_verified_at`` records the first verification and is never moved, so it stays the
    date the address became usable. ``email_proven_at`` is not monotonic.

    NOTE: An authority that does not prove the holder also retires an earlier proof, because
    whoever supplied the address can sign in from here on. A caller that establishes no
    address calls :func:`mark_address_usable` instead, which cannot retire one.
    """
    stamp = address_stamp(authority, now=now, verified_at=identity.email_verified_at)
    identity.email_verified_at = stamp.verified_at
    identity.email_proven_at = stamp.proven_at


def mark_address_usable(identity: User, *, now: datetime) -> None:
    """Stamp that an address the identity already held became usable, which establishes nothing.

    Gaining a way to sign in with an address says nothing about who receives mail there, so
    whoever proved it, or did not, is left as they were. This reaches ``email_proven_at``
    through no path at all, which is what separates it from :func:`record_address`.
    """
    if identity.email_verified_at is None:
        identity.email_verified_at = now
