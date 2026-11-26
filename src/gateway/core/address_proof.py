"""What established an identity's address, and what that establishes about its holder.

``user.email_verified_at`` and ``user.email_proven_at`` are the two columns this answers for,
and the only module that writes either.

NOTE: This holds a rule of the identity domain, which `DOMAINS.md` says `core/` must not.
It sits here while the services that record an address are split across two packages, one
of which is not a domain. It belongs in the identity domain once that split lands, and the
list of modules that may write these columns is deliberately not a baseline so that move
costs an edit. Nothing else should take this placement as a pattern.
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

    ``verified_at`` is the verification the identity carries now, and None where it carries none.
    """
    return AddressStamp(
        verified_at=verified_at if verified_at is not None else now,
        proven_at=now if authority.proves_the_holder else None,
    )


def address_values(authority: AddressAuthority, *, now: datetime) -> dict[str, datetime | None]:
    """What this authority records on an identity whose address nothing has established yet.

    For a caller that writes through a statement rather than a model, so the columns are named
    here and not there. A proof absent from the result is absent rather than null, which leaves
    a proof the statement did not expect to meet as it was.
    """
    stamp = address_stamp(authority, now=now, verified_at=None)
    values: dict[str, datetime | None] = {"email_verified_at": stamp.verified_at}
    if stamp.proven_at is not None:
        values["email_proven_at"] = stamp.proven_at
    return values


def record_address(identity: User, *, now: datetime, authority: AddressAuthority) -> None:
    """Stamp that this authority established the identity's address.

    ``email_verified_at`` records the first verification and is never moved, so it stays the
    date the address became usable. ``email_proven_at`` is not monotonic.

    NOTE: An authority that does not prove the holder also retires an earlier proof, because
    whoever supplied the address can sign in from here on. A caller that establishes no address
    must call :func:`mark_address_usable`, which cannot retire one.
    """
    stamp = address_stamp(authority, now=now, verified_at=identity.email_verified_at)
    identity.email_verified_at = stamp.verified_at
    identity.email_proven_at = stamp.proven_at


def mark_address_usable(identity: User, *, now: datetime) -> None:
    """Stamp that an address the identity already held became usable, which establishes nothing.

    Gaining a way to sign in with an address says nothing about who receives mail there, so
    whoever proved it, or did not, is left as they were.
    """
    if identity.email_verified_at is None:
        identity.email_verified_at = now
