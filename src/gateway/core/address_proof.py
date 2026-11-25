"""What established an identity's address, and what that establishes about its holder.

``user.email_verified_at`` and ``user.email_proven_at`` are the two columns this answers for.
Both the identity domain and the tenancy services record addresses, so the vocabulary and the
writer sit below both rather than in either, beside the address handling they are about.
"""

from datetime import datetime

from gateway.models.identity import AddressAuthority
from gateway.models.tenancy import User


def record_address(identity: User, *, now: datetime, authority: AddressAuthority) -> None:
    """Stamp that this authority established the identity's address.

    ``email_verified_at`` records the first verification and is never moved, so it stays the
    date the address became usable. ``email_proven_at`` is not monotonic.

    NOTE: An authority that does not prove the holder also retires an earlier proof, because
    whoever supplied the address can sign in from here on. A caller that establishes no
    address calls :func:`mark_address_usable` instead, which cannot retire one.
    """
    if identity.email_verified_at is None:
        identity.email_verified_at = now
    identity.email_proven_at = now if authority.proves_the_holder else None


def mark_address_usable(identity: User, *, now: datetime) -> None:
    """Stamp that an address the identity already held became usable, which establishes nothing.

    Gaining a way to sign in with an address says nothing about who receives mail there, so
    whoever proved it, or did not, is left as they were. This reaches ``email_proven_at``
    through no path at all, which is what separates it from :func:`record_address`.
    """
    if identity.email_verified_at is None:
        identity.email_verified_at = now
