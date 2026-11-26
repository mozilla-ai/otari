"""What each writer of the two address columns may do to them.

``email_verified_at`` says an address became usable and never moves once set.
``email_proven_at`` says somebody proved the holder controls it, and is not monotonic:
an authority that proves nothing retires an earlier proof, because whoever supplied the
address can sign in from there on.

Which of those a caller may reach is the distinction these tests pin. A caller that
establishes an address reaches both. A caller that only makes one usable reaches neither
a proof nor its retirement.
"""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.core.address_proof import mark_address_usable, record_address
from gateway.models.identity import AddressAuthority
from gateway.models.tenancy import User

PROVED_EARLIER = datetime(2026, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _account(*, verified_at: datetime | None = None, proven_at: datetime | None = None) -> User:
    account = User(email="ada@example.com", full_name="Ada Lovelace", is_active=True)
    account.email_verified_at = verified_at
    account.email_proven_at = proven_at
    return account


@pytest.mark.parametrize(
    "authority", [AddressAuthority.MAILBOX, AddressAuthority.OPERATOR, AddressAuthority.TRUSTED_PROVIDER]
)
def test_an_authority_that_proves_the_holder_records_a_proof(authority: AddressAuthority) -> None:
    account = _account()
    record_address(account, now=NOW, authority=authority)
    assert account.email_verified_at == NOW
    assert account.email_proven_at == NOW


@pytest.mark.parametrize("authority", [AddressAuthority.ORGANIZATION, AddressAuthority.SESSION])
def test_an_authority_that_proves_nothing_retires_an_earlier_proof(authority: AddressAuthority) -> None:
    """Whoever supplied the address can sign in from here on, so an older proof stops standing."""
    account = _account(verified_at=PROVED_EARLIER, proven_at=PROVED_EARLIER)
    record_address(account, now=NOW, authority=authority)
    assert account.email_proven_at is None


def test_a_verification_already_recorded_is_never_moved() -> None:
    """``email_verified_at`` is the date the address became usable, so a later call leaves it."""
    account = _account(verified_at=PROVED_EARLIER)
    record_address(account, now=NOW, authority=AddressAuthority.MAILBOX)
    assert account.email_verified_at == PROVED_EARLIER


@pytest.mark.parametrize("verified_at", [None, PROVED_EARLIER])
def test_making_an_address_usable_cannot_retire_a_proof(verified_at: datetime | None) -> None:
    """Separating the two functions is what makes this unconstructible rather than merely caught.

    The reachable case is the second: a provider proved the address and the identity holds no
    password, so a first password makes it usable without establishing anything.
    """
    account = _account(verified_at=verified_at, proven_at=PROVED_EARLIER)
    mark_address_usable(account, now=NOW)
    assert account.email_proven_at == PROVED_EARLIER


def test_making_an_address_usable_cannot_record_a_proof() -> None:
    account = _account()
    mark_address_usable(account, now=NOW)
    assert account.email_verified_at == NOW
    assert account.email_proven_at is None


def test_making_an_address_usable_leaves_an_earlier_verification_alone() -> None:
    account = _account(verified_at=PROVED_EARLIER, proven_at=PROVED_EARLIER)
    mark_address_usable(account, now=NOW + timedelta(days=1))
    assert account.email_verified_at == PROVED_EARLIER
    assert account.email_proven_at == PROVED_EARLIER
