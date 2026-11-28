"""What each writer of the two address columns may do to them.

``email_verified_at`` says an address became usable and never moves once set.
``email_proven_at`` says somebody proved the holder controls it, and is not monotonic.

Which of those a caller may reach is the distinction these tests pin, and the three
writers differ, so a caller has to know which one it holds. ``record_address`` assigns
the proof, so a non-proving authority retires an earlier one. ``address_values`` omits
the column, so an earlier proof survives a statement that did not expect to meet one.
``mark_address_usable`` establishes nothing and can reach neither a proof nor its
retirement.
"""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.core.address_proof import address_values, mark_address_usable, record_address
from gateway.models.identity import AddressAuthority
from gateway.models.tenancy import User

PROVED_EARLIER = datetime(2026, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 6, 1, tzinfo=UTC)

PROVE_THE_HOLDER = [AddressAuthority.MAILBOX, AddressAuthority.OPERATOR, AddressAuthority.TRUSTED_PROVIDER]
PROVE_NOTHING = [AddressAuthority.ORGANIZATION, AddressAuthority.SESSION]


def _account(*, verified_at: datetime | None = None, proven_at: datetime | None = None) -> User:
    account = User(email="ada@example.com", full_name="Ada Lovelace", is_active=True)
    account.email_verified_at = verified_at
    account.email_proven_at = proven_at
    return account


def test_every_authority_is_covered_by_exactly_one_of_these_groups() -> None:
    """A member these tests do not name is a member whose rule nothing here pins.

    NOTE: An authority added to the enum must join one group or the other.
    """
    grouped = PROVE_THE_HOLDER + PROVE_NOTHING
    assert set(grouped) == set(AddressAuthority)
    assert len(grouped) == len(AddressAuthority)
    assert [member for member in PROVE_THE_HOLDER if not member.proves_the_holder] == []
    assert [member for member in PROVE_NOTHING if member.proves_the_holder] == []


@pytest.mark.parametrize("authority", PROVE_THE_HOLDER)
def test_an_authority_that_proves_the_holder_records_a_proof(authority: AddressAuthority) -> None:
    account = _account()
    record_address(account, now=NOW, authority=authority)
    assert account.email_verified_at == NOW
    assert account.email_proven_at == NOW


@pytest.mark.parametrize("authority", PROVE_NOTHING)
def test_an_authority_that_proves_nothing_retires_an_earlier_proof(authority: AddressAuthority) -> None:
    """Whoever supplied the address can sign in from here on, so an older proof stops standing."""
    account = _account(verified_at=PROVED_EARLIER, proven_at=PROVED_EARLIER)
    record_address(account, now=NOW, authority=authority)
    assert account.email_proven_at is None


@pytest.mark.parametrize("authority", PROVE_THE_HOLDER)
def test_a_statement_records_a_proof_for_an_authority_that_proves_the_holder(authority: AddressAuthority) -> None:
    assert address_values(authority, now=NOW) == {"email_verified_at": NOW, "email_proven_at": NOW}


@pytest.mark.parametrize("authority", PROVE_NOTHING)
def test_a_statement_names_no_proof_column_for_an_authority_that_proves_nothing(authority: AddressAuthority) -> None:
    """Naming it as None would retire a proof the statement did not expect to meet.

    The caller is reached with an authority a port supplies, so this holds for a member
    the caller never names itself.
    """
    assert address_values(authority, now=NOW) == {"email_verified_at": NOW}


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
