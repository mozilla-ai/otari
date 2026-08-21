"""The password primitives underneath the dashboard sign-in.

Pure hashing, no database and no app: the flows that use these are covered in
``test_password_sign_in.py``. What is asserted here is the part that has to stay
true for the ``user.hashed_password`` column to keep meaning the same thing in
this edition and on the platform, and the refusals that would otherwise reach a
caller as a 500 from inside bcrypt.
"""

import pytest

from gateway.services.password_service import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    hash_password,
    verify_absent_password,
    verify_password,
)

PASSWORD = "correct horse battery"  # pragma: allowlist secret

# Produced by the platform's own ``passlib.CryptContext(schemes=["bcrypt"])``
# hashing the value above. Checked in as a fixture rather than generated, since
# the point is that a hash this tree did not write still verifies: an identity
# imported from otari.ai has to be able to sign in here.
PLATFORM_HASH = "$2b$12$wTYK0ksNxWxM2L0PRLcSL.V6H6lO8BVfjeAvhXJWn7oWO27ZVee5a"  # pragma: allowlist secret


def test_a_hash_round_trips_and_a_wrong_password_does_not() -> None:
    hashed = hash_password(PASSWORD)

    assert verify_password(PASSWORD, hashed)
    assert not verify_password(PASSWORD + "!", hashed)


def test_two_hashes_of_one_password_differ() -> None:
    """Salted, so a stolen table does not group the people who chose one password."""
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_the_stored_format_is_the_platform_s() -> None:
    """``$2b$`` at cost 12, which is what passlib writes and what #1644 will import."""
    assert hash_password(PASSWORD).startswith("$2b$12$")


def test_a_hash_written_by_the_platform_verifies_here() -> None:
    assert verify_password(PASSWORD, PLATFORM_HASH)
    assert not verify_password("something else", PLATFORM_HASH)


def test_an_over_length_candidate_is_refused_rather_than_raising() -> None:
    """bcrypt raises past 72 bytes; a sign-in attempt must be a 401, not a 500."""
    hashed = hash_password(PASSWORD)

    assert not verify_password("x" * (MAX_PASSWORD_BYTES + 1), hashed)


def test_a_stored_value_that_is_not_a_bcrypt_hash_verifies_nothing() -> None:
    """A hand-edited column must lock the account, not crash every attempt on it."""
    assert not verify_password(PASSWORD, "not-a-hash")
    assert not verify_password(PASSWORD, "")


def test_the_stand_in_hash_never_matches() -> None:
    """It exists to cost time where there is no stored hash, and to answer False."""
    assert not verify_absent_password(PASSWORD)
    assert not verify_absent_password("")


@pytest.mark.parametrize("length", [MIN_PASSWORD_LENGTH, MAX_PASSWORD_BYTES])
def test_the_published_bounds_are_both_hashable(length: int) -> None:
    """Neither end of the advertised range may be a value bcrypt then refuses."""
    password = "a" * length

    assert verify_password(password, hash_password(password))
