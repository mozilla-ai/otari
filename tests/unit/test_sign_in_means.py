"""``User.sign_in_means``: the four exclusive states two credential columns give an identity."""

from datetime import UTC, datetime

import pytest

from gateway.models.tenancy import SignInMeans, User

_VERIFIED = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("hashed_password", "email_verified_at", "expected"),
    [
        (None, None, SignInMeans.NONE),
        ("hash", None, SignInMeans.UNVERIFIED_PASSWORD),
        ("hash", _VERIFIED, SignInMeans.PASSWORD),
        (None, _VERIFIED, SignInMeans.WITHOUT_PASSWORD),
    ],
)
@pytest.mark.parametrize("is_active", [True, False])
def test_sign_in_means_reads_the_password_and_the_verified_address(
    hashed_password: str | None, email_verified_at: datetime | None, expected: SignInMeans, is_active: bool
) -> None:
    """``is_active`` plays no part, so a caller that cares has to read it as well."""
    identity = User(
        email="someone@example.com",
        hashed_password=hashed_password,
        email_verified_at=email_verified_at,
        is_active=is_active,
    )

    assert identity.sign_in_means is expected


def test_sign_in_means_follows_the_columns_as_they_change() -> None:
    """A property rather than a stored value, so it never reads a state the row has left."""
    identity = User(email="someone@example.com")
    seen = [identity.sign_in_means]
    identity.hashed_password = "hash"
    seen.append(identity.sign_in_means)
    identity.email_verified_at = _VERIFIED
    seen.append(identity.sign_in_means)
    identity.hashed_password = None
    seen.append(identity.sign_in_means)

    assert seen == [
        SignInMeans.NONE,
        SignInMeans.UNVERIFIED_PASSWORD,
        SignInMeans.PASSWORD,
        SignInMeans.WITHOUT_PASSWORD,
    ]
