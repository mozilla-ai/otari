"""A provider identity holds only a provider and subject the link table can store."""

import pytest

from gateway.models.identity import (
    MAX_EMAIL_LENGTH,
    MAX_PROVIDER_LENGTH,
    MAX_SUBJECT_LENGTH,
    EmailAssertion,
    ProviderIdentity,
)


def _identity(*, provider: str = "google", subject: str = "a-subject") -> ProviderIdentity:
    return ProviderIdentity(
        provider=provider, subject=subject, email=None, full_name=None, email_verified=EmailAssertion.UNASSERTED
    )


def test_the_longest_provider_and_subject_the_table_stores_are_accepted() -> None:
    identity = _identity(provider="p" * MAX_PROVIDER_LENGTH, subject="s" * MAX_SUBJECT_LENGTH)

    assert len(identity.subject) == MAX_SUBJECT_LENGTH


@pytest.mark.parametrize(
    ("provider", "subject"),
    [
        ("", "a-subject"),
        ("google", ""),
        ("p" * (MAX_PROVIDER_LENGTH + 1), "a-subject"),
        ("google", "s" * (MAX_SUBJECT_LENGTH + 1)),
    ],
    ids=["empty-provider", "empty-subject", "long-provider", "long-subject"],
)
def test_an_identity_the_table_cannot_store_is_refused(provider: str, subject: str) -> None:
    with pytest.raises(ValueError, match="must hold 1 to"):
        _identity(provider=provider, subject=subject)


def test_an_address_the_table_cannot_store_is_refused() -> None:
    with pytest.raises(ValueError, match="email must hold at most"):
        ProviderIdentity(
            provider="google",
            subject="a-subject",
            email="a" * (MAX_EMAIL_LENGTH + 1),
            full_name=None,
            email_verified=EmailAssertion.VERIFIED,
        )
