"""Which sign-ins are evidence about the address an account already holds.

``user.email_proven_at`` is what domain auto-join and the add-or-invite refusals read to
decide that the holder of an address controls it. A returning sign-in may move it only on
evidence about that address, and each condition below is one way a sign-in supplies none.
"""

import pytest

from gateway.models.identity import EmailAssertion, ProviderIdentity, is_organization_connection_provider
from gateway.models.tenancy import User
from gateway.services.identity._service import _asserts_this_address

HELD = "ada@example.com"
TRUSTED = "google"
TENANT_CONNECTION = "oidc:https://sso.acme.example"


def _account(email: str | None = HELD) -> User:
    return User(email=email, full_name="Ada Lovelace", is_active=True)


def _signed_in(
    *,
    provider: str = TRUSTED,
    email: str | None = HELD,
    email_verified: EmailAssertion = EmailAssertion.VERIFIED,
) -> ProviderIdentity:
    """A trusted provider reporting the held address as verified, which each test varies one field of."""
    return ProviderIdentity(
        provider=provider,
        subject="ada-subject",
        email=email,
        full_name="Ada Lovelace",
        email_verified=email_verified,
    )


def test_a_trusted_provider_asserting_the_held_address_is_evidence() -> None:
    assert _asserts_this_address(_account(), _signed_in()) is True


def test_letter_case_and_surrounding_space_are_the_same_address() -> None:
    assert _asserts_this_address(_account(), _signed_in(email=" ADA@Example.com ")) is True


def test_a_connection_a_tenant_supplies_is_not_evidence_whatever_it_asserts() -> None:
    """The issuer is a tenant's own, so what it reports about an address is its own word."""
    assert is_organization_connection_provider(TENANT_CONNECTION), "must name a connection, not an unknown provider"
    signed_in = _signed_in(provider=TENANT_CONNECTION)
    assert signed_in.email == HELD
    assert signed_in.email_verified is EmailAssertion.VERIFIED
    assert _asserts_this_address(_account(), signed_in) is False


@pytest.mark.parametrize("assertion", [EmailAssertion.UNASSERTED, EmailAssertion.UNVERIFIED])
def test_an_address_the_provider_did_not_verify_is_not_evidence(assertion: EmailAssertion) -> None:
    assert _asserts_this_address(_account(), _signed_in(email_verified=assertion)) is False


def test_another_address_is_not_evidence_about_the_one_the_account_holds() -> None:
    assert _asserts_this_address(_account(), _signed_in(email="grace@example.com")) is False


def test_a_sign_in_carrying_no_address_is_not_evidence() -> None:
    assert _asserts_this_address(_account(), _signed_in(email=None)) is False


def test_an_account_holding_no_address_has_none_to_prove() -> None:
    assert _asserts_this_address(_account(email=None), _signed_in()) is False
