"""Which proof of an address a signup's invitation token carries."""

import pytest

from gateway.api.routes.auth_signup import _address_proof
from gateway.core.tokens import generate_shared_token, generate_token
from gateway.services.tenancy import AddressProof


def test_an_emailed_invitation_proves_the_mailbox_answers() -> None:
    assert _address_proof("ada@example.com", "Ada@Example.com ", token=generate_token()) is AddressProof.MAILBOX


def test_a_hand_carried_invitation_only_vouches_for_the_address() -> None:
    """Nobody proved the mailbox answers, so the account is marked as vouched for rather than verified by mail."""
    proof = _address_proof("ada@example.com", "ada@example.com", token=generate_shared_token())

    assert proof is AddressProof.VOUCHED


def test_an_invitation_to_another_address_proves_nothing() -> None:
    assert _address_proof("ada@example.com", "bob@example.com", token=generate_token()) is None


def test_no_token_and_no_invitation_prove_nothing() -> None:
    assert _address_proof(None, "ada@example.com", token=generate_token()) is None
    assert _address_proof("ada@example.com", "ada@example.com", token=None) is None


@pytest.mark.parametrize(
    ("invited", "claimed"),
    [
        ("not an address", "not an address"),
        ("Ada <ada@example.com>", "Ada <ada@example.com>"),
        ("ada@example.com, bob@example.com", "ada@example.com, bob@example.com"),
    ],
)
def test_two_values_that_are_not_addresses_do_not_prove_each_other(invited: str, claimed: str) -> None:
    """Both normalize to nothing, and nothing must not match nothing."""
    assert _address_proof(invited, claimed, token=generate_token()) is None
