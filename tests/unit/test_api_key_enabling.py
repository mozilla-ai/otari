"""Who can enable a disabled API key, and what a refused caller is told."""

import pytest

from gateway.exceptions.api_keys_exceptions import KeyEnableRefusedError
from gateway.models.api_keys import APIKey, DisabledReason
from gateway.services.api_keys._enabling import operator_enable_refusal, operator_set_active


def _key(reason: DisabledReason | None) -> APIKey:
    key = APIKey()
    key.is_active = reason is None
    key.disabled_reason = reason
    return key


def test_every_reason_states_whether_an_operator_can_clear_it() -> None:
    """A member declared without an answer does not construct, which is why it is on the member."""
    assert DisabledReason.OPERATOR.operator_may_clear is True
    assert DisabledReason.REVOKED.operator_may_clear is False


def test_an_operator_is_blocked_only_by_a_reason_they_cannot_clear() -> None:
    assert operator_enable_refusal(_key(None)) is None
    assert operator_enable_refusal(_key(DisabledReason.OPERATOR)) is None
    assert operator_enable_refusal(_key(DisabledReason.REVOKED)) is DisabledReason.REVOKED


def test_an_operator_enabling_a_key_clears_its_reason() -> None:
    key = _key(DisabledReason.OPERATOR)

    operator_set_active(key, True)

    assert key.is_active is True
    assert key.disabled_reason is None


def test_an_operator_disabling_a_key_records_their_reason() -> None:
    key = _key(None)

    operator_set_active(key, False)

    assert key.is_active is False
    assert key.disabled_reason is DisabledReason.OPERATOR


@pytest.mark.parametrize("is_active", [True, False])
def test_a_revoked_key_refuses_an_operator_either_way(is_active: bool) -> None:
    """A disable would otherwise replace the revocation with a reason an operator can clear."""
    key = _key(DisabledReason.REVOKED)

    with pytest.raises(KeyEnableRefusedError):
        operator_set_active(key, is_active)

    assert key.disabled_reason is DisabledReason.REVOKED


def test_the_refusal_names_an_operator_only_where_one_can_act() -> None:
    operator_reason = str(KeyEnableRefusedError(DisabledReason.OPERATOR))
    revoked = str(KeyEnableRefusedError(DisabledReason.REVOKED))

    assert "operator" in operator_reason
    # Chosen by clearability, so it must hold for any reason an operator cannot
    # clear, not only for the revocation that is the one such reason today.
    assert "operator" not in revoked
    assert "revoked" not in revoked
