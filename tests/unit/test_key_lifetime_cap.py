"""The ``max_api_key_lifetime_days`` clamp on a key's requested expiry.

The route behavior lives with the routes (``tests/integration/test_max_key_lifetime.py``);
this pins the pure arithmetic: what is stored, when the answer counts as clamped, and the
refusal when the ceiling has already passed.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.exceptions.api_keys_exceptions import KeyLifetimeCapPassedError
from gateway.services.api_keys import ApiKeyService

_CREATED = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


def _config(days: int | None) -> GatewayConfig:
    return GatewayConfig(max_api_key_lifetime_days=days)


def test_no_cap_leaves_the_ask_untouched() -> None:
    requested = _CREATED + timedelta(days=3650)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(None), requested, created_at=_CREATED, now=_CREATED)
    assert stored == requested
    assert capped_to is None


def test_no_cap_leaves_an_unbounded_key_unbounded() -> None:
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(None), None, created_at=_CREATED, now=_CREATED)
    assert stored is None
    assert capped_to is None


def test_an_expiry_within_the_cap_is_stored_as_the_instant_it_names() -> None:
    requested = _CREATED + timedelta(days=10)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), requested, created_at=_CREATED, now=_CREATED)
    assert stored == requested
    assert capped_to is None


def test_an_expiry_at_the_cap_exactly_is_within_it() -> None:
    requested = _CREATED + timedelta(days=30)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), requested, created_at=_CREATED, now=_CREATED)
    assert stored == requested
    assert capped_to is None


def test_an_expiry_past_the_cap_is_clamped_to_it() -> None:
    requested = _CREATED + timedelta(days=365)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), requested, created_at=_CREATED, now=_CREATED)
    assert stored == _CREATED + timedelta(days=30)
    assert capped_to == _CREATED + timedelta(days=30)


def test_a_key_minted_without_an_expiry_gets_one_at_the_cap() -> None:
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(90), None, created_at=_CREATED, now=_CREATED)
    assert stored == _CREATED + timedelta(days=90)
    assert capped_to == stored


def test_a_naive_expiry_is_read_as_utc() -> None:
    """Pydantic parses an offset-less timestamp naive; it is stored as the UTC instant it names."""
    naive = (_CREATED + timedelta(days=365)).replace(tzinfo=None)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), naive, created_at=_CREATED, now=_CREATED)
    assert stored == _CREATED + timedelta(days=30)
    assert capped_to == _CREATED + timedelta(days=30)

    within = (_CREATED + timedelta(days=3)).replace(tzinfo=None)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), within, created_at=_CREATED, now=_CREATED)
    assert stored == within.replace(tzinfo=UTC)
    assert capped_to is None


def test_a_naive_created_at_is_read_as_utc() -> None:
    """SQLite hands ``created_at`` back naive; the ceiling must not shift with the process zone."""
    naive_created = _CREATED.replace(tzinfo=None)
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), None, created_at=naive_created, now=_CREATED)
    assert stored == _CREATED + timedelta(days=30)
    assert capped_to == stored


def test_a_cap_already_outlived_refuses_a_write_that_becomes_the_cap() -> None:
    """A cap the key has outlived refuses the write rather than storing an expiry that ends the key."""
    now = datetime.now(UTC)
    created = now - timedelta(days=100)
    with pytest.raises(KeyLifetimeCapPassedError):
        ApiKeyService.cap_key_expiry(_config(30), now + timedelta(days=1), created_at=created, now=now)
    # Clearing the expiry is also a write of the ceiling, so it is refused the same way.
    with pytest.raises(KeyLifetimeCapPassedError):
        ApiKeyService.cap_key_expiry(_config(30), None, created_at=created, now=now)


def test_a_ceiling_exactly_at_now_has_already_passed() -> None:
    now = datetime.now(UTC)
    with pytest.raises(KeyLifetimeCapPassedError):
        ApiKeyService.cap_key_expiry(_config(30), None, created_at=now - timedelta(days=30), now=now)


def test_the_callers_own_past_expiry_stands_after_the_cap_has_passed() -> None:
    """An ask already within the ceiling is the caller's own; only a write clamped to a past cap is refused."""
    now = datetime.now(UTC)
    created = now - timedelta(days=100)
    past = now - timedelta(days=90)  # past, and under this key's ceiling of creation plus 30 days
    stored, capped_to = ApiKeyService.cap_key_expiry(_config(30), past, created_at=created, now=now)
    assert stored == past
    assert capped_to is None


def test_the_cap_bounds_its_days() -> None:
    assert GatewayConfig().max_api_key_lifetime_days is None
    assert GatewayConfig(max_api_key_lifetime_days=1).max_api_key_lifetime_days == 1
    assert GatewayConfig(max_api_key_lifetime_days=36500).max_api_key_lifetime_days == 36500
    with pytest.raises(ValidationError):
        GatewayConfig(max_api_key_lifetime_days=0)
    # Past this, creation plus the cap leaves datetime.max and every capped write raises.
    with pytest.raises(ValidationError):
        GatewayConfig(max_api_key_lifetime_days=36501)
