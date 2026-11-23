"""The ``max_api_key_lifetime_days`` clamp on a key's requested expiry.

The route behaviour lives with the routes (``tests/integration/test_max_key_lifetime.py``);
this pins the pure arithmetic: what is stored, and when the answer counts as clamped.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from gateway.api.routes._helpers import cap_key_expiry
from gateway.core.config import GatewayConfig

_CREATED = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)


def _config(days: int | None) -> GatewayConfig:
    return GatewayConfig(max_api_key_lifetime_days=days)


def test_no_cap_leaves_the_ask_untouched() -> None:
    requested = _CREATED + timedelta(days=3650)
    stored, capped_to = cap_key_expiry(_config(None), requested, created_at=_CREATED)
    assert stored == requested
    assert capped_to is None


def test_no_cap_leaves_an_unbounded_key_unbounded() -> None:
    stored, capped_to = cap_key_expiry(_config(None), None, created_at=_CREATED)
    assert stored is None
    assert capped_to is None


def test_an_expiry_within_the_cap_is_stored_as_it_arrived() -> None:
    requested = _CREATED + timedelta(days=10)
    stored, capped_to = cap_key_expiry(_config(30), requested, created_at=_CREATED)
    assert stored == requested
    assert capped_to is None


def test_an_expiry_at_the_cap_exactly_is_within_it() -> None:
    requested = _CREATED + timedelta(days=30)
    stored, capped_to = cap_key_expiry(_config(30), requested, created_at=_CREATED)
    assert stored == requested
    assert capped_to is None


def test_an_expiry_past_the_cap_is_clamped_to_it() -> None:
    requested = _CREATED + timedelta(days=365)
    stored, capped_to = cap_key_expiry(_config(30), requested, created_at=_CREATED)
    assert stored == _CREATED + timedelta(days=30)
    assert capped_to == _CREATED + timedelta(days=30)


def test_a_key_minted_without_an_expiry_gets_one_at_the_cap() -> None:
    stored, capped_to = cap_key_expiry(_config(90), None, created_at=_CREATED)
    assert stored == _CREATED + timedelta(days=90)
    assert capped_to == stored


def test_a_naive_expiry_is_read_as_utc() -> None:
    """Pydantic parses an offset-less timestamp naive; comparing it to an aware ceiling would raise."""
    naive = (_CREATED + timedelta(days=365)).replace(tzinfo=None)
    stored, capped_to = cap_key_expiry(_config(30), naive, created_at=_CREATED)
    assert stored == _CREATED + timedelta(days=30)
    assert capped_to == _CREATED + timedelta(days=30)

    within = (_CREATED + timedelta(days=3)).replace(tzinfo=None)
    stored, capped_to = cap_key_expiry(_config(30), within, created_at=_CREATED)
    assert stored == within
    assert capped_to is None


def test_a_naive_created_at_is_read_as_utc() -> None:
    """SQLite hands ``created_at`` back naive; the ceiling must not shift with the process zone."""
    naive_created = _CREATED.replace(tzinfo=None)
    stored, capped_to = cap_key_expiry(_config(30), None, created_at=naive_created)
    assert stored == _CREATED + timedelta(days=30)
    assert capped_to == stored


def test_a_cap_already_outlived_clamps_an_update_back_over_the_past() -> None:
    """A cap lowered after a key's mint binds that key's next expiry write, even expired."""
    created = datetime.now(UTC) - timedelta(days=100)
    stored, capped_to = cap_key_expiry(_config(30), datetime.now(UTC) + timedelta(days=1), created_at=created)
    assert stored is not None
    assert stored < datetime.now(UTC)
    assert capped_to == stored


def test_the_cap_takes_at_least_one_day() -> None:
    assert GatewayConfig().max_api_key_lifetime_days is None
    assert GatewayConfig(max_api_key_lifetime_days=1).max_api_key_lifetime_days == 1
    with pytest.raises(ValidationError):
        GatewayConfig(max_api_key_lifetime_days=0)
