"""The scope list and the rules for which scopes a key may hold."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from gateway.types.key_scope import (
    DEFAULT_KEY_SCOPES,
    END_USER_SCOPES,
    UNSCOPED_KEY_SCOPES,
    KeyScope,
    checked_scopes,
    known_scopes,
    normalize_scopes,
    unscoped_key_scopes,
    updated_scopes,
)

_MIGRATION = Path(__file__).parents[2] / "alembic" / "versions" / "c3f8a1d6e2b9_add_api_key_scopes.py"


def _migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("add_api_key_scopes", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_keys_call_models_and_nothing_else() -> None:
    assert DEFAULT_KEY_SCOPES == (KeyScope.INFERENCE,)


def test_scopes_are_stored_once_each_in_one_order() -> None:
    assert normalize_scopes(["usage:read", KeyScope.INFERENCE, "usage:read"]) == ["inference", "usage:read"]


def test_an_unknown_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="admin"):
        normalize_scopes(["admin"])


def test_a_key_needs_a_scope() -> None:
    with pytest.raises(ValueError, match="at least one scope"):
        checked_scopes([], is_service_key=False)


@pytest.mark.parametrize("scope", sorted(END_USER_SCOPES))
def test_only_a_service_key_holds_an_end_user_scope(scope: KeyScope) -> None:
    with pytest.raises(ValueError, match=f"Only a service key may hold {scope.value}"):
        checked_scopes([KeyScope.INFERENCE, scope], is_service_key=False)

    assert checked_scopes([scope], is_service_key=True) == [scope.value]


def test_a_key_from_before_scopes_holds_what_it_could_reach() -> None:
    """The frozen list is what the migration granted, so a key declared without scopes matches a migrated one."""
    migration = _migration()

    assert unscoped_key_scopes(is_service_key=False) == migration._KEY_SCOPES
    assert unscoped_key_scopes(is_service_key=True) == migration._SERVICE_KEY_SCOPES
    assert normalize_scopes(UNSCOPED_KEY_SCOPES) == migration._SERVICE_KEY_SCOPES


def test_no_scope_lets_a_key_manage_keys() -> None:
    """A key that could mint keys would outlive its own revocation; see ``types/key_scope.py``."""
    assert not [scope for scope in KeyScope if scope.value.startswith("keys")]


def test_an_update_checks_scopes_against_what_the_key_becomes() -> None:
    granted = updated_scopes(["inference"], ["inference", "end_users:read"], was_service_key=False, is_service_key=True)

    assert granted == ["inference", "end_users:read"]


def test_an_update_naming_no_scopes_leaves_them() -> None:
    assert updated_scopes(["inference", "web"], None, was_service_key=False, is_service_key=False) == [
        "inference",
        "web",
    ]
    service = ["inference", "end_users:read"]
    assert updated_scopes(service, None, was_service_key=False, is_service_key=True) == service


def test_a_key_that_stops_being_a_service_key_loses_its_end_user_scopes() -> None:
    held = updated_scopes(["inference", "end_users:delete"], None, was_service_key=True, is_service_key=False)

    assert held == ["inference"]


def test_a_service_key_holding_only_end_user_scopes_cannot_simply_stop_being_one() -> None:
    with pytest.raises(ValueError, match="at least one scope"):
        updated_scopes(["end_users:read"], None, was_service_key=True, is_service_key=False)


def test_a_scope_this_gateway_does_not_know_is_left_out_when_read() -> None:
    assert known_scopes(["inference", "from:a-newer-gateway"]) == [KeyScope.INFERENCE]
