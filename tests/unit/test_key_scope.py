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
    KeyScopeSet,
)

_MIGRATION = Path(__file__).parents[2] / "alembic" / "versions" / "c3f8a1d6e2b9_add_api_key_scopes.py"


def _migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("add_api_key_scopes", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_keys_call_models_and_nothing_else() -> None:
    assert DEFAULT_KEY_SCOPES == (KeyScope.INFERENCE_RUN,)


def test_scopes_are_stored_once_each_in_one_order() -> None:
    assert KeyScopeSet.checked(["usage:read", KeyScope.INFERENCE_RUN, "usage:read"], is_service_key=False).stored == [
        "inference:run",
        "usage:read",
    ]


def test_an_unknown_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="admin"):
        KeyScopeSet.checked(["admin"], is_service_key=False)


def test_a_key_needs_a_scope() -> None:
    with pytest.raises(ValueError, match="at least one scope"):
        KeyScopeSet.checked([], is_service_key=False)


@pytest.mark.parametrize("scope", sorted(END_USER_SCOPES))
def test_only_a_service_key_holds_an_end_user_scope(scope: KeyScope) -> None:
    with pytest.raises(ValueError, match=f"Only a service key may hold {scope.value}"):
        KeyScopeSet.checked([KeyScope.INFERENCE_RUN, scope], is_service_key=False)

    assert KeyScopeSet.checked([scope], is_service_key=True).stored == [scope.value]


def test_a_key_from_before_scopes_holds_what_it_could_reach() -> None:
    """The frozen list is what the migration granted, so a key declared without scopes matches a migrated one."""
    migration = _migration()

    assert KeyScopeSet.unscoped(is_service_key=False).stored == migration._KEY_SCOPES
    assert KeyScopeSet.unscoped(is_service_key=True).stored == migration._SERVICE_KEY_SCOPES
    assert KeyScopeSet.checked(UNSCOPED_KEY_SCOPES, is_service_key=True).stored == migration._SERVICE_KEY_SCOPES


def test_no_scope_lets_a_key_manage_keys() -> None:
    """A key that could mint keys would outlive its own revocation; see ``types/key_scope.py``."""
    assert not [scope for scope in KeyScope if scope.value.startswith("keys")]


def test_an_update_checks_scopes_against_what_the_key_becomes() -> None:
    stored = KeyScopeSet.checked(["inference:run"], is_service_key=False)

    granted = stored.updated(["inference:run", "end-users:read"], was_service_key=False, is_service_key=True)

    assert granted.stored == ["inference:run", "end-users:read"]


def test_an_update_naming_no_scopes_leaves_them() -> None:
    plain = KeyScopeSet.checked(["inference:run", "web:search"], is_service_key=False)
    assert plain.updated(None, was_service_key=False, is_service_key=False) == plain
    service = KeyScopeSet.checked(["inference:run", "end-users:read"], is_service_key=True)
    assert service.updated(None, was_service_key=False, is_service_key=True) == service


def test_a_key_that_stops_being_a_service_key_loses_its_end_user_scopes() -> None:
    service = KeyScopeSet.checked(["inference:run", "end-users:delete"], is_service_key=True)

    held = service.updated(None, was_service_key=True, is_service_key=False)

    assert held.stored == ["inference:run"]


def test_a_service_key_holding_only_end_user_scopes_cannot_simply_stop_being_one() -> None:
    with pytest.raises(ValueError, match="at least one scope"):
        KeyScopeSet.checked(["end-users:read"], is_service_key=True).updated(
            None, was_service_key=True, is_service_key=False
        )


def test_a_scope_this_gateway_does_not_know_is_left_out_when_read() -> None:
    assert list(KeyScopeSet.read(["inference:run", "from:a-newer-gateway"])) == [KeyScope.INFERENCE_RUN]
