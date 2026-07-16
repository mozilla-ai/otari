"""Unit tests for module-level state reset helpers."""

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from gateway.api.deps import get_config, reset_config, set_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db


def _request_without_app_config() -> Request:
    """Build a minimal request whose app state has no config attached.

    Forces ``get_config`` down its legacy module-level fallback path so the
    compatibility shim (``set_config`` / ``reset_config``) can be exercised in
    isolation from ``app.state``.
    """
    return cast(Request, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())))


def test_reset_config_clears_state() -> None:
    config = GatewayConfig(
        database_url="postgresql://localhost/test",
        master_key="test",
    )
    request = _request_without_app_config()

    set_config(config)
    assert get_config(request) is config

    reset_config()

    with pytest.raises(RuntimeError, match="Config not initialized"):
        get_config(request)


def test_two_apps_hold_independent_configs() -> None:
    """Two apps in one process must not share a single config instance."""
    from gateway.main import create_app

    config_a = GatewayConfig(database_url="sqlite:///./a.db", master_key="key-a", auto_migrate=False)
    config_b = GatewayConfig(database_url="sqlite:///./b.db", master_key="key-b", auto_migrate=False)

    app_a = create_app(config_a)
    app_b = create_app(config_b)

    assert app_a.state.config is config_a
    assert app_b.state.config is config_b
    assert app_a.state.config is not app_b.state.config
    assert app_a.state.config.master_key == "key-a"
    assert app_b.state.config.master_key == "key-b"

    reset_config()


def test_reset_db_allows_reinit() -> None:
    from gateway.core import database

    reset_db()

    assert database._engine is None
    assert database._SessionLocal is None
