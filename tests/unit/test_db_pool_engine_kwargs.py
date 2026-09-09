"""Unit tests for the database engine's pool and timeout configuration.

The gateway's whole request path starts with a database read, so a database
call that can hang without a deadline is a request that can hang without one.
These pin the settings that bound it, and the second pool that keeps metering
off the request pool's contention.
"""

from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.pool import NullPool

from gateway.core.config import GatewayConfig
from gateway.core.database import engine_kwargs


def _pg_kwargs(**overrides: Any) -> dict[str, Any]:
    config = GatewayConfig(**overrides)
    return engine_kwargs(config, connect_args={}, is_sqlite=False)


def test_postgres_pool_uses_configured_sizes() -> None:
    kwargs = _pg_kwargs()
    assert kwargs["pool_size"] == 10
    assert kwargs["max_overflow"] == 20
    assert kwargs["pool_timeout"] == 30.0
    assert kwargs["pool_pre_ping"] is True


def test_connections_are_recycled_by_default() -> None:
    # A managed database, or the NAT in front of it, drops an idle connection
    # without a FIN. Recycling retires one before it can be handed out in that
    # state; the pre-ping that would otherwise catch it is itself a statement,
    # and on a half-open socket it blocks rather than failing.
    assert _pg_kwargs()["pool_recycle"] == 1800


def test_pool_recycle_can_be_disabled() -> None:
    assert "pool_recycle" not in _pg_kwargs(db_pool_recycle=-1)


def test_postgres_connect_args_carry_every_timeout() -> None:
    args = _pg_kwargs()["connect_args"]
    assert args["timeout"] == 10.0
    assert args["command_timeout"] == 60.0
    assert args["server_settings"]["statement_timeout"] == "65000"


def test_the_server_side_backstop_fires_after_the_client_side_timeout() -> None:
    # Equal, whichever fires first is a race, and the two report differently.
    config = GatewayConfig()
    assert config.db_statement_timeout_ms > config.db_command_timeout * 1000


def test_a_backstop_that_would_race_is_rejected_at_config_load() -> None:
    with pytest.raises(ValidationError, match="db_statement_timeout_ms"):
        GatewayConfig(db_command_timeout=90, db_statement_timeout_ms=60000)


def test_ordering_is_not_enforced_when_either_timeout_is_off() -> None:
    assert GatewayConfig(db_command_timeout=0, db_statement_timeout_ms=1000)
    assert GatewayConfig(db_command_timeout=90, db_statement_timeout_ms=0)


def test_timeouts_are_individually_disablable() -> None:
    args = _pg_kwargs(db_command_timeout=0, db_statement_timeout_ms=0)["connect_args"]
    assert "command_timeout" not in args
    assert "server_settings" not in args


def test_existing_server_settings_are_preserved() -> None:
    # A deployment that already passes server settings through its URL keeps
    # them, and its own statement_timeout wins over the configured default.
    connect_args: dict[str, Any] = {
        "server_settings": {"application_name": "otari", "statement_timeout": "5000"}
    }
    args = engine_kwargs(GatewayConfig(), connect_args=connect_args, is_sqlite=False)["connect_args"]
    assert args["server_settings"]["application_name"] == "otari"
    assert args["server_settings"]["statement_timeout"] == "5000"


def test_the_callers_connect_args_are_never_mutated() -> None:
    # Two engines are built from one parsed URL, and the first has captured the
    # reference by the time the second is built.
    connect_args: dict[str, Any] = {"ssl": "require"}
    first = engine_kwargs(GatewayConfig(), connect_args=connect_args, is_sqlite=False)
    second = engine_kwargs(
        GatewayConfig(), connect_args=connect_args, is_sqlite=False, pool_size=5, max_overflow=0
    )
    assert connect_args == {"ssl": "require"}
    assert first["connect_args"] is not second["connect_args"]
    assert first["connect_args"] is not connect_args


def test_sqlite_stays_on_nullpool_and_takes_no_timeouts() -> None:
    connect_args: dict[str, Any] = {"check_same_thread": False}
    kwargs = engine_kwargs(GatewayConfig(), connect_args=connect_args, is_sqlite=True)
    assert kwargs["poolclass"] is NullPool
    assert "pool_size" not in kwargs
    # asyncpg keywords would be rejected by aiosqlite.
    assert kwargs["connect_args"] == {"check_same_thread": False}


def test_secondary_pool_overrides_the_request_pool_sizes() -> None:
    # What the usage-log writer's engine asks for: a few reserved connections
    # and no overflow, so metering cannot be starved by request traffic and
    # cannot burst against it either.
    kwargs = engine_kwargs(
        GatewayConfig(), connect_args={}, is_sqlite=False, pool_size=5, max_overflow=0
    )
    assert kwargs["pool_size"] == 5
    assert kwargs["max_overflow"] == 0
