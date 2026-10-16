"""A failed database write is logged by what failed, never by the values it carried."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

import gateway
from gateway.api.routes._helpers import database_error
from gateway.core.config import GatewayConfig
from gateway.core.database import describe_database_error, engine_kwargs

_SECRET = "sk-live-do-not-log"


def _duplicate_key_error(**engine_options: Any) -> IntegrityError:
    """A real constraint failure whose statement carried a secret."""

    async def main() -> IntegrityError:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:", **engine_options)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE TABLE creds (name TEXT PRIMARY KEY, client_args TEXT)"))
                await conn.execute(text("INSERT INTO creds VALUES ('openai', 'x')"))
                try:
                    await conn.execute(text("INSERT INTO creds VALUES ('openai', :args)"), {"args": _SECRET})
                except IntegrityError as exc:
                    return exc
        finally:
            await engine.dispose()
        raise AssertionError("the duplicate insert was not refused")

    return asyncio.run(main())


def test_description_names_the_failure_and_leaves_out_the_values() -> None:
    exc = _duplicate_key_error()
    assert _SECRET in str(exc), "the message carries the parameters, which is why it is not logged"

    described = describe_database_error(exc)

    assert _SECRET not in described
    assert described.startswith("IntegrityError")
    assert "sqlstate=SQLITE_CONSTRAINT_PRIMARYKEY" in described


def test_the_gateways_engines_keep_parameters_out_of_every_error_message() -> None:
    """Covers what no handler sees: an error that escapes a route is logged whole by the server."""
    options = engine_kwargs(GatewayConfig(), connect_args={}, is_sqlite=True)

    assert _SECRET not in str(_duplicate_key_error(**options))


def test_a_connect_timeout_is_described_by_its_class() -> None:
    assert describe_database_error(TimeoutError()) == "TimeoutError"


def test_database_error_logs_the_operation_and_answers_the_generic_500(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="gateway")

    raised = database_error("create stored provider", _duplicate_key_error())

    assert isinstance(raised, HTTPException)
    assert (raised.status_code, raised.detail) == (500, "Database error")
    [record] = [r for r in caplog.records if r.name == "gateway"]
    assert record.getMessage().startswith("create stored provider failed: IntegrityError")
    assert record.exc_info is None
    assert _SECRET not in caplog.text


def test_only_the_helper_answers_database_error() -> None:
    """A route that built the 500 itself would skip the log line, or log the values."""
    package = Path(gateway.__file__).parent
    built = [
        str(path.relative_to(package))
        for path in sorted(package.rglob("*.py"))
        for line in path.read_text().splitlines()
        if 'detail="Database error"' in line
    ]
    assert built == ["api/routes/_helpers.py"]
