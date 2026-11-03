"""A failed database write is logged by what failed, never by the values it carried."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
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
from gateway.services.inference import _country_database

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


def test_the_stored_country_file_logs_a_database_error_without_its_values(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At startup and on each refresh, a failed read of the stored file is logged by what failed."""
    exc = _duplicate_key_error()
    calls = 0

    @contextlib.asynccontextmanager
    async def no_unit_of_work() -> AsyncIterator[None]:
        yield None

    async def refresh(_: object) -> bool:
        nonlocal calls
        calls += 1
        if calls > 2:
            raise asyncio.CancelledError
        raise exc

    monkeypatch.setattr(_country_database, "create_unit_of_work", no_unit_of_work)
    monkeypatch.setattr(_country_database, "refresh_stored_country_database", refresh)
    caplog.set_level(logging.WARNING, logger="gateway")

    asyncio.run(_country_database.load_stored_country_database_at_startup(GatewayConfig()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_country_database.run_stored_country_database_refresher(interval=0))

    records = [r for r in caplog.records if r.name == "gateway"]
    assert [r.levelno for r in records] == [logging.ERROR, logging.WARNING]
    assert all("IntegrityError" in r.getMessage() and r.exc_info is None for r in records)
    assert _SECRET not in caplog.text
