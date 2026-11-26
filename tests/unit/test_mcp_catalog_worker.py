"""The MCP catalog sync worker's poll: what it does with the sync off, and what it logs when a sync fails."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from sqlalchemy.exc import OperationalError

from gateway.core.config import GatewayConfig
from gateway.services.mcp_catalog import McpCatalogService, _service

# Read at import: the root conftest replaces the poll with a no-op for the length of every test.
_poll = McpCatalogService.poll


class _Stop(BaseException):
    """Ends the worker's endless loop from inside a test."""


@asynccontextmanager
async def _no_database() -> AsyncIterator[object]:
    yield object()


async def _run(monkeypatch: pytest.MonkeyPatch, config: GatewayConfig, failures: list[BaseException]) -> int:
    """Run the worker until ``failures`` is used up, answering how many times it built a service."""
    built = 0

    def build(_uow: Any) -> Any:
        nonlocal built
        built += 1
        raise failures.pop(0)

    monkeypatch.setattr(_service, "create_unit_of_work", _no_database)
    with pytest.raises(_Stop):
        await _poll(build, config, interval=0)
    return built


@pytest.mark.asyncio
async def test_a_database_failure_is_logged_without_its_statement_or_a_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    failure = OperationalError("UPDATE mcp_catalog_sync_state SET cursor=?", {"cursor": "secret-cursor"}, Exception())
    caplog.set_level(logging.WARNING)

    built = await _run(monkeypatch, GatewayConfig(mcp_catalog_sync="manual"), [failure, _Stop()])

    assert built == 2
    [record] = [r for r in caplog.records if "MCP catalog sync" in r.getMessage()]
    assert record.exc_info is None
    assert "OperationalError" in record.getMessage()
    assert "secret-cursor" not in record.getMessage()
    assert "UPDATE" not in record.getMessage()


@pytest.mark.asyncio
async def test_any_other_failure_is_logged_by_its_type_and_the_poll_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    built = await _run(monkeypatch, GatewayConfig(mcp_catalog_sync="manual"), [RuntimeError("boom"), _Stop()])

    assert built == 2
    [record] = [r for r in caplog.records if "MCP catalog sync" in r.getMessage()]
    assert record.exc_info is None
    assert "RuntimeError" in record.getMessage()


@pytest.mark.asyncio
async def test_the_poll_touches_nothing_while_the_sync_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GatewayConfig(mcp_catalog_sync="off")
    polls = 0

    async def count_then_stop(_seconds: float) -> None:
        nonlocal polls
        polls += 1
        if polls == 3:
            raise _Stop

    def never(_uow: Any) -> Any:
        raise AssertionError("a service was built while the sync is off")

    monkeypatch.setattr(asyncio, "sleep", count_then_stop)
    monkeypatch.setattr(_service, "create_unit_of_work", never)
    with pytest.raises(_Stop):
        await _poll(never, config, interval=0)

    assert polls == 3
