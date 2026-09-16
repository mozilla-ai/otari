"""Teardown of ``get_db_if_needed`` closes its session whether the request succeeds, fails or is canceled."""

import asyncio
from types import TracebackType
from typing import Any

import pytest

from gateway.api.deps import get_db_if_needed
from gateway.core import database
from gateway.core.config import GatewayConfig


class _FakeSession:
    """Stands in for ``AsyncSession``, recording whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_teardown_closes_the_session_it_borrowed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()
    monkeypatch.setattr(database, "_SessionLocal", lambda: session)

    dependency = get_db_if_needed(GatewayConfig(database_url="sqlite:///./test.db"))
    yielded: object = await anext(dependency)
    assert yielded is session
    assert not session.closed, "the session must stay open for the life of the request"

    with pytest.raises(StopAsyncIteration):
        await anext(dependency)

    assert session.closed, "tearing the dependency down must return the connection to the pool"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("handler blew up"), asyncio.CancelledError()], ids=["error", "canceled"]
)
async def test_teardown_closes_the_session_when_the_request_fails(
    error: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The connection goes back when the request raises or is canceled."""
    session = _FakeSession()
    monkeypatch.setattr(database, "_SessionLocal", lambda: session)

    dependency = get_db_if_needed(GatewayConfig(database_url="sqlite:///./test.db"))
    await anext(dependency)

    with pytest.raises(type(error)):
        await dependency.athrow(error)

    assert session.closed


@pytest.mark.asyncio
async def test_hybrid_mode_yields_no_session_and_opens_no_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unreachable() -> Any:
        raise AssertionError("hybrid mode must not open a local session")

    monkeypatch.setattr(database, "_SessionLocal", _unreachable)
    config = GatewayConfig(mode="hybrid", platform={"base_url": "https://example.invalid", "token": "t"})

    dependency = get_db_if_needed(config)
    assert await anext(dependency) is None
    await dependency.aclose()
