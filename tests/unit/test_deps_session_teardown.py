"""Regression coverage for the pooled connection leak in ``get_db_if_needed``.

The dependency used to consume ``get_db`` with a bare ``async for``. An
``async for`` never closes the iterator it drives, so tearing the dependency
down abandoned ``get_db``'s generator with its ``async with`` block unfinished.
The session was then closed only when the garbage collector finalized that
generator, at an arbitrary later moment in an unrelated task, leaving its
pooled connection checked out until then. Under sustained traffic that outran
the pool and every request failed with a ``QueuePool`` timeout.
"""

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

    await dependency.aclose()

    assert session.closed, "tearing the dependency down must return the connection to the pool"


@pytest.mark.asyncio
async def test_teardown_closes_the_session_when_the_request_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The connection goes back even when the handler raises, which is the path a
    cancelled or erroring request takes."""
    session = _FakeSession()
    monkeypatch.setattr(database, "_SessionLocal", lambda: session)

    dependency = get_db_if_needed(GatewayConfig(database_url="sqlite:///./test.db"))
    await anext(dependency)

    with pytest.raises(RuntimeError):
        await dependency.athrow(RuntimeError("handler blew up"))

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
