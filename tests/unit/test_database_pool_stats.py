"""Unit tests for the connection-pool stats helper and the readiness probe."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from gateway.api.routes import health
from gateway.core import database
from gateway.core.database import LOG_POOL, REQUEST_POOL, PoolStats, pool_stats, request_pool_stats


class _QueuePool:
    """Stand-in for ``AsyncAdaptedQueuePool`` exposing the counters read here."""

    def __init__(self, checked_out: int, checked_in: int, overflow: int, size: int, max_overflow: int) -> None:
        self._checked_out = checked_out
        self._checked_in = checked_in
        self._overflow = overflow
        self._size = size
        self._max_overflow = max_overflow

    def checkedout(self) -> int:
        return self._checked_out

    def checkedin(self) -> int:
        return self._checked_in

    def overflow(self) -> int:
        return self._overflow

    def size(self) -> int:
        return self._size


class _NullPool:
    """Stand-in for ``NullPool``, which implements none of those counters."""

    def status(self) -> str:
        return "NullPool"


def _engine(pool: Any) -> Any:
    return SimpleNamespace(pool=pool)


@pytest.fixture(autouse=True)
def _restore_engines() -> Iterator[None]:
    engine = database._engine
    log_engine = database._log_engine
    yield
    database._engine = engine
    database._log_engine = log_engine


def test_pool_stats_reads_the_queue_pool_counters() -> None:
    database._engine = _engine(_QueuePool(checked_out=7, checked_in=3, overflow=2, size=10, max_overflow=20))
    database._log_engine = None

    stats = request_pool_stats()

    assert stats == PoolStats(checked_out=7, checked_in=3, overflow=2, size=10, max_overflow=20)
    assert stats is not None
    assert stats.capacity == 30
    assert not stats.is_saturated


def test_pool_stats_clamps_a_negative_overflow() -> None:
    """A pool that has not created its base connections yet reports a negative overflow."""
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=0, overflow=-9, size=10, max_overflow=20))

    stats = request_pool_stats()

    assert stats is not None
    assert stats.overflow == 0


def test_pool_is_saturated_when_every_connection_is_out() -> None:
    database._engine = _engine(_QueuePool(checked_out=30, checked_in=0, overflow=20, size=10, max_overflow=20))

    stats = request_pool_stats()

    assert stats is not None
    assert stats.is_saturated


def test_pool_stats_returns_none_for_a_null_pool() -> None:
    database._engine = _engine(_NullPool())

    assert request_pool_stats() is None


def test_pool_stats_returns_none_when_the_database_is_not_initialized() -> None:
    database._engine = None
    database._log_engine = None

    assert request_pool_stats() is None
    assert pool_stats() == {}


def test_pool_stats_covers_both_engines() -> None:
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=2, overflow=0, size=10, max_overflow=20))
    database._log_engine = _engine(_QueuePool(checked_out=0, checked_in=1, overflow=0, size=2, max_overflow=0))

    readings = pool_stats()

    assert set(readings) == {REQUEST_POOL, LOG_POOL}
    assert readings[LOG_POOL].capacity == 2


def test_pool_stats_omits_an_engine_without_a_pool() -> None:
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=2, overflow=0, size=10, max_overflow=20))
    database._log_engine = _engine(_NullPool())

    assert set(pool_stats()) == {REQUEST_POOL}


class _Session:
    """Session that records whether the readiness query was attempted."""

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, _statement: Any) -> None:
        self.executed = True


# The route only reads ``is_hybrid_mode``, so a stand-in keeps this a unit test.
_STANDALONE: Any = SimpleNamespace(is_hybrid_mode=False)


@pytest.mark.asyncio
async def test_readiness_refuses_immediately_on_a_saturated_pool() -> None:
    database._engine = _engine(_QueuePool(checked_out=30, checked_in=0, overflow=20, size=10, max_overflow=20))
    session: Any = _Session()

    with pytest.raises(HTTPException) as excinfo:
        await health.health_readiness(config=_STANDALONE, db=session)

    assert excinfo.value.status_code == 503
    detail: Any = excinfo.value.detail
    assert detail["status"] == "unhealthy"
    assert detail["database"] == health.POOL_EXHAUSTED
    assert detail["database"] != "unavailable"
    assert "version" in detail
    assert not session.executed


@pytest.mark.asyncio
async def test_readiness_queries_the_database_when_the_pool_has_headroom() -> None:
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=9, overflow=0, size=10, max_overflow=20))
    session: Any = _Session()

    payload = await health.health_readiness(config=_STANDALONE, db=session)

    assert payload["status"] == "healthy"
    assert payload["database"] == "connected"
    assert session.executed


@pytest.mark.asyncio
async def test_readiness_queries_the_database_on_a_null_pool() -> None:
    database._engine = _engine(_NullPool())
    session: Any = _Session()

    payload = await health.health_readiness(config=_STANDALONE, db=session)

    assert payload["status"] == "healthy"
    assert session.executed
