"""Unit tests for the connection-pool stats helper and its Prometheus collector."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.core import database
from gateway.core.database import LOG_POOL, REQUEST_POOL, PoolStats, pool_stats
from gateway.metrics import REGISTRY


class _QueuePool:
    """Stand-in for ``AsyncAdaptedQueuePool`` exposing the counters read here."""

    def __init__(self, checked_out: int, checked_in: int, overflow: int, size: int) -> None:
        self._checked_out = checked_out
        self._checked_in = checked_in
        self._overflow = overflow
        self._size = size

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
    overflow = dict(database._pool_max_overflow)
    yield
    database._engine = engine
    database._log_engine = log_engine
    database._pool_max_overflow.clear()
    database._pool_max_overflow.update(overflow)


def test_pool_stats_reads_the_queue_pool_counters() -> None:
    database._engine = _engine(_QueuePool(checked_out=7, checked_in=3, overflow=2, size=10))
    database._log_engine = None
    database._pool_max_overflow[REQUEST_POOL] = 20

    readings = pool_stats()

    assert readings[REQUEST_POOL] == PoolStats(checked_out=7, checked_in=3, overflow=2, size=10, max_overflow=20)
    assert readings[REQUEST_POOL].capacity == 30


def test_capacity_uses_the_configured_overflow_not_the_pool_object() -> None:
    """The gateway chose ``max_overflow``, so capacity comes from the config, not a private field."""
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=1, overflow=0, size=10))
    database._log_engine = None
    database._pool_max_overflow[REQUEST_POOL] = 5

    assert pool_stats()[REQUEST_POOL].capacity == 15


def test_pool_stats_clamps_a_negative_overflow() -> None:
    """A pool that has not created its base connections yet reports a negative overflow."""
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=0, overflow=-9, size=10))
    database._log_engine = None

    assert pool_stats()[REQUEST_POOL].overflow == 0


def test_pool_stats_omits_a_null_pool() -> None:
    database._engine = _engine(_NullPool())
    database._log_engine = None

    assert pool_stats() == {}


def test_pool_stats_is_empty_when_the_database_is_not_initialized() -> None:
    database._engine = None
    database._log_engine = None

    assert pool_stats() == {}


def test_pool_stats_covers_both_engines() -> None:
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=2, overflow=0, size=10))
    database._log_engine = _engine(_QueuePool(checked_out=0, checked_in=1, overflow=0, size=2))
    database._pool_max_overflow[REQUEST_POOL] = 20
    database._pool_max_overflow[LOG_POOL] = 0

    readings = pool_stats()

    assert set(readings) == {REQUEST_POOL, LOG_POOL}
    assert readings[LOG_POOL].capacity == 2


def test_pool_stats_omits_an_engine_without_a_pool() -> None:
    database._engine = _engine(_QueuePool(checked_out=1, checked_in=2, overflow=0, size=10))
    database._log_engine = _engine(_NullPool())

    assert set(pool_stats()) == {REQUEST_POOL}


def test_the_collector_publishes_the_live_pool_on_every_scrape() -> None:
    database._engine = _engine(_QueuePool(checked_out=4, checked_in=6, overflow=1, size=10))
    database._log_engine = None
    database._pool_max_overflow[REQUEST_POOL] = 20

    labels = {"pool": REQUEST_POOL}
    assert REGISTRY.get_sample_value("gateway_db_pool_connections_checked_out", labels) == 4.0
    assert REGISTRY.get_sample_value("gateway_db_pool_connections_idle", labels) == 6.0
    assert REGISTRY.get_sample_value("gateway_db_pool_overflow_connections", labels) == 1.0
    assert REGISTRY.get_sample_value("gateway_db_pool_capacity", labels) == 30.0

    database._engine = _engine(_QueuePool(checked_out=9, checked_in=1, overflow=0, size=10))

    assert REGISTRY.get_sample_value("gateway_db_pool_connections_checked_out", labels) == 9.0


def test_the_collector_publishes_no_series_without_a_pool() -> None:
    """SQLite runs on NullPool, and an uninitialized engine has no pool at all."""
    database._engine = None
    database._log_engine = None

    assert REGISTRY.get_sample_value("gateway_db_pool_capacity", {"pool": REQUEST_POOL}) is None
    assert REGISTRY.get_sample_value("gateway_db_pool_capacity", {"pool": LOG_POOL}) is None
