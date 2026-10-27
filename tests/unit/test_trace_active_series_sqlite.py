"""The active-sessions series counts on SQLite, the engine a laptop runs, not only on PostgreSQL.

The integration suite runs on PostgreSQL. The series query joins a table of buckets
it writes itself, and SQLite accepts less of that syntax, so it is run here on SQLite.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import Trace
from gateway.repositories.traces import TracesRepositories
from gateway.services.traces import TraceService
from gateway.types.traces import TraceBucketGrain, TraceFilter, TraceScope


def _no_window(provider: str | None, model: str) -> int | None:
    return None


_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)


def _trace(trace_id: str, *, start: int, end: int, failed: bool = False) -> Trace:
    return Trace(
        workspace_id=uuid.uuid4(),
        trace_id=trace_id,
        session_source="otlp",
        span_count=1,
        started_at=_T0 + timedelta(minutes=start),
        last_activity_at=_T0 + timedelta(minutes=end),
        error_count=1 if failed else 0,
        failing_error_count=1 if failed else 0,
        cost_snapshot=Decimal(0),
    )


async def _series(*traces: Trace, filters: TraceFilter, bucket: TraceBucketGrain) -> list[tuple[str, int, int]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            session.add_all(traces)
            await session.commit()
            uow = UnitOfWork(session)
            series = await TraceService(uow, TracesRepositories.on(uow), _no_window).series(
                TraceScope.deployment(), filters, bucket=bucket
            )
    finally:
        await engine.dispose()
    return [(point.bucket, point.succeeded, point.failed) for point in series]


@pytest.mark.asyncio
async def test_a_session_counts_in_every_bucket_it_was_active_in_on_sqlite() -> None:
    series = await _series(
        _trace("long", start=0, end=35),
        _trace("short", start=50, end=51, failed=True),
        filters=TraceFilter(),
        bucket="30m",
    )

    assert series == [("2026-10-07T12:00:00Z", 1, 0), ("2026-10-07T12:30:00Z", 1, 1)]


@pytest.mark.asyncio
async def test_a_session_still_running_past_the_window_counts_inside_it_on_sqlite() -> None:
    window = TraceFilter(start=_T0 + timedelta(minutes=10), end=_T0 + timedelta(minutes=20))

    series = await _series(_trace("long", start=0, end=90), filters=window, bucket="5m")

    assert [bucket for bucket, _, _ in series] == ["2026-10-07T12:10:00Z", "2026-10-07T12:15:00Z"]
