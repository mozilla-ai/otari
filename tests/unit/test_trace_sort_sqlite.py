"""The session list's sort on SQLite, where timestamps are text and latency cannot be a plain subtraction."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.traces import TracesRepositories
from gateway.services.traces import TraceService
from gateway.types.traces import SpanRecord, TraceFilter, TraceScope, TraceSort, TraceWrite


def _no_window(provider: str | None, model: str) -> int | None:
    return None


pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
_WORKSPACE = uuid.uuid4()


@pytest_asyncio.fixture
async def store() -> AsyncIterator[TraceService]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            yield TraceService(uow, TracesRepositories.on(uow), _no_window)
    finally:
        await engine.dispose()


def _lasting(trace_id: str, seconds: int) -> TraceWrite:
    span = SpanRecord(
        span_id=f"{trace_id}-span",
        kind="step",
        origin="gateway",
        name="step",
        outcome="ok",
        start_time=_T0,
        end_time=_T0 + timedelta(seconds=seconds),
    )
    return TraceWrite(
        workspace_id=_WORKSPACE,
        trace_id=trace_id,
        user_id=None,
        api_key_id=None,
        session_source="harness",
        harness="claude-code",
        spans=(span,),
    )


async def test_the_list_sorts_by_latency(store: TraceService) -> None:
    await store.write((_lasting("minute", 60), _lasting("hour", 3600), _lasting("second", 1)))

    page = await store.search(TraceScope.deployment(), TraceFilter(), limit=10, offset=0, sort=TraceSort(key="latency"))

    assert [item.trace_id for item in page.items] == ["hour", "minute", "second"]
